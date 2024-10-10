import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay, topi
import tvm.te

from tvm.ir import Attrs
import tvm.relay.op.op as _op
import copy
from tvm.target import generic_func, override_native_generic_func
import math
from tvm.ir import register_intrin_lowering, Op, register_op_attr

def cpu_memcpy_rule(op):
    return tvm.tir.call_pure_extern("int32", "memcpy_", *op.args)

register_op_attr("tir.memcpy", "TCallEffectKind", tvm.tir.CallEffectKind.UpdateState) # Use UpdateState mark to prevent optimized out
register_intrin_lowering("tir.memcpy", target="default", f=cpu_memcpy_rule, level=10)

class IterFuncAttrs(Attrs):
    """"""
    def __init__(self):
        super().__init__()
        self.iterator = None

# Define the new operator in Relay
relay.op.op.register("iter_func")
op_name = "iter_func"
# _op.get(op_name).set_num_inputs(1)
# _op.get(op_name).add_argument("func", "Function", "The input data tensor.")
# _op.get(op_name).add_argument("data_1", "Tensor", "The input data tensor.")
# call default relation functions
def iter_func_rel(args, attrs):
    return args[1]
_op.get(op_name).add_type_rel("IterFuncTypeRel", iter_func_rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, False)


def iter_func(iter_begin, iter_end, iter_strides, func_args, func):
    attrs = tvm.ir.make_node("DictAttrs", iter_begin=iter_begin,iter_end=iter_end, iter_strides=iter_strides, relay_func=func)
    # breakpoint()
    return relay.Call(relay.op.get("iter_func"), func_args, attrs)

dtype_bytes = {"int32" : 4, "float32" : 4}

# Define the compute function for the my_add operator
def wrap_iter_func_compute_tir(attrs, inputs, output_type):
    def _iter_func_compute_tir(ins, outs):
        ins_data = [i.data for i in ins]
        begin = attrs["iter_begin"]
        end = attrs["iter_end"]
        strides = attrs["iter_strides"]
        func = attrs["relay_func"]
        ib = tvm.tir.ir_builder.create()
        iterator = ib.buffer_ptr(ins[0])
        iter_var = ib.buffer_ptr(ins[1])
        iterator[2] = begin[2]
        with ib.while_loop(iterator[2] < end[2]):
            iterator[3] = begin[3]
            with ib.while_loop(iterator[3] < end[3]):
                ib.emit(tvm.tir.call_extern("int32", 
                    "tvmgen_default_" + func.attrs["global_symbol"],
                        *ins_data, outs[0].data))
                ib.emit(tvm.tir.call_intrin("int32", 
                    "tir.memcpy",
                        ins_data[1], outs[0].data, math.prod(outs[0].shape) * dtype_bytes[outs[0].dtype]))
                iterator[3] += strides[3]
            iterator[2] += strides[2]
        return ib.get()
    
    return _iter_func_compute_tir

@relay.op.op.register_compute("iter_func")
def iter_func_compute(attrs, inputs, output_type):
    print("We are now at iter_func_comp")
    lhs = inputs[0]
    return [tvm.te.extern(output_type.shape, inputs,
               wrap_iter_func_compute_tir(attrs, inputs, output_type),
            name="iter_func", dtype=output_type.dtype)
]

def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper

@override_native_generic_func("iter_func_strategy")
def iter_func_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        iter_func_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="iter_func.generic",
    )
    return strategy
_op.register_strategy("iter_func", iter_func_strategy)


if __name__ == "__main__":
    A = relay.var("A", shape=(4,))
    B = relay.var("B", shape=(4,))
    iterator = relay.var("iterator", shape=(4,), dtype="int32")
    iter_end = [1,3,11,11]
    # iterator = [1,1,]
    iter_var = relay.var("iter_var", shape=(4,))
    body = relay.subtract(relay.add(iter_var, A), B)
    params = [iterator, iter_var, A, B]
    params_cp = copy.deepcopy(params)
    func = relay.Function(params, body).with_attr("Primitive", tvm.tir.IntImm("int32", 1)) \
                                    .with_attr("global_symbol", "iteratee") \
    # func_cp = copy.copy(func)
    initial_var = relay.Call(func, params_cp)
    iter_func_body = iter_func([0,0,0,0], iter_end, [1,1,4,4],[params_cp[0], initial_var,*params_cp[2:]], func)
    # main_body = relay.Function(relay.analysis.free_vars(func_cp), iter_func_body)
    mod = tvm.IRModule.from_expr(iter_func_body)
    # mod["iteratee"] = func
    # breakpoint()


    mod = relay.transform.InferType()(mod)


    # breakpoint()

    RUNTIME = tvm.relay.backend.Runtime("crt", {'system-lib':False}) # should not use 'system-lib:true' while AoT
    EXECUTOR = tvm.relay.backend.Executor(
        "aot",
        {
        "unpacked-api": True, 
        "interface-api": "c", 
        "workspace-byte-alignment": 4,
        "link-params": True,
        },
    )
    TARGET = tvm.target.target.micro('host')

    from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll


    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": False, # what is usmp? -> Enable Unified Static Memory Planning
                                                    # "tir.usmp.algorithm": "hill_climb",
                                                    "relay.backend.use_auto_scheduler": True,
                                                    },
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                    ): 
        # print(params.keys())
        # opt_module, _ = relay.optimize(mod, target=TARGET)
        module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)
        # breakpoint()
    from tvm.micro import export_model_library_format

    export_model_library_format(module, "./loop_poc.tar")
