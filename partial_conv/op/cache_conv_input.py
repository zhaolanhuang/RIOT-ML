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

from .global_identifier import get_fusion_worker_id, get_fusion_op_num, increase_fusion_op_num
from .macro_globalvar import invoke_c_macro, CHECK_IF_SKIP_COMPUTE, DECLEAR_EXTERN_WAIT_FOR_VAR, INSERT_STUB


# Define the new operator in Relay
op_name = "cache_conv_input"
relay.op.op.register(op_name)


# call default relation functions
def cache_conv_input_rel(args, attrs):
    return args[1]
_op.get(op_name).add_type_rel("CacheConvInputTypeRel", cache_conv_input_rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, True)


def cache_conv_input(data, buffer_shape, max_idx, conv_kernel_size, conv_strides, conv_padding, conv_dtype, conv_input_shape,layout="NCHW"):
    attrs = tvm.ir.make_node("DictAttrs", buffer_shape=buffer_shape, max_idx=max_idx, 
                             conv_kernel_size=conv_kernel_size, conv_strides=conv_strides, conv_padding=conv_padding,
                             conv_input_shape=conv_input_shape
                             )
    buffer_var = relay.var("cache_buffer_var", shape=buffer_shape, dtype=conv_dtype)
    current_idx = relay.var("cache_cur_idx", shape=(4,), dtype="int32") # NCHW
    return relay.Call(relay.op.get(op_name), [data, buffer_var, current_idx], attrs)


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper


from tvm.topi import utils

@relay.op.op.register_compute(op_name)
def _compute(attrs, inputs, output_type):
    print("We are now at cache_conv_input_compute")  
    
    data, buffer_var, current_idx = inputs

    # bg_x = utils.get_const_int(bg_x)
    # bg_y = utils.get_const_int(bg_y)
    max_idx = attrs["max_idx"]
    buffer_shape = attrs["buffer_shape"]
    kernel_size = attrs["conv_kernel_size"]
    strides = attrs["conv_strides"]
    padding = attrs["conv_padding"]
    input_shape = attrs["conv_input_shape"]
    
    print("kernel_size:", kernel_size)
    print("input_shape:", input_shape)
    print("padding:", padding)
    print("strides:", strides)
    worker_id = get_fusion_worker_id()
    cur_op_num = get_fusion_op_num()
    print("worker_id:", worker_id, "op_num:", cur_op_num)

    def gen_ib(data_buf, buffer_var_buf, current_idx_buf, out_buf):
        ib = tvm.tir.ir_builder.create()
        ib.emit(invoke_c_macro(INSERT_STUB, worker_id, cur_op_num))
        ib.emit(invoke_c_macro(DECLEAR_EXTERN_WAIT_FOR_VAR, worker_id))
        ib.emit(invoke_c_macro(CHECK_IF_SKIP_COMPUTE, worker_id, cur_op_num, cur_op_num+1))
        data_w = data_buf.shape[3]
        data = ib.buffer_ptr(data_buf)
        buffer = ib.buffer_ptr(buffer_var_buf)
        cur_idx = ib.buffer_ptr(current_idx_buf)
        out = ib.buffer_ptr(out_buf)
        #shrift elements inside buffer
        # with ib.for_range(0 , buffer_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
        #     with ib.for_range(0,  buffer_shape[1], "i") as i:
        #         with ib.for_range(0, buffer_shape[2], "j") as j:
        #             with ib.for_range(0, buffer_shape[3] - data_w, "k") as k:
        #                 buffer[n , i , j , k] = buffer[n, i , j , k + data_w]

        # Copy new data to buffer
        with ib.for_range(0 , buffer_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
            with ib.for_range(0,  buffer_shape[1], "i") as i:
                with ib.for_range(0, buffer_shape[2], "j") as j:
                    with ib.for_range(0, data_w, "k") as k:
                        buffer[n , i , j , cur_idx[3] % (buffer_shape[3] - data_w + k)] = data[n, i , j , k]

        cur_idx[3] = cur_idx[3] + data_w

        with ib.if_scope(tvm.tir.all((cur_idx[3] + 1) >= kernel_size[1], ((cur_idx[3] + 1 - kernel_size[1]) % strides[1]) == 0)):
            # copy to Output buffer
            with ib.for_range(0 , buffer_shape[0], "n") as n: # begin of for loop has to be zero for c codegen...
                with ib.for_range(0,  buffer_shape[1], "i") as i:
                    with ib.for_range(0, buffer_shape[2], "j") as j:
                        with ib.for_range(0, buffer_shape[3], "k") as k:
                            out[n , i , j , k] = buffer[n, i , j , k]

        with ib.else_scope():
            # skip output
            ib.emit(tvm.tir.ret(-1))

        return ib.get()
    increase_fusion_op_num()
    out_ib = [tvm.te.extern(output_type.shape, [data, buffer_var, current_idx],
#                lambda ins, outs: gen_ib(ins[0], ins[1], ins[2]),
              lambda ins, outs: gen_ib(ins[0], ins[1], ins[2], outs[0]),
            name="cache_conv_input_compute.generic", 
            # in_buffers=input_placeholders[1:],
            # out_buffers=[input_placeholders[0]],
            dtype=output_type.dtype,
            # tag='inplace'
            )]
    return out_ib

@override_native_generic_func(op_name + "_strategy")
def _strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        _compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="iter_func.generic",
    )
    return strategy
_op.register_strategy(op_name, _strategy)


if __name__ == "__main__":
    data = relay.var("data", shape=(1, 16, 7, 1), dtype="int32")

    body = cache_conv_input(data, [1,32,7,3], [16,16], [3,3], [2,2], [0,0], "int32")

    func = relay.Function(relay.analysis.free_vars(body), body)

    mod = tvm.IRModule.from_expr(func)
    # mod["iteratee"] = func
    breakpoint()


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

    export_model_library_format(module, "./test_cache_conv_input.tar")