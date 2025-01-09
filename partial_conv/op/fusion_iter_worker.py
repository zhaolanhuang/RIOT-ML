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

# def cpu_memcpy_rule(op):
#     return tvm.tir.call_pure_extern("int32", "memcpy_", *op.args)

# register_op_attr("tir.memcpy", "TCallEffectKind", tvm.tir.CallEffectKind.UpdateState) # Use UpdateState mark to prevent optimized out
# register_intrin_lowering("tir.memcpy", target="default", f=cpu_memcpy_rule, level=10)


# Define the new operator in Relay
relay.op.op.register("fusion_iter_worker")
op_name = "fusion_iter_worker"

# call default relation functions
def fusion_iter_worker_rel(args, attrs):
    func = attrs["relay_func"]
    # return relay.TupleType([relay.TensorType(attrs["output_shape"], args[1].dtype), relay.TensorType(func.body.checked_type.shape,func.body.checked_type.dtype)])
    return relay.TensorType(attrs["output_shape"], args[1].dtype)
_op.get(op_name).add_type_rel("FusionIterWorkerTypeRel", fusion_iter_worker_rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, False)

# Original 07.11
# Fusion Iteratee function: (iterator, iter_var, wrapped_func_args) -> TensorType(shape=iter_var.shape, dtype=iter_var.dtype) # deprecated
# iterator: 1D tensor with shape=(4,), representing the begining indices of input tile in layout (N, C, H, W) of each iteration # deprecated
# iter_var: var that will be change and reuse for each iteration, treated as output var as well (e.g. the aggregated result of the fusion block) # deprecated
# Updated 07.11
# Fusion Iteratee function: (iterator, wrapped_func_args) -> TensorType(shape=wrapped_func_args[0].shape, dtype=rapped_func_args[0].dtype)
# Implicit: wrapped_func_args[0] seen as the input node (tensor) for the fusion block
# iterator and iter_var(output) should be allocated inside iteratee worker, rathen than input from outside
# and the worker will do the in-place update on the output buffer
# wrapped_func_args: other parameters needed by the original fused layers (input, weights, bias etc.)
# Update 08.11
# Trick 1: fetch and use current used te compiler to lower iteratee function, so it will be store in the lowered mods
# Trick 2: Disable the mutex of te_compiler to un-block the lower of iteratee (context: in LowerInternal)
# Trick 3: Disable RemoveUnusedFunction of the remove_standalone_reshapes, so the iteratee wouldn't be optimized out
def fusion_iter_worker(iter_begin, iter_end, iter_strides, output_shape, func_args, func, cache_vars=None):
    attrs = tvm.ir.make_node("DictAttrs", iter_begin=iter_begin,iter_end=iter_end, iter_strides=iter_strides, 
                             relay_func=func, output_shape=output_shape, cache_vars=cache_vars)
    return relay.Call(relay.op.get("fusion_iter_worker"), func_args, attrs)

dtype_bytes = {"int32" : 4, "float32" : 4}


def copy_in_place_spatial_ib(ib, dst_buf, src_buf, begin_indices):
    size_x = src_buf.shape[2]
    size_y = src_buf.shape[3]
    size_c = src_buf.shape[1]
    size_n = src_buf.shape[0]
    dst = ib.buffer_ptr(dst_buf)
    src = ib.buffer_ptr(src_buf)
    indices = ib.buffer_ptr(begin_indices)
    with ib.for_range(0 , size_n, "n") as n: # begin of for loop has to be zero for c codegen...
        with ib.for_range(0,  size_c, "i") as i:
            with ib.for_range(0, size_x, "j") as j:
                with ib.for_range(0, size_y, "k") as k:
                    dst[n + indices[0], i + indices[1], j + indices[2], k+ indices[3]] = src[n, i, j, k]

# Define the compute function for the my_add operator
def wrap_fusion_iter_worker_compute_tir(attrs, inputs, output_type):
    def _fusion_iter_worker_compute_tir(ins, outs):
        ins_data = [i.data for i in ins]
        begin = attrs["iter_begin"]
        end = attrs["iter_end"]
        strides = attrs["iter_strides"]
        func = attrs["relay_func"]
        
        ib = tvm.tir.ir_builder.create()

        iterator = ib.allocate("int32", (4,),  "iterator")
        iterator[0]=iterator[1]=iterator[2]=iterator[3]=0
        iteratee_output = ib.allocate(func.body.checked_type.dtype, func.body.checked_type.shape,  "iteratee_output")
        bg_ind = ib.allocate("int32", (4,),  "bg_ind")
        bg_ind[0]=bg_ind[1]=bg_ind[2]=bg_ind[3]=0
        
        iterator[2] = begin[2]
        with ib.while_loop(iterator[2] < end[2]):
            iterator[3] = begin[3]
            with ib.while_loop(iterator[3] < end[3]):
                ib.emit(tvm.tir.call_extern("int32", 
                    "tvmgen_default_" + func.attrs["global_symbol"],
                    iterator.asobject().data,
                        *ins_data, 
                        iteratee_output.asobject().data))
                # ib.emit(prim_func.body.block)
                bg_ind[2] = iterator[2] // strides[2]
                bg_ind[3] = iterator[3] // strides[3]
                copy_in_place_spatial_ib(ib, outs[0], iteratee_output.asobject(), bg_ind.asobject())
                # ib.emit(tvm.tir.call_intrin("int32", 
                #     "tir.memcpy",
                #         ins_data[1], outs[0].data, math.prod(outs[0].shape) * dtype_bytes[outs[0].dtype]))
                iterator[3] += strides[3]
            iterator[2] += strides[2]
        ib_stmt = ib.get()
        # breakpoint()
        return ib_stmt
    
    return _fusion_iter_worker_compute_tir

def allocate_cache_buffers_for(ib, cache_vars):
    bufs = []
    for cvar in cache_vars:
        buf = ib.allocate(cvar.type_annotation.dtype, cvar.type_annotation.shape, 
                          cvar.name_hint, scope="global"
                          )
        bufs.append(buf)
    return bufs

# Define the compute function for the my_add operator
def wrap_fusion_iter_worker_with_cache_compute_tir(attrs, inputs, output_type):
    def _fusion_iter_worker_compute_tir(ins, outs):
        input_tensors = inputs
        ins_data = [i.data for i in ins]
        begin = attrs["iter_begin"]
        end = attrs["iter_end"]
        strides = attrs["iter_strides"]
        func = attrs["relay_func"]
        cache_vars = attrs["cache_vars"]

        ib = tvm.tir.ir_builder.create()
        
        # Without this dummy block, in some corner cases the USMP analyser can not discover conflicts between io data
        # In that case the allocation of i/o workspaces will overlapped.
        in_data = ib.buffer_ptr(ins[0])
        out_data = ib.buffer_ptr(outs[0])
        in_data[0]=out_data[0]=tvm.runtime.const(0, dtype=in_data.dtype)

        iterator = ib.allocate("int32", (4,),  "iterator", scope="global")
        iterator[0]=iterator[1]=iterator[2]=iterator[3]=0
        iteratee_output = ib.allocate(func.body.checked_type.dtype, func.body.checked_type.shape,  "iteratee_output", scope="global")
        iteratee_output[0] = tvm.runtime.const(0, dtype=iteratee_output.dtype)
        bg_ind = ib.allocate("int32", (4,),  "bg_ind", scope="global")
        bg_ind[0]=bg_ind[1]=bg_ind[2]=bg_ind[3]=0

        cache_bufs = allocate_cache_buffers_for(ib, cache_vars)

        cache_bufs_data = [i.asobject().data for i in cache_bufs]

        def set_cur_idx_var_zero():
            for i in cache_bufs:
                if i.asobject().shape[0] == 4:
                    i[0]=i[1]=i[2]=i[3]=0
                else:
                    i[0] = tvm.runtime.const(0, dtype=i.dtype)
        output_shape = attrs['output_shape']
        
        iterator[2] = begin[2]
        set_cur_idx_var_zero()
        with ib.while_loop(bg_ind[2] < output_shape[2] - 1):
            iterator[3] = begin[3]
            bg_ind[3] = 0
            with ib.while_loop(bg_ind[3] < output_shape[3] - 1):
                with ib.if_scope(tvm.tir.call_extern("int32", 
                        "tvmgen_default_" + func.attrs["global_symbol"],
                        iterator.asobject().data,
                        *ins_data, *cache_bufs_data,
                        iteratee_output.asobject().data) == 0):
                    copy_in_place_spatial_ib(ib, outs[0], iteratee_output.asobject(), bg_ind.asobject())
                    bg_ind[3] = bg_ind[3] + 1
                    with ib.if_scope(bg_ind[3] ==  output_shape[3] - 1):
                        bg_ind[2] = bg_ind[2] + 1                        

                iterator[3] += strides[3]
            # this assignment of value must be put latter than the iteratee function, 
            # so that its liveness in tvm will longer than all the variable in iteratee function.
            # which means it will all `conflict` with vars in iteratee, and wouldnt be accidently overlapped with each other..
            set_cur_idx_var_zero() 
            iterator[2] += strides[2]
        ib_stmt = ib.get()
        # breakpoint()
        return ib_stmt
    
    return _fusion_iter_worker_compute_tir

@relay.op.op.register_compute("fusion_iter_worker")
def fusion_iter_worker_compute(attrs, inputs, output_type):
    print("We are now at fusion_iter_worker_comp")
    func = attrs["relay_func"]
    te_compiler = tvm.relay.backend.te_compiler.current()
    te_compiler.lower(func, tvm.target.Target.current())
    # prim_func = tvm.relay.backend.te_compiler.lower_to_primfunc(func, tvm.target.Target.current())
    # breakpoint()
    # iterator = tvm.te.placeholder((4,), name="iterator", dtype="int32")
    # breakpoint()
    return [tvm.te.extern(attrs["output_shape"], inputs,
               wrap_fusion_iter_worker_with_cache_compute_tir(attrs, inputs, output_type),
            name="fusion_iter_worker", dtype=func.body.checked_type.dtype),
            # tvm.te.extern_primfunc([*inputs], prim_func)
]

def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper

@override_native_generic_func("fusion_iter_worker_strategy")
def fusion_iter_worker_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        fusion_iter_worker_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name="fusion_iter_worker.generic",
    )
    return strategy
_op.register_strategy("fusion_iter_worker", fusion_iter_worker_strategy)


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
    fusion_iter_worker_body = fusion_iter_worker([0,0,0,0], iter_end, [1,1,4,4],[params_cp[0], initial_var,*params_cp[2:]], func)
    # main_body = relay.Function(relay.analysis.free_vars(func_cp), fusion_iter_worker_body)
    mod = tvm.IRModule.from_expr(fusion_iter_worker_body)
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
