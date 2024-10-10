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

op_name = "dyn_slice_fixed_size"
relay.op.op.register(op_name)
# _op.get(op_name).set_num_inputs(1)
# _op.get(op_name).add_argument("func", "Function", "The input data tensor.")
# _op.get(op_name).add_argument("data_1", "Tensor", "The input data tensor.")
# call default relation functions
def rel(args, attrs):
    return relay.TensorType(attrs["slice_size"], args[0].dtype)
_op.get(op_name).add_type_rel("OpTypeRel", rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, False)


def dyn_slice_fixed_size(sliced_data, slice_begin, slice_size):
    attrs = tvm.ir.make_node("DictAttrs", slice_size=slice_size)
    return relay.Call(relay.op.get("dyn_slice_fixed_size"), [sliced_data, slice_begin], attrs)

dtype_bytes = {"int32" : 4, "float32" : 4}

def wrap_te_fcompute(data, begin, size):
    def te_compute(*i):
        real_indices = [i[j] + begin[j] for j in range(0, len(i))]
        return data(*real_indices)
    return te_compute

@relay.op.op.register_compute(op_name)
def iter_func_compute(attrs, inputs, output_type):
    data = inputs[0]
    begin = inputs[1]
    size = attrs["slice_size"]
    return [tvm.te.compute(size, 
                           wrap_te_fcompute(data,begin,size), 
                           name=op_name)]


def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""
    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper

@override_native_generic_func(op_name + "_strategy")
def iter_func_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        iter_func_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name=op_name + ".generic",
    )
    return strategy
_op.register_strategy(op_name, iter_func_strategy)