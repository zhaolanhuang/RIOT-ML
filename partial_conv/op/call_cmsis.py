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

def cpu_arm_convolve_wrapper_s8_rule(op):
    return tvm.tir.call_pure_extern("int32", "arm_convolve_wrapper_s8_", *op.args)

register_op_attr("tir.arm_convolve_wrapper_s8_", "TCallEffectKind", tvm.tir.CallEffectKind.UpdateState) # Use UpdateState mark to prevent optimized out
# register_intrin_lowering("tir.arm_convolve_wrapper_s8_", target="default", f=cpu_arm_convolve_wrapper_s8_rule, level=10)
register_op_attr('tir.arm_convolve_wrapper_s8_', "TGlobalSymbol", "arm_convolve_wrapper_s8_")

def cpu_arm_depthwise_conv_wrapper_s8_rule(op):
    return tvm.tir.call_pure_extern("int32", "arm_depthwise_conv_wrapper_s8_", *op.args)

register_op_attr("tir.arm_depthwise_conv_wrapper_s8_", "TCallEffectKind", tvm.tir.CallEffectKind.UpdateState) # Use UpdateState mark to prevent optimized out
# register_intrin_lowering("tir.arm_depthwise_conv_wrapper_s8_", target="default", f=cpu_arm_depthwise_conv_wrapper_s8_rule, level=10)
register_op_attr('tir.arm_depthwise_conv_wrapper_s8_', "TGlobalSymbol", "arm_depthwise_conv_wrapper_s8_")

# Define the new operator in Relay
op_name = "call_cmsis"
relay.op.op.register(op_name)

# _op.get(op_name).set_num_inputs(1)
# _op.get(op_name).add_argument("func", "Function", "The input data tensor.")
# _op.get(op_name).add_argument("data_1", "Tensor", "The input data tensor.")
# call default relation functions
def call_cmsis_rel(args, attrs):
    return attrs["call_type"]
_op.get(op_name).add_type_rel("CallCMSISTypeRel", call_cmsis_rel) # -> Key for TypeInference


_op.get(op_name).set_support_level(1)
_op.register_pattern(op_name, _op.OpPattern.ELEMWISE)
_op.register_stateful(op_name, True)


# def iter_func(iter_begin, iter_end, iter_strides, func_args, func):
#     attrs = tvm.ir.make_node("DictAttrs", iter_begin=iter_begin,iter_end=iter_end, iter_strides=iter_strides, relay_func=func)
#     # breakpoint()
#     return relay.Call(relay.op.get("iter_func"), func_args, attrs)

_GLOBAL_NAME_TO_OP = {}

def save_cmsisnn_op(name, op):
    _GLOBAL_NAME_TO_OP[name] = op

dtype_bytes = {"int32" : 4, "float32" : 4}

#INPUT LAYOUT: NCHW
#FILTER LAYOUT: OIHW

def is_depthwise(conv_attrs, input_shape, filter_shape):
    return (conv_attrs.channels == filter_shape[0] * filter_shape[1])


# Define the compute function for the my_add operator
def wrap_call_cmsis_compute_tir(attrs, inputs, output_type):
    func = _GLOBAL_NAME_TO_OP[attrs["global_var"].name_hint]
    conv_op = func.body.op.body
    conv_attrs =  conv_op.attrs
    input_shape = conv_op.type_args[0].shape
    filter_shape = conv_op.type_args[1].shape
    output_shape = conv_op.checked_type.shape
    depth_multiplier = -1
    if is_depthwise(conv_attrs, input_shape, filter_shape):
        kernel_pos_dm = 0 if input_shape[1] == 1 else 1
        depth_multiplier = filter_shape[kernel_pos_dm]
    _is_depthwise = depth_multiplier != -1

    input_nhwc = [input_shape[0], input_shape[2], input_shape[3], input_shape[1]]
    output_nhwc = [output_shape[0], output_shape[2], output_shape[3], output_shape[1]]
    filter_ohwi = [filter_shape[0], filter_shape[2], filter_shape[3], filter_shape[1]]
    stride = conv_attrs.strides[0]
    padding = conv_attrs.padding[0]
    dilation = conv_attrs.dilation[1]
    
    def _call_cmsis_compute_tir(ins, outs):
        ins_data = [i.data for i in ins]
        ext_name = attrs["global_var"].name_hint
        ib = tvm.tir.ir_builder.create()
        if(_is_depthwise):
            ib.emit(tvm.tir.call_intrin("int32", 
                "tir.arm_depthwise_conv_wrapper_s8_",
                    *ins_data, outs[0].data, *input_nhwc, *filter_ohwi, *output_nhwc,
                    stride, padding, dilation, depth_multiplier
                    ))
        else:
            ib.emit(tvm.tir.call_intrin("int32", 
                "tir.arm_convolve_wrapper_s8_",
                    *ins_data, outs[0].data, *input_nhwc, *filter_ohwi, *output_nhwc,
                    stride, padding, dilation
                    ))
        return ib.get()
    
    return _call_cmsis_compute_tir

@relay.op.op.register_compute(op_name)
def call_cmsis_compute(attrs, inputs, output_type):
    # print("We are now at iter_func_comp")
    return [tvm.te.extern(output_type.shape, inputs,
               wrap_call_cmsis_compute_tir(attrs, inputs, output_type),
            name=op_name, dtype=output_type.dtype)
]

def wrap_topi_schedule(topi_schedule):
    """Wrap TOPI schedule which doesn't use attrs"""

    def wrapper(attrs, outs, target):
        with target:
            return topi_schedule(outs)

    return wrapper

@override_native_generic_func(f"{op_name}_strategy")
def call_cmsis_strategy(attrs, inputs, out_type, target):
    strategy = _op.OpStrategy()
    strategy.add_implementation(
        call_cmsis_compute,
        wrap_topi_schedule(topi.generic.schedule_extern),
        name=f"{op_name}.generic",
    )
    return strategy
_op.register_strategy(op_name, call_cmsis_strategy)

