import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay, nd
from tvm.relay import dataflow_pattern as dfp
from tvm.relay.transform import function_pass, InferType
from tvm.micro import export_model_library_format
import numpy as np

from model_converter import compile_per_model_eval
from mlmci_utils import generate_mlmci_files
from partial_conv.relay_rewrite import rewrite_conv_chain_to_function

import partial_conv.op.strategy.conv2d
# from tvm.relay.op.strategy.generic import conv2d_strategy
# conv2d_strategy.register("generic", conv2d_strategy.__wrapped__) # use unoptimized conv2d to avoid kernel_vec copy

idx_iter = iter(range(1,100000))

def MBConv_FLOAT32(input ,intput_channel, output_channel, expansion=1, stride=1, padding=(1,1), kernel_size=(3,3)):

    global idx_iter
    i = next(idx_iter)

    weight_1x1_conv2d = relay.var(f"weight_1x1_conv2d_{i}", shape=(intput_channel * expansion, intput_channel, 1, 1))
    _1x1_conv2d = relay.nn.conv2d(input, weight_1x1_conv2d, kernel_size=(1, 1))

    

    weight_depth_wise = relay.var(f"weight_depth_wise_{i}", shape=(intput_channel * expansion, 1, *kernel_size))
    
    # padded_conv2d = tvm.relay.nn.pad(_1x1_conv2d, ((0,0), (0,0), padding, padding))
    # depth_wise = relay.nn.conv2d(padded_conv2d, weight_depth_wise, kernel_size=kernel_size, strides=(stride, stride), padding=(0,0),
    #                              groups=intput_channel * expansion, channels=intput_channel * expansion)
    depth_wise = relay.nn.conv2d(_1x1_conv2d, weight_depth_wise, kernel_size=kernel_size, strides=(stride, stride), padding=padding,
                                 groups=intput_channel * expansion, channels=intput_channel * expansion)
    
    weight_1x1_conv2d_linear = relay.var(f"weight_1x1_conv2d_linear_{i}", shape=(output_channel, intput_channel * expansion, 1, 1))
    _1x1_conv2d_linear = relay.nn.conv2d(depth_wise, weight_1x1_conv2d_linear, kernel_size=(1, 1))

    return _1x1_conv2d_linear


def tvm_const(obj):
    return relay.Constant(nd.array(obj))

def MBConv_INT8(input ,intput_channel, output_channel, expansion=1, stride=1, padding=(1,1), kernel_size=(3,3)):

    global idx_iter
    i = next(idx_iter)

    # in_scale_1x1_conv2d = relay.var(f"in_scale_1x1_conv2d_{i}", shape=(1,), dtype="float32")
    in_scale_1x1_conv2d = np.float32(0.5)
    out_scale_1x1_conv2d = np.full((intput_channel * expansion,), 0.5, dtype="float32")
    # out_scale_1x1_conv2d = relay.var(f"out_scale_1x1_conv2d_{i}", shape=(intput_channel * expansion,), dtype="float32")
    weight_1x1_conv2d = relay.var(f"weight_1x1_conv2d_{i}", shape=(intput_channel * expansion, intput_channel, 1, 1), dtype="int8")

    _1x1_conv2d = relay.qnn.conv2d(
        input,
        weight_1x1_conv2d,
        tvm_const(np.int32(0)),
        tvm_const(np.int32(0)),
        tvm_const(in_scale_1x1_conv2d),
        tvm_const(out_scale_1x1_conv2d),
        kernel_size=(1, 1),
        channels=intput_channel * expansion,
        # out_dtype="int16"
    )

    # in_scale_depth_wise = relay.var(f"in_scale_depth_wise_{i}", "float32")
    in_scale_depth_wise = np.float32(0.5)
    # requant_1x1_conv2d = _1x1_conv2d
    requant_1x1_conv2d = relay.qnn.requantize(
        _1x1_conv2d,
        tvm_const(in_scale_1x1_conv2d * out_scale_1x1_conv2d),
        tvm_const(np.int32(0)),
        tvm_const(in_scale_depth_wise),
        tvm_const(np.int32(0)),
        axis=1,
        out_dtype="int8",
        
    )

    
    # out_scale_depth_wise = relay.var(f"out_scale_depth_wise_{i}", shape=(intput_channel * expansion,), dtype="float32")
    out_scale_depth_wise = np.full((intput_channel * expansion,), 0.5, dtype="float32")
    weight_depth_wise = relay.var(f"weight_depth_wise_{i}", shape=(intput_channel * expansion, 1, *kernel_size), dtype="int8")
    
    depth_wise = relay.qnn.conv2d(
        requant_1x1_conv2d,
        weight_depth_wise,
        tvm_const(np.int32(0)),
        tvm_const(np.int32(0)),
        tvm_const(in_scale_depth_wise),
        tvm_const(out_scale_depth_wise),
        kernel_size=kernel_size,
        strides=(stride, stride), 
        padding=padding,
        groups=intput_channel * expansion, 
        channels=intput_channel * expansion,
        # out_dtype="int16"
    )

    # in_scale_1x1_conv2d_linear = relay.var(f"in_scale_1x1_conv2d_linear_{i}", "float32")
    in_scale_1x1_conv2d_linear = np.float32(0.5)
    # requant_depth_wise = depth_wise
    requant_depth_wise = relay.qnn.requantize(
        depth_wise,
        tvm_const(in_scale_depth_wise * out_scale_depth_wise),
        tvm_const(np.int32(0)),
        tvm_const(in_scale_1x1_conv2d_linear),
        tvm_const(np.int32(0)),
        axis=1,
        out_dtype="int8",
    )

    # out_scale_1x1_conv2d_linear = relay.var(f"out_scale_1x1_conv2d_linear_{i}", shape=(output_channel,), dtype="float32")
    out_scale_1x1_conv2d_linear = np.full((output_channel,), 0.5, dtype="float32")
    weight_1x1_conv2d_linear = relay.var(f"weight_1x1_conv2d_linear_{i}", shape=(output_channel, intput_channel * expansion, 1, 1), dtype="int8")
    _1x1_conv2d_linear = relay.qnn.conv2d(
        requant_depth_wise,
        weight_1x1_conv2d_linear,
        tvm_const(np.int32(0)),
        tvm_const(np.int32(0)),
        tvm_const(in_scale_1x1_conv2d_linear),
        tvm_const(out_scale_1x1_conv2d_linear),
        kernel_size=(1, 1),
        channels=output_channel,
        # out_dtype="int16"
    )

    # requant_1x1_conv2d_linear = _1x1_conv2d_linear
    requant_1x1_conv2d_linear = relay.qnn.requantize(
        _1x1_conv2d_linear,
        tvm_const(in_scale_1x1_conv2d_linear * out_scale_1x1_conv2d_linear),
        tvm_const(np.int32(0)),
        tvm_const(np.float32(0.5)),
        tvm_const(np.int32(0)),
        axis=1,
        out_dtype="int8",
    )

    return requant_1x1_conv2d_linear

def create_mbv2_int8(input_size=(1, 3, 224, 224)):
    data = relay.var("data", shape=(1, 3, 224, 224), dtype="int8")
    # in_scale_conv2d_1 = relay.var("in_scale_conv2d_1", "float32") # scalar
    in_scale_conv2d_1 = np.float32(0.5)
    out_scale_conv2d_1 = np.full((32,), 0.5, dtype="float32")
    # out_scale_conv2d_1 = relay.var("out_scale_conv2d_1", shape=(32,), dtype="float32")

    weight1 = relay.var("conv2d_1", shape=(32, 3, 3 ,3), dtype="int8")
    conv2d_1 = relay.qnn.conv2d(
        data,
        weight1,
        tvm_const(np.int32(0)),
        tvm_const(np.int32(0)),
        tvm_const(in_scale_conv2d_1),
        tvm_const(out_scale_conv2d_1),
        kernel_size=(3, 3), strides=(2, 2), padding=(1,1),
        channels=32,
        # out_dtype="int16"
    )
    # requant_conv2d_1 = conv2d_1

    requant_conv2d_1 = relay.qnn.requantize(
        conv2d_1,
        tvm_const(in_scale_conv2d_1 * out_scale_conv2d_1),
        tvm_const(np.int32(0)),
        tvm_const(np.float32(0.5)),
        tvm_const(np.int32(0)),
        axis=1,
        out_dtype="int8",
    )

    MBConv=MBConv_INT8

    mb = MBConv(requant_conv2d_1, 32, 16, 1, 1)
   
    mb = MBConv(mb, 16, 24, 6, 2)
    mb = MBConv(mb, 24, 24, 6, 1)

    mb = MBConv(mb, 24, 32, 6, 2)

    mb = MBConv(mb, 32, 32, 6, 1)
    mb = MBConv(mb, 32, 32, 6, 1)

    mb = MBConv(mb, 32, 64, 6, 2)
    mb = MBConv(mb, 64, 64, 6, 1)
    mb = MBConv(mb, 64, 64, 6, 1)
    mb = MBConv(mb, 64, 64, 6, 1)

    mb = MBConv(mb, 64, 96, 6, 1)
    mb = MBConv(mb, 96, 96, 6, 1)
    mb = MBConv(mb, 96, 96, 6, 1)

    mb = MBConv(mb, 96, 160, 6, 2)
    mb = MBConv(mb, 160, 160, 6, 1)
    mb = MBConv(mb, 160, 160, 6, 1)

    mb = MBConv(mb, 160, 320, 6, 1)

    in_scale_conv2d_2 = np.float32(0.5)
    # out_scale_conv2d_2 = relay.var("out_scale_conv2d_2", shape=(1280,), dtype="float32")
    out_scale_conv2d_2 = np.full((1280,), 0.5, dtype="float32")
    weight2 = relay.var("conv2d_2", shape=(1280, 320, 1 ,1), dtype="int8")
    conv2d_2 = relay.qnn.conv2d(
        mb,
        weight2,
        tvm_const(np.int32(0)),
        tvm_const(np.int32(0)),
        tvm_const(in_scale_conv2d_2),
        tvm_const(out_scale_conv2d_2),
        kernel_size=(1, 1),
        channels=1280,
        # out_dtype="int16"
    )
    # requant_conv2d_2 = conv2d_2
    requant_conv2d_2 = relay.qnn.requantize(
        conv2d_2,
        tvm_const(in_scale_conv2d_2 * out_scale_conv2d_2),
        tvm_const(np.int32(0)),
        tvm_const(np.float32(0.5)),
        tvm_const(np.int32(0)),
        axis=1,
        out_dtype="int8",
    )
    
    
    global_avg_pool = relay.qnn.avg_pool2d(requant_conv2d_2, 
                                           tvm_const(np.float32(0.5)),
                                           tvm_const(np.int32(0)),
                                           tvm_const(np.float32(0.5)),
                                           tvm_const(np.int32(-128)),
                                           strides=(7, 7),
                                           padding=(0, 0),
                                           dilation=(1, 1),
                                           pool_size=(7, 7),
                                           layout="NCHW",
                                           )

    ### Dummy Block ###
    reshape = relay.reshape(global_avg_pool, (1,1280,))
    dense_weight = relay.var("dense_weight", shape=(1, 1280))
    dummpy_dense = relay.nn.dense(reshape, dense_weight)

    func = relay.Function(relay.analysis.free_vars(dummpy_dense), dummpy_dense)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    print("Before QNN Legalize and CanonicalizeOps:")
    print(mod)
    mod = relay.qnn.transform.Legalize()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    return mod

def create_mbv2_float32(input_size=(1, 3, 224, 224)):
    MBConv=MBConv_FLOAT32
    data = relay.var("data", shape=(1, 3, 224, 224))

    weight1 = relay.var("conv2d_1", shape=(32, 3, 3 ,3))
    conv2d_1 = relay.nn.conv2d(data, weight1, kernel_size=(3, 3), strides=(2, 2), padding=(1,1))

    mb = MBConv(conv2d_1, 32, 16, 1, 1)
   
    mb = MBConv(mb, 16, 24, 6, 2)
    mb = MBConv(mb, 24, 24, 6, 1)

    mb = MBConv(mb, 24, 32, 6, 2)

    mb = MBConv(mb, 32, 32, 6, 1)
    mb = MBConv(mb, 32, 32, 6, 1)

    mb = MBConv(mb, 32, 64, 6, 2)
    mb = MBConv(mb, 64, 64, 6, 1)
    mb = MBConv(mb, 64, 64, 6, 1)
    mb = MBConv(mb, 64, 64, 6, 1)

    mb = MBConv(mb, 64, 96, 6, 1)
    mb = MBConv(mb, 96, 96, 6, 1)
    mb = MBConv(mb, 96, 96, 6, 1)

    mb = MBConv(mb, 96, 160, 6, 2)
    mb = MBConv(mb, 160, 160, 6, 1)
    mb = MBConv(mb, 160, 160, 6, 1)

    mb = MBConv(mb, 160, 320, 6, 1)

    weight2 = relay.var("conv2d_2", shape=(1280, 320, 1 ,1))
    conv2d_2 = relay.nn.conv2d(mb, weight2, kernel_size=(1, 1))
    global_avg_pool = relay.nn.avg_pool2d(conv2d_2, pool_size=(7, 7))

    ### Dummy Block ###
    reshape = relay.reshape(global_avg_pool, (1,1280,))
    dense_weight = relay.var("dense_weight", shape=(1, 1280))
    dummpy_dense = relay.nn.dense(reshape, dense_weight)

    func = relay.Function(relay.analysis.free_vars(dummpy_dense), dummpy_dense)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod

def MBConv_Fake_INT8(input ,intput_channel, output_channel, expansion=1, stride=1, padding=(1,1), kernel_size=(3,3)):
    global idx_iter
    i = next(idx_iter)

    intput_channel = int(intput_channel)
    output_channel = int(output_channel)

    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    weight_1x1_conv2d = relay.var(f"weight_1x1_conv2d_{i}", shape=(intput_channel * expansion, intput_channel, 1, 1), dtype="int8")
    _1x1_conv2d = relay.nn.conv2d(input, weight_1x1_conv2d, kernel_size=(1, 1), out_dtype="int8")

    weight_depth_wise = relay.var(f"weight_depth_wise_{i}", shape=(intput_channel * expansion, 1, *kernel_size), dtype="int8")

    # padded_conv2d = tvm.relay.nn.pad(_1x1_conv2d, ((0,0), (0,0), padding, padding))
    # depth_wise = relay.nn.conv2d(padded_conv2d, weight_depth_wise, kernel_size=kernel_size, strides=(stride, stride), padding=(0, 0),
    #                              groups=intput_channel * expansion, channels=intput_channel * expansion, out_dtype="int8")
    
    depth_wise = relay.nn.conv2d(_1x1_conv2d, weight_depth_wise, kernel_size=kernel_size, strides=(stride, stride), padding=padding,
                                 groups=intput_channel * expansion, channels=intput_channel * expansion, out_dtype="int8")
    
    weight_1x1_conv2d_linear = relay.var(f"weight_1x1_conv2d_linear_{i}", shape=(output_channel, intput_channel * expansion, 1, 1), dtype="int8")
    _1x1_conv2d_linear = relay.nn.conv2d(depth_wise, weight_1x1_conv2d_linear, kernel_size=(1, 1), out_dtype="int8")

    return _1x1_conv2d_linear

def create_mbv2_fake_int8(input_size=(1, 3, 224, 224), w=1):
    MBConv=MBConv_Fake_INT8
    data = relay.var("data", shape=input_size, dtype="int8")

    weight1 = relay.var("conv2d_1", shape=(int(32*w), input_size[1], 3 ,3), dtype="int8")
    conv2d_1 = relay.nn.conv2d(data, weight1, kernel_size=(3, 3), strides=(2, 2), 
                               padding=(1,1), 
                               out_dtype="int8"
                               )

    mb = MBConv(conv2d_1, 32*w, 16*w, 1, 1)
   
    mb = MBConv(mb, 16*w, 24*w, 6, 2)
    mb = MBConv(mb, 24*w, 24*w, 6, 1)

    mb = MBConv(mb, 24*w, 32*w, 6, 2)

    mb = MBConv(mb, 32*w, 32*w, 6, 1)
    mb = MBConv(mb, 32*w, 32*w, 6, 1)

    mb = MBConv(mb, 32*w, 64*w, 6, 2)
    mb = MBConv(mb, 64*w, 64*w, 6, 1)
    mb = MBConv(mb, 64*w, 64*w, 6, 1)
    mb = MBConv(mb, 64*w, 64*w, 6, 1)

    mb = MBConv(mb, 64*w, 96*w, 6, 1)
    mb = MBConv(mb, 96*w, 96*w, 6, 1)
    mb = MBConv(mb, 96*w, 96*w, 6, 1)

    mb = MBConv(mb, 96*w, 160*w, 6, 2)
    mb = MBConv(mb, 160*w, 160*w, 6, 1)
    mb = MBConv(mb, 160*w, 160*w, 6, 1)

    mb = MBConv(mb, 160*w, 320*w, 6, 1)

    weight2 = relay.var("conv2d_2", shape=(int(1280*w), int(320*w), 1 ,1), dtype="int8")
    conv2d_2 = relay.nn.conv2d(mb, weight2, kernel_size=(1, 1), out_dtype="int8")
    global_avg_pool = relay.nn.avg_pool2d(conv2d_2, pool_size=(7, 7))

    ### Dummy Block ###
    reshape = relay.reshape(global_avg_pool, (1,int(1280*w),))
    dense_weight = relay.var("dense_weight", shape=(1, int(1280*w)))
    dummpy_dense = relay.nn.dense(reshape, dense_weight)

    body = conv2d_2 # TODO

    func = relay.Function(relay.analysis.free_vars(body), body)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod

def create_mbv2_vww_5fps_int8(input_size=(1, 3, 80, 80)):
    MBConv=MBConv_Fake_INT8
    data = relay.var("data", shape=input_size, dtype="int8")

    weight1 = relay.var("conv2d_1", shape=(int(16), input_size[1], 3 ,3), dtype="int8")
    conv2d_1 = relay.nn.conv2d(data, weight1, kernel_size=(3, 3), strides=(2, 2), 
                               padding=(1,1), 
                               out_dtype="int8"
                               )
    weight_depth_wise = relay.var(f"weight_depth_wise_1_", shape=(16, 1, 3, 3), dtype="int8")
    depth_wise = relay.nn.conv2d(conv2d_1, weight_depth_wise, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                                 groups=16, channels=16, out_dtype="int8")
    weight2 = relay.var("conv2d_2", shape=(8, 16, 1, 1), dtype="int8")
    conv2d_2 = relay.nn.conv2d(depth_wise, weight2, kernel_size=(1, 1), strides=(1, 1), 
                               padding=(0, 0), 
                               out_dtype="int8"
                               )
    
    mb = MBConv(conv2d_2, 8, 16, 6, 2)
    mb = MBConv(mb, 16, 16, 3, 1)
    mb = MBConv(mb, 16, 16, 3, 1)

    mb = MBConv(mb, 16, 24, 3, 2, 3, 7)

    mb = MBConv(mb, 24, 24, 6, 1)
    mb = MBConv(mb, 24, 24, 5, 1, 2, 5)
    mb = MBConv(mb, 24, 40, 6, 2, 3, 7)

    mb = MBConv(mb, 40, 40, 6, 1, 3, 7)

    mb = MBConv(mb, 40, 48, 6, 1, 1, 3)
    mb = MBConv(mb, 48, 48, 4, 1, 1, 3)
    mb = MBConv(mb, 48, 96, 5, 2, 2, 5)

    mb = MBConv(mb, 96, 96, 5, 1, 1, 3)
    mb = MBConv(mb, 96, 96, 4, 1, 1, 3)
    mb = MBConv(mb, 96, 160, 3, 1, 3, 7)
    # PoolingLayer(pool_size=3, stride=1),
    body = mb # TODO

    func = relay.Function(relay.analysis.free_vars(body), body)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod

def create_mcunet_320kb_imagenet_int8(input_size=(1, 3, 176, 176)):
    MBConv=MBConv_Fake_INT8
    data = relay.var("data", shape=input_size, dtype="int8")

    weight1 = relay.var("conv2d_1", shape=(int(16), input_size[1], 3 ,3), dtype="int8")
    conv2d_1 = relay.nn.conv2d(data, weight1, kernel_size=(3, 3), strides=(2, 2), 
                               padding=(1,1), 
                               out_dtype="int8"
                               )
    weight_depth_wise = relay.var(f"weight_depth_wise_1_", shape=(16, 1, 3, 3), dtype="int8")
    depth_wise = relay.nn.conv2d(conv2d_1, weight_depth_wise, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                                 groups=16, channels=16, out_dtype="int8")
    weight2 = relay.var("conv2d_2", shape=(8, 16, 1, 1), dtype="int8")
    conv2d_2 = relay.nn.conv2d(depth_wise, weight2, kernel_size=(1, 1), strides=(1, 1), 
                               padding=(0, 0), 
                               out_dtype="int8"
                               )
    mb = MBConv(conv2d_2, 8, 16, 3, 2, 3, 7)
    mb = MBConv(mb, 16, 16, 5, 1)
    mb = MBConv(mb, 16, 16, 5, 1, 3, 7)
    
    mb = MBConv(mb, 16, 16, 4, 1, 2, 5)

    mb = MBConv(mb, 16, 24, 5, 2, 2, 5)

    mb = MBConv(mb, 24, 24, 5, 1, 2, 5)

    mb = MBConv(mb, 24, 24, 5, 1, 2, 5)

    mb = MBConv(mb, 24, 40, 5, 2, 1, 3)

    mb = MBConv(mb, 40, 40, 6, 1, 3, 7)
    mb = MBConv(mb, 40, 40, 4, 1, 2, 5)
    mb = MBConv(mb, 40, 48, 5, 1, 2, 5)

    mb = MBConv(mb, 48, 48, 5, 1, 3, 7)
    mb = MBConv(mb, 48, 48, 5, 1, 1, 3)

    mb = MBConv(mb, 48, 96, 6, 2, 1, 3)
    mb = MBConv(mb, 96, 96, 5, 1, 3, 7)
    mb = MBConv(mb, 96, 96, 4, 1, 1, 3)
    mb = MBConv(mb, 96, 160, 5, 1, 3, 7)
    # PoolingLayer(pool_size=3, stride=1),
    body = mb # TODO

    func = relay.Function(relay.analysis.free_vars(body), body)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    return mod

def create_np_ramdon_params_for(mod):
    relay_params = mod["main"].params
    params = {}

    for p in relay_params:
        if p.name_hint == 'data':
            continue
        shape = [int(i) for i in p.type_annotation.shape]
        nd_arr = tvm.nd.array(np.random.randint(-255, 254, size=shape).astype(np.int8))
        params[p.name_hint] = nd_arr
    return params

from partial_conv.fusion_network_rewrite import MultiStageFusionNetworkRewriter
from tvm.relay import dataflow_pattern as dfp
import copy

def export_models_from_analytic_settings(s_path="analytic_results_v1.json", export_dir="./MFC_models/"):
    import json
    with open(s_path, "rb") as f:
        settings_and_res = json.load(f)
    mbv2_w35 = create_mbv2_fake_int8(input_size=(1, 3, 144, 144), w=.35)
    mbv2_vvw = create_mbv2_vww_5fps_int8()
    mbv2_320kb = create_mcunet_320kb_imagenet_int8()
    new_results = []

    for setting in settings_and_res:
        if setting["Model"] == "MBV2-w0.35-r144":
            mod_ori = mbv2_w35
        elif setting["Model"] == "MCUNetv2-vww-5fps-r80":
            mod_ori = mbv2_vvw
        else:
            mod_ori = mbv2_320kb
        fusion_range = setting["Fusion Range"]
        model_dir_name = f'{setting["Model"]}_{setting["Configuration"]}'
        export_dir_path = os.path.join(export_dir, model_dir_name)
        if not os.path.exists(export_dir_path):
            os.mkdir(export_dir_path)

        export_model_path = os.path.join(export_dir_path, "default.tar")

        fusion_rewriter = MultiStageFusionNetworkRewriter(fusion_range, mod_ori["main"])  
        fusion_body = fusion_rewriter.fused_neural_network
        mod_fusion = tvm.IRModule.from_expr(fusion_body)
        mod_fusion = relay.transform.InferType()(mod_fusion)
        mod = mod_fusion

        params = create_np_ramdon_params_for(mod)
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
        # TARGET = tvm.target.target.micro('host')
        # TARGET = tvm.target.target.micro('nrf52840')
        TARGET = "c -keys=partial_conv,arm_cpu,cpu -mcpu=cortex-m4+nodsp -model=nrf52840"      
        with tvm.transform.PassContext(opt_level=0, config={
                                                        "tir.disable_vectorize": True, 
                                                        "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                        "tir.usmp.algorithm": "hill_climb",
                                                        "relay.backend.use_auto_scheduler": True, # Keep that for Primitive Function with multiple heavy ops (like Convs)
                                                        "relay.remove_standalone_reshapes.enable": False
                                                        },
                                                        # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                        ): 

            module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)
            mem_pool_size = int([v for v in module.function_metadata['__tvm_main__'].workspace_sizes.values()][0])
            print(f"mem pool size: {mem_pool_size}")
        export_model_library_format(module, export_model_path)
        generate_mlmci_files(module, params, export_dir_path)
        
        new_s = copy.deepcopy(setting)
        new_s['peak_mem_usage'] = mem_pool_size
        new_results.append(new_s)
        with open(export_dir_path + "/result.json", "w") as f:
            json.dump(new_s, f)

    with open(export_dir + "/result.json", "w") as f:
        json.dump(new_results, f)



if __name__=="__main__":
    # export_models_from_analytic_settings(export_dir="./MFC_models_20250112/")
    mod_ori = create_mbv2_fake_int8(input_size=(1, 3, 144, 144), w=.35)
    # mod_ori = create_mbv2_vww_5fps_int8()
    # mod_ori = create_mcunet_320kb_imagenet_int8()

    fusion_range = [[0, 6], [7, 12], [13, 15], [16, 21], [22, 22], [23, 24], [25, 25], [26, 27], [28, 28], [29, 30], [31, 31], [32, 33], [34, 36], [37, 41], [42, 43], [44, 45], [46, 46], [47, 48], [49, 49], [50, 52]]
    fusion_rewriter = MultiStageFusionNetworkRewriter(fusion_range, mod_ori["main"])  
    
    fusion_body = fusion_rewriter.fused_neural_network

    mod_fusion = tvm.IRModule.from_expr(fusion_body)
    mod_fusion = relay.transform.InferType()(mod_fusion)
    # mod_ori = create_mcunet_320kb_imagenet_int8()

    mod = mod_fusion

    params = create_np_ramdon_params_for(mod)
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
    # TARGET = tvm.target.target.micro('host')
    # TARGET = tvm.target.target.micro('nrf52840')
    TARGET = "c -keys=partial_conv,arm_cpu,cpu -mcpu=cortex-m4+nodsp -model=nrf52840"
    # """c -keys=arm_cpu,cpu -mcpu=cortex-m4+nodsp -model=nrf52840"""
    # TARGET = tvm.target.target.stm32('stm32F7xx')
    
    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                    "tir.usmp.algorithm": "hill_climb",
                                                    "relay.backend.use_auto_scheduler": True, # Keep that for Primitive Function with multiple heavy ops (like Convs)
                                                    "relay.remove_standalone_reshapes.enable": False
                                                    },
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                    ): 

        module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)
        mem_pool_size = [v for v in module.function_metadata['__tvm_main__'].workspace_sizes.values()][0]
        print(f"mem pool size: {mem_pool_size}")
    export_model_library_format(module, "./models/default/default.tar")
    generate_mlmci_files(module, params, "./")
    
    from partial_conv.op.macro_globalvar import get_recorded_worker_id_op_num
    print(get_recorded_worker_id_op_num())





