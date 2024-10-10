import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay
from tvm.relay import dataflow_pattern as dfp
from tvm.relay.transform import function_pass, InferType
from tvm.micro import export_model_library_format
import numpy as np

from model_converter import compile_per_model_eval
from mlmci_utils import generate_mlmci_files
from partial_conv.relay_rewrite import rewrite_conv_chain_to_function

from tvm.relay.op.strategy.generic import conv2d_strategy

conv2d_strategy.register("generic", conv2d_strategy.__wrapped__) # use unoptimized conv2d to avoid kernel_vec copy

if __name__=="__main__":
    data = relay.var("data", shape=(1, 1, 128, 128))
    # data = relay.var("data", shape=(1, 3, 15, 15))
###################### Big ##############################333
#    weight1 = relay.var("weight1", shape=(64, 1, 7, 7))
#    weight2 = relay.var("weight2", shape=(64, 64, 3, 3))
#    weight3 = relay.var("weight3", shape=(64, 64, 3, 3))
#    dense_weight = relay.var("dense_weight", shape=(10, 64))
#
#    params = {"weight1": tvm.nd.array(np.random.rand(64, 1, 7, 7).astype(np.float32)),
#              "weight2": tvm.nd.array(np.random.rand(64, 64, 3, 3).astype(np.float32)),
#              "weight3": tvm.nd.array(np.random.rand(64, 64, 3, 3).astype(np.float32)),
#              "dense_weight": tvm.nd.array(np.random.rand(10, 64).astype(np.float32)),
#              }
#
#    conv1 = relay.nn.conv2d(data, weight1, kernel_size=(7, 7))
#    conv2 = relay.nn.conv2d(conv1, weight2, kernel_size=(3, 3))
#    conv3 = relay.nn.conv2d(conv2, weight3, kernel_size=(3, 3))
#    avg_pool = relay.nn.avg_pool2d(conv3, pool_size=(14, 14))
#    reshape = relay.reshape(avg_pool, (1,64,))
################################################################3

###################### Small ##############################333
    weight1 = relay.var("weight1", shape=(6, 1, 7, 7))
    weight2 = relay.var("weight2", shape=(16, 6, 3, 3))
    weight3 = relay.var("weight3", shape=(6, 16, 3, 3))
    dense_weight = relay.var("dense_weight", shape=(10, 6))

    params = {"weight1": tvm.nd.array(np.random.rand(6, 1, 7, 7).astype(np.float32)),
              "weight2": tvm.nd.array(np.random.rand(16, 6, 3, 3).astype(np.float32)),
              "weight3": tvm.nd.array(np.random.rand(6, 16, 3, 3).astype(np.float32)),
              "dense_weight": tvm.nd.array(np.random.rand(10, 6).astype(np.float32)),
              }

    conv1 = relay.nn.conv2d(data, weight1, kernel_size=(7, 7))
    conv2 = relay.nn.conv2d(conv1, weight2, kernel_size=(3, 3))
    conv3 = relay.nn.conv2d(conv2, weight3, kernel_size=(3, 3))
    avg_pool = relay.nn.avg_pool2d(conv3, pool_size=(118, 118))
    reshape = relay.reshape(avg_pool, (1,6,))
################################################################3


    dense = relay.nn.dense(reshape, dense_weight)

    func = relay.Function([data, weight1, weight2, weight3 ,dense_weight], dense)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    print(mod)
    # Apply the transformation
    # mod = rewrite_conv_chain_to_function(mod)
    
    # Print the rewritten module
    print(mod)

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
    TARGET = "c -keys=generic,cpu -model=host"




    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                    "tir.usmp.algorithm": "hill_climb",
                                                    "relay.backend.use_auto_scheduler": True, # Keep that for Primitive Function with multiple heavy ops (like Convs)
                                                    },
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                    ): 

        module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)
    export_model_library_format(module, "./models/default/default.tar")
    generate_mlmci_files(module, params, "./")

