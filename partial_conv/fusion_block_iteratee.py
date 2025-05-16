import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay, topi
import tvm.te

from .dyn_slice_fixed_size import dyn_slice_fixed_size
from .op.copy_inplace import copy_in_place_spatial
from .utils import CollectOpShapeInfo, ReWriteInputsShape, ReWriteSwapVars, InferCallNodeType, EliminateIterateeDummyCallNode

from .iterative_ops import iterative_global_avg_pool_step

import copy

def copy_var_with_name_prefix(var, prefix='_'):
    return copy_var_with_new_name(var, prefix + str(var.name_hint))

def copy_var_with_new_name(var, name=None):
    return relay.Var(name, var.type_annotation)


def calculate_new_input_shape_and_strides(op_info, expected_output_shape):
    new_input_shape = [*expected_output_shape]
    new_input_stride = [*expected_output_shape]
    for i in op_info[::-1]:
        s = [1, 1]
        d = [1, 1]
        p = [0, 0]
        k = [1, 1]

        if i['op_name'] == 'nn.dense':
            continue
        elif i['op_name'] == 'reshape':
            continue
        elif i['op_name'] == 'nn.avg_pool2d':
            input_hw = [i['input_shape'][-2], i['input_shape'][-1]]
            is_iterative_pool = input_hw == i['kernel_size']
            if is_iterative_pool:
                avg_dividend = input_hw[0] + input_hw[1]
                continue
        k = i['kernel_size']
        s = i['strides']
        # TODO deal with (zero)-padding
        # p = i['padding'][-2:]
        
        new_input_shape[0] = (new_input_shape[0] - 1) * s[0] - 2 * p[0] + d[0] * (k[0]-1) + 1
        new_input_shape[1] = (new_input_shape[1] - 1) * s[1] - 2 * p[1] + d[1] * (k[1]-1) + 1
        new_input_stride[0] = new_input_stride[0] * s[0]
        new_input_stride[1] = new_input_stride[1] * s[1]
        i['input_tile_size'] = [*new_input_shape]
        i['input_tile_stride'] = [*new_input_stride]
    return new_input_shape, new_input_stride


# 0. analysis input tile size wrt block_output_size
# 1. rewrite input shape of block, and infer the corresponding output type
# 2. insert dyn_slice at the beginining
# 3. insert copy_inplace at the end to aggregated result
# 4. create relay function iteratee(iterator, iter_var, layer_args) -> Tensor(type=iter_var.type)
def create_fusion_block_iteratee(layers_node, block_output_size=1, global_symbol="iteratee"):
    iteratee_func = None
    new_output_hw = [block_output_size, block_output_size]
    
    op_shape_collector = CollectOpShapeInfo()
    op_shape_collector.visit(layers_node)
    op_info = op_shape_collector.op_info
    op_to_info_map = op_shape_collector.op_to_info_map

    new_input_layout, new_input_stride = calculate_new_input_shape_and_strides(op_info, new_output_hw)

    conv_chain_node = layers_node
    conv_chain_node_cp = copy.deepcopy(conv_chain_node)

    conv_chain_params = relay.analysis.free_vars(conv_chain_node)
    conv_chain_params_cp =  relay.analysis.free_vars(conv_chain_node_cp)
    input_var = conv_chain_params_cp[0]
    input_shape = input_var.checked_type.shape
    in_w = int(input_shape[-2])
    in_h = int(input_shape[-1])

    output_shape = conv_chain_node.checked_type.shape
    iteratee_output_shape = (output_shape[0],output_shape[1], *new_output_hw)

    # re_write_inputs = ReWriteInputsShape({'data': (input_shape[0],input_shape[1],new_input_layout[0],new_input_layout[1])})
    
    # conv_chain_block = re_write_inputs.visit(conv_chain_node_cp)
    conv_chain_block = conv_chain_node_cp

    slice_begin_var = relay.var("iterator", shape=(4,), dtype="int32")

    # initial_output_var = relay.var("iter_var", shape=output_shape, dtype="float32")

    new_input_var = dyn_slice_fixed_size(input_var, slice_begin_var, [int(input_shape[0]),int(input_shape[1]),*new_input_layout])


    rewrite_swap_vars = ReWriteSwapVars({"data":new_input_var})
    conv_chain_block = rewrite_swap_vars.visit(conv_chain_block)
    conv_chain_block = InferCallNodeType().visit(conv_chain_block)

    # begin_indices = relay.zeros(shape=(4,), dtype="int32")
    # slice_x_y = relay.take(slice_begin_var, relay.const([2 , 3]))
    # begin_x_y = relay.divide(slice_x_y, relay.const(new_input_stride))
    # begin_indices = relay.concatenate([relay.const([0 , 0]), begin_x_y], 0)
    # iteratee_body = copy_in_place_spatial(initial_output_var, conv_chain_block, begin_indices)

    params = [slice_begin_var, *conv_chain_params_cp]

    iteratee_func = relay.Function(params, conv_chain_block).with_attr("Primitive", tvm.tir.IntImm("int32", 1)) \
                                # .with_attr("global_symbol", global_symbol) # keep that for stable func name
    iteratee_func = InferCallNodeType().visit(iteratee_func).with_attr("global_symbol", global_symbol)

    return iteratee_func, new_input_layout, new_input_stride, output_shape

from .op.cache_conv_input import cache_conv_input
class InsertConvInputCacheAndIterPool(relay.ExprMutator):
    def __init__(self, op_to_info_map):
        super().__init__()
        self.op_to_info_map = op_to_info_map  # The new input expression to replace the original input

    def visit_call(self, call):
        new_args = []
        for arg in call.args:
            new_args.append(self.visit(arg))
        
        # Check if the call node is a `conv2d` operation
        if call.op.name == "nn.conv2d":
            # Replace the input of the `conv2d` with `self.new_input`
            op_info = self.op_to_info_map[call]
            input_shape = op_info['input_shape']
            input_tile_size = op_info['input_tile_size']
            input_tile_stride = op_info['input_tile_stride']
            kernel_size = op_info['kernel_size']
            strides = op_info['strides']
            padding = op_info['padding']
            buffer_shape = [op_info['input_shape'][0], op_info['input_shape'][1], 0, 0]
            buffer_shape[2] = input_tile_size[0]
            buffer_shape[3] = op_info['kernel_size'][1]

            print("insert conv cache:", op_info)
            input_to_cache = new_args[0]
            if op_info['first_conv']:
                print("insert dyn slice for first conv node")
                slice_begin_var = relay.var("iterator", shape=(4,), dtype="int32")
                input_to_cache = dyn_slice_fixed_size(new_args[0], slice_begin_var, [int(input_shape[0]),int(input_shape[1]), input_tile_size[0], 1])
            
            if not kernel_size == [1, 1]:
                cache_out = cache_conv_input(input_to_cache, buffer_shape=buffer_shape, max_idx=[0, 0], 
                                            conv_kernel_size=kernel_size, conv_strides=strides, conv_padding=padding, conv_input_shape=input_shape,
                                            input_tile_size=input_tile_size, input_tile_stride=input_tile_stride,
                                            conv_dtype=call.checked_type.dtype)
            else:
                cache_out = input_to_cache
            
            attrs = {**call.attrs}
            attrs['padding'] = [0,0,0,0] # Let cache manager to care about padding 
            modified_conv = relay.nn.conv2d(cache_out, *call.args[1:], **attrs)
            
            return modified_conv
        
        elif call.op.name == "nn.avg_pool2d":
            # Replace the input of the `conv2d` with `self.new_input`
            op_info = self.op_to_info_map[call]
            input_shape = op_info['input_shape']
            input_tile_size = op_info['input_tile_size']
            kernel_size = op_info['kernel_size']
            strides = op_info['strides']
            padding = op_info['padding']

            out_shape = op_info['output_shape']
            out_hw = out_shape[2:]
            out_ch = out_shape[1]

            if out_hw == (1, 1): #global pooling
                print("insert iterative avg_pool:", op_info)
                avg_input = new_args[0]
                cache_prev_output = relay.var("cache_prev_gp_output", shape=out_shape, dtype=call.checked_type.dtype) # gp: global pooling
                output = iterative_global_avg_pool_step(avg_input, cache_prev_output, input_shape[-2] * input_shape[-1], call.checked_type.dtype)    
                return output

        # For other operations, continue visiting as usual
        return super().visit_call(call)

# class IterizeGlobalPooling(relay.ExprMutator):
#     def __init__(self, op_to_info_map):
#         super().__init__()
#         self.op_to_info_map = op_to_info_map  # The new input expression to replace the original input

#     def visit_call(self, call):
#         new_args = []
#         for arg in call.args:
#             new_args.append(self.visit(arg))
        
#         # Check if the call node is a `conv2d` operation
#         if call.op.name == "nn.avg_pool2d":
#             # Replace the input of the `conv2d` with `self.new_input`
#             op_info = self.op_to_info_map[call]
#             input_shape = op_info['input_shape']
#             input_tile_size = op_info['input_tile_size']
#             kernel_size = op_info['kernel_size']
#             strides = op_info['strides']
#             padding = op_info['padding']

#             out_shape = op_info['output_shape']
#             out_hw = out_shape[2:]
#             out_ch = out_shape[1]

#             if out_hw == (1, 1): #global pooling
#                 print("insert iterative avg_pool:", op_info)
#                 avg_input = new_args[0]
#                 cache_prev_output = relay.var("cache_prev_gp_output", shape=out_shape, dtype=call.checked_type.dtype) # gp: global pooling
#                 output = iterative_global_avg_pool_step(avg_input, cache_prev_output, input_shape[-2] * input_shape[-1])    
#                 return output

#         # For other operations, continue visiting as usual
#         return super().visit_call(call)

    

def create_fusion_block_iteratee_with_cache(layers_node, block_output_size=1, global_symbol="iteratee"):
    iteratee_func = None
    new_output_hw = [block_output_size, block_output_size]
    layers_node = InferCallNodeType().visit(layers_node)
    op_shape_collector = CollectOpShapeInfo()
    op_shape_collector.visit(layers_node)
    op_info = op_shape_collector.op_info
    op_to_info_map = op_shape_collector.op_to_info_map

    new_input_layout, new_input_stride = calculate_new_input_shape_and_strides(op_info, new_output_hw)

    conv_chain_node = layers_node
    conv_chain_node_cp = copy.deepcopy(conv_chain_node)

    conv_chain_params = relay.analysis.free_vars(conv_chain_node)
    conv_chain_params_cp =  relay.analysis.free_vars(conv_chain_node_cp)
    input_var = conv_chain_params_cp[0]
    input_shape = input_var.checked_type.shape
    in_w = int(input_shape[-2])
    in_h = int(input_shape[-1])

    output_shape = conv_chain_node.checked_type.shape

    conv_chain_block = InsertConvInputCacheAndIterPool(op_to_info_map).visit(conv_chain_node)

    conv_chain_block = InferCallNodeType().visit(conv_chain_block)

    iterator_var = None
    cache_vars = []

    params_var = relay.analysis.free_vars(conv_chain_block)
    for v in params_var:
        if (str(v.name_hint) == 'iterator'):
            iterator_var = v
        elif (str(v.name_hint).startswith('cache')):
            cache_vars.append(v)

    conv_chain_block_params = [iterator_var, *conv_chain_params, *cache_vars]

    iteratee_func = relay.Function(conv_chain_block_params, conv_chain_block).with_attr("Primitive", tvm.tir.IntImm("int32", 1))

    iteratee_func = InferCallNodeType().visit(iteratee_func).with_attr("global_symbol", global_symbol)

    return iteratee_func, conv_chain_params, cache_vars, new_input_layout, new_input_stride, output_shape, input_shape

def create_zeros_for_vars(vars):
    zeros_vars = []
    for v in vars:
        z = relay.zeros(shape=v.type_annotation.shape, dtype=v.type_annotation.dtype)
        zeros_vars.append(z)
    return zeros_vars

from .op.fusion_iter_worker import fusion_iter_worker
from .op.strategy import conv2d

if __name__=="__main__":
    data = relay.var("data", shape=(1, 1, 224, 224))
    # data = relay.var("data", shape=(1, 3, 15, 15))
###################### Big ##############################333
    weight1 = relay.var("weight1", shape=(64, 1, 7, 7))
    weight2 = relay.var("weight2", shape=(64, 64, 3, 3))
    weight3 = relay.var("weight3", shape=(64, 64, 3, 3))
    dense_weight = relay.var("dense_weight", shape=(10, 64))

    # params = {"weight1": tvm.nd.array(np.random.rand(64, 1, 7, 7).astype(np.float32)),
    #             "weight2": tvm.nd.array(np.random.rand(64, 64, 3, 3).astype(np.float32)),
    #             "weight3": tvm.nd.array(np.random.rand(64, 64, 3, 3).astype(np.float32)),
    #             "dense_weight": tvm.nd.array(np.random.rand(10, 64).astype(np.float32)),
    #             }

    conv1 = relay.nn.conv2d(data, weight1, kernel_size=(7, 7))
    conv2 = relay.nn.conv2d(conv1, weight2, kernel_size=(3, 3))
    conv3 = relay.nn.conv2d(conv2, weight3, kernel_size=(3, 3))
    func = relay.Function([data, weight1, weight2, weight3], conv3)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    iteratee_func, conv_chain_params, cache_vars, new_input_layout, new_input_stride, output_shape, input_shape = create_fusion_block_iteratee_with_cache(mod["main"].body, 1)

    cache_zeros = create_zeros_for_vars(cache_vars)

    in_w = input_shape[2]
    in_h = input_shape[3]
    iter_begin = [0,0,0,0]
    iter_end = [0, 0, in_w - new_input_layout[0] + 1, in_h - new_input_layout[1] + 1]
    iter_strides = [1,1, *new_input_stride]
    # iterator = relay.zeros(shape=(4,), dtype="int32")
    # iter_output_var = relay.zeros(shape=(1,64,15,15), dtype="float32")

    # # dummy_var to keep the iteratee not to be optimized out
    # dummy_var = relay.Call(iteratee_func, [iterator, data, weight1, weight2, weight3], tvm.ir.make_node("DictAttrs", iteratee_dummy=1))
    # dummy_var = relay.zeros(shape=(4,), dtype="float32")
    
    iter_worker = fusion_iter_worker(iter_begin, iter_end, iter_strides, output_shape,
                                     [data, weight1, weight2, weight3], iteratee_func, cache_vars=cache_vars)
    

    iter_body = relay.Function(relay.analysis.free_vars(iter_worker), iter_worker)

    # iter_body = EliminateIterateeDummyCallNode().visit(iter_body)

    iteratee_mod = tvm.IRModule.from_expr(iter_body)
    iteratee_mod['iteratee'] = iteratee_func
    iteratee_mod = relay.transform.InferType()(iteratee_mod)

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
    TARGET = "c -keys=partial_conv,arm_cpu,cpu -mcpu=cortex-m4+nodsp -model=nrf52840"

    from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll


    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                    "tir.usmp.algorithm": "hill_climb",
                                                    "relay.backend.use_auto_scheduler": True,
                                                    "relay.remove_standalone_reshapes.enable": False
                                                    },
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                    ): 
        # print(params.keys())
        # opt_module, _ = relay.optimize(mod, target=TARGET)
        module = relay.build(iteratee_mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)

    from tvm.micro import export_model_library_format

    export_model_library_format(module, "./test_fusion_block_iteratee.tar")
