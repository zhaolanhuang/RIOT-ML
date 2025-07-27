import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay
from tvm.relay import dataflow_pattern as dfp
from tvm.relay.transform import function_pass, InferType
from tvm.micro import export_model_library_format

import copy

from .fusion_block_iteratee import create_fusion_block_iteratee, create_fusion_block_iteratee_with_cache
from .dyn_slice_fixed_size import dyn_slice_fixed_size

from .op.copy_inplace import copy_in_place_spatial
from .op.fusion_iter_worker import fusion_iter_worker

from .utils import CollectOpShapeInfo, ReWriteSwapVars



class MultiStageFusionNetworkRewriter(relay.ExprMutator):
    def __init__(self, fusion_indices, neural_network):
        super().__init__()
        self.fusion_indices = fusion_indices # [[begin_conv_idx1, end_conv_idx1], [begin_conv_idx2, end_conv_idx2] ...] #end_idx2 should include the last layer
        
        self.swap_var_to_node_input_arg_map = {}
        self.cur_dummy_input_var = None
        self.cur_fusion_begin_node = None
        self.neural_network = neural_network
        # self.conv_node_to_ori_args_map = {}

        # To record input/output shapes and other attributes
        info_collector = CollectOpShapeInfo()
        info_collector.visit(self.neural_network)
        self.op_info = info_collector.op_info
        self.op_to_info_map = info_collector.op_to_info_map
        self.fused_neural_network_ = self.visit(self.neural_network)

    @property
    def fused_neural_network(self):
        return self.fused_neural_network_

    def visit_call(self, call):
        new_args = []
        for arg in call.args:
            new_args.append(self.visit(arg))
        
        if call.op.name == "nn.conv2d":
            # Replace the input of the `conv2d` with `self.new_input`
            op_info = self.op_to_info_map[call]
            conv_idx = op_info['conv_index']
            input_shape = op_info['input_shape']
            is_fusion_begin = False
            is_fusion_end = False

            for indices in self.fusion_indices:
                if indices[0] == conv_idx and indices[1] == conv_idx:
                    break
                if indices[0] == conv_idx:
                    is_fusion_begin = True
                    break
                if indices[1] == conv_idx:
                    is_fusion_end = True
                    break
            if is_fusion_begin:
                dummy_input_var = relay.var(f"dummy_input_{conv_idx}", shape=input_shape, dtype=call.args[0].checked_type.dtype)
                self.swap_var_to_node_input_arg_map[dummy_input_var] = new_args[0]
                self.cur_dummy_input_var = dummy_input_var
                modified_conv = relay.nn.conv2d(dummy_input_var, *new_args[1:], **call.attrs)
                self.cur_fusion_begin_node = modified_conv
                return modified_conv
            
            elif is_fusion_end:
                new_normal_conv = relay.nn.conv2d(*new_args, **call.attrs)
                # print("end conv_idx:", conv_idx)
                iteratee_func, conv_chain_params, cache_vars, \
                new_input_layout, new_input_stride, \
                output_shape, input_shape = create_fusion_block_iteratee_with_cache(new_normal_conv, 1, f'iteratee_{conv_idx}')
                # breakpoint()
                in_w = input_shape[2]
                in_h = input_shape[3]
                iter_begin = [0,0,0,0]
                iter_end = [0, 0, in_w - new_input_layout[0] + 1, in_h - new_input_layout[1] + 1]
                iter_strides = [1,1, *new_input_stride]

                iter_worker = fusion_iter_worker(iter_begin, iter_end, iter_strides, output_shape,
                                     conv_chain_params, iteratee_func, cache_vars=cache_vars)
                
                name_to_var = {self.cur_dummy_input_var.name_hint: self.swap_var_to_node_input_arg_map[self.cur_dummy_input_var]}

                iter_worker = ReWriteSwapVars(name_to_var).visit(iter_worker)

                # breakpoint()

                return iter_worker
            
            else:
                modified_conv = relay.nn.conv2d(*new_args, **call.attrs)
                return modified_conv
            
        elif call.op.name == "nn.avg_pool2d":
            op_info = self.op_to_info_map[call]
            conv_idx = op_info['conv_index']
            input_shape = op_info['input_shape']
            is_fusion_begin = False
            is_fusion_end = False

            for indices in self.fusion_indices:
                if indices[0] == conv_idx and indices[1] == conv_idx:
                    break
                if indices[0] == conv_idx:
                    is_fusion_begin = True
                    break
                if indices[1] == conv_idx:
                    is_fusion_end = True
                    break

            if is_fusion_end:
                new_normal_conv = relay.nn.avg_pool2d(*new_args, **call.attrs)
                iteratee_func, conv_chain_params, cache_vars, \
                new_input_layout, new_input_stride, \
                output_shape, input_shape = create_fusion_block_iteratee_with_cache(new_normal_conv, 1, f'iteratee_{conv_idx}')
                # breakpoint()
                in_w = input_shape[2]
                in_h = input_shape[3]
                iter_begin = [0,0,0,0]
                iter_end = [0, 0, in_w - new_input_layout[0] + 1, in_h - new_input_layout[1] + 1]
                iter_strides = [1,1, *new_input_stride]

                iter_worker = fusion_iter_worker(iter_begin, iter_end, iter_strides, output_shape,
                                     conv_chain_params, iteratee_func, cache_vars=cache_vars)
                
                name_to_var = {self.cur_dummy_input_var.name_hint: self.swap_var_to_node_input_arg_map[self.cur_dummy_input_var]}

                iter_worker = ReWriteSwapVars(name_to_var).visit(iter_worker)

                # breakpoint()

                return iter_worker
            
            else:
                modified_conv = relay.nn.avg_pool2d(*new_args, **call.attrs)
                return modified_conv

        return super().visit_call(call)

    

class MultiFusionConv2DWorker(relay.ExprMutator):
    def __init__(self, new_input):
        super().__init__()
        self.new_input = new_input  # The new input expression to replace the original input

    def visit_call(self, call):
        # Check if the call node is a `conv2d` operation
        for arg in call.args:
            self.visit(arg)
        if isinstance(call.op, relay.op.Op) and call.op.name == "nn.conv2d":
            # Replace the input of the `conv2d` with `self.new_input`
            modified_conv = relay.nn.conv2d(self.new_input, call.args[1], **call.attrs)
            return modified_conv

        # For other operations, continue visiting as usual
        return call