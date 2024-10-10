import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay
from tvm.relay import dataflow_pattern as dfp
from tvm.relay.transform import function_pass, InferType
from tvm.micro import export_model_library_format

import copy

from .iter_func import iter_func
from .dyn_slice_fixed_size import dyn_slice_fixed_size

def copy_var_with_name_prefix(var, prefix='_'):
    return copy_var_with_new_name(var, prefix + str(var.name_hint))

def copy_var_with_new_name(var, name=None):
    return relay.Var(name, var.type_annotation)


class ReWriteInputsShape(relay.ExprMutator):
    """This pass partitions the subgraph based on the if conditin

    """
    def __init__(self, name_to_shape):
        super().__init__()
        self.name_to_shape = name_to_shape

    def visit_function(self, fn):
        """This function returns concatenated add operators for a one add operator.
        It creates multiple add operators.

        :param call:
        :return:
        """

        new_params = []
        for x in range(len(fn.params)):
            new_params.append(self.visit(fn.params[x]))

        new_body = self.visit(fn.body)
        func = relay.Function(list(new_params), new_body, fn.ret_type, fn.type_params, fn.attrs)
        return func

    def visit_var(self, var):
        if var.name_hint in self.name_to_shape:
            print(f'Change Shape of params {var.name_hint}, {var.type_annotation.shape} to {self.name_to_shape[var.name_hint]}')
            d = self.name_to_shape[var.name_hint]
            var_new = relay.var(var.name_hint, shape=d, dtype=var.type_annotation.dtype)
            return var_new
        else:
            print("Do nothing for other cases")
            return var
        
class ReWriteSwapVars(relay.ExprMutator):
    """This pass partitions the subgraph based on the if conditin

    """
    def __init__(self, name_to_var):
        super().__init__()
        self.name_to_shape = name_to_var

    def visit_var(self, var):
        if var.name_hint in self.name_to_shape:
            print(f'Change Shape of params {var.name_hint}, {var.type_annotation.shape} to {self.name_to_shape[var.name_hint]}')
            d = self.name_to_shape[var.name_hint]
            return d
        else:
            print("Do nothing for other cases")
            return var

def int_list(l):
    return [int(i) for i in l]

class CollectOpShapeInfo(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.op_info = []

    def collect_op_info(self, op):
        input_shape = tuple(int(dim) for dim in op.args[0].checked_type.shape)
        output_shape = tuple(int(dim) for dim in op.checked_type.shape)
        kernel_size = int_list([*getattr(op.attrs, 'kernel_size', getattr(op.attrs, 'pool_size', [1 ,1]))])
        padding = int_list([*getattr(op.attrs, 'padding', [0, 0])])
        strides = int_list([*getattr(op.attrs, 'strides', [1, 1])])

        self.op_info.append({
            'op_name': op.op.name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'kernel_size': kernel_size,
            'padding': padding,
            'strides': strides
        })
    def visit_call(self, call):
        self.collect_op_info(call)
        for arg in call.args:
            self.visit(arg)

    def op_info(self):
        return self.op_info

# Duplicated, we used user-define dynamic slice fixed size op
def DynamicStridedSlice(*args, **kwargs):
    return relay.op.dyn._make.strided_slice(*args, **kwargs)
class DynSlice:
    def __init__(self,in_shape, out_shape) -> None:
        slice_begin = relay.var("begin", shape=(4,), dtype="int32")
        self.strided = relay.const([1,1,1,1], "int32")
        self.slice_end = relay.const(out_shape, "int32")
        sliced_data = relay.var("sliced_data", shape=in_shape)
        
        body = DynamicStridedSlice(sliced_data, slice_begin, self.slice_end, self.strided,'size')
        self.func = relay.Function([sliced_data, slice_begin], body, 
                                    ).with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        self.ty = relay.transform.InferTypeLocal(body)

    def __call__(self, sliced_data, slice_begin, slice_end):
        return DynamicStridedSlice(sliced_data, slice_begin, self.slice_end, self.strided,'size')

class ConvChainPattern(dfp.DFPatternCallback):
    def __init__(self):
        super().__init__(require_type=True)
        
        # Define the patterns for conv2d, avg_pool2d, and dense
        self.conv2d = dfp.is_op("nn.conv2d")(dfp.wildcard(), dfp.wildcard())
        self.avg_pool2d = dfp.is_op("nn.avg_pool2d")(self.conv2d)
        self.reshape = dfp.is_op("reshape")(self.avg_pool2d | self.conv2d)
        self.dense = dfp.is_op("nn.dense")(self.avg_pool2d | self.conv2d | self.reshape, dfp.wildcard())

        # Define the overall pattern: multiple conv2d -> avg_pool2d -> dense
        self.pattern = self.dense

        # To record input/output shapes and other attributes
        self.op_info = []

        self.external_funcs = None

    def callback(self, pre, post, node_map):
        # Step 2: Record Input/Output Shapes and Attributes
        
        conv_nodes = node_map[self.conv2d]
        avg_pool_node = node_map[self.avg_pool2d][0]
        dense_node = node_map[self.dense][0]
        reshape_node = node_map[self.reshape][0]
        
        op_shape_collector = CollectOpShapeInfo()
        op_shape_collector.visit(dense_node)
        self.op_info = op_shape_collector.op_info

        is_reshape_before_dense = reshape_node is not None
        is_pool_node_before_dense = avg_pool_node is not None
        
        new_input_layout = [1 , 1]
        new_input_stride = [1 , 1]
        for i in self.op_info:
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
            # breakpoint()
            
            new_input_layout[0] = (new_input_layout[0] - 1) * s[0] - 2 * p[0] + d[0] * (k[0]-1) + 1
            new_input_layout[1] = (new_input_layout[1] - 1) * s[1] - 2 * p[1] + d[1] * (k[1]-1) + 1
            new_input_stride[0] = new_input_stride[0] * s[0]
            new_input_stride[1] = new_input_stride[1] * s[1]

        # Collect info for dense node (only shapes, no kernel/padding/stride)
        # collect_op_info(dense_node, 'dense')

        # Step 3: Wrap Conv Chain in an External Function
        # Extract the part of the graph to be wrapped in an external function
        conv_chain_node = avg_pool_node.args[0]
        conv_chain_node_cp = copy.deepcopy(conv_chain_node)

        conv_chain_params = relay.analysis.free_vars(conv_chain_node)
        conv_chain_params_cp =  relay.analysis.free_vars(conv_chain_node_cp)
        input_var = conv_chain_params_cp[0]
        input_shape = input_var.checked_type.shape
        in_w = int(input_shape[-2])
        in_h = int(input_shape[-1])

        output_shape = conv_chain_node.checked_type.shape
        new_output_shape = (output_shape[0],output_shape[1], 1 , 1)

        re_write_inputs = ReWriteInputsShape({'data': (input_shape[0],input_shape[1],new_input_layout[0],new_input_layout[1])})
        
        conv_chain_block = re_write_inputs.visit(conv_chain_node_cp)
        conv_chain_block_params = relay.analysis.free_vars(conv_chain_block)
     
        from .iterative_ops import iterative_global_avg_pool_step, IterativeGlobalAvgPool
        from .relay_op import iter_avg_pool

        slice_begin_var = relay.var("slice_begin", shape=(4,), dtype="int32")

        initial_output_var = relay.var("initial_output", shape=new_output_shape, dtype="float32")

        new_input_var = dyn_slice_fixed_size(input_var, slice_begin_var, [int(input_shape[0]),int(input_shape[1]),*new_input_layout])

        rewrite_swap_vars = ReWriteSwapVars({"data":new_input_var})
        conv_chain_block = rewrite_swap_vars.visit(conv_chain_block)

        iterative_output = iterative_global_avg_pool_step(conv_chain_block, initial_output_var, avg_dividend)

        params = [slice_begin_var, initial_output_var, *conv_chain_params_cp]

        iteratee_func = relay.Function(params, iterative_output).with_attr("Primitive", tvm.tir.IntImm("int32", 1)) \
                                    .with_attr("global_symbol", "iteratee") # keep that for stable func name

        slice_begin = relay.zeros(shape=(4,), dtype="int32")
        initial_output = relay.zeros(shape=new_output_shape, dtype="float32")
        initial_var = relay.Call(iteratee_func, [slice_begin, initial_output, *conv_chain_params])

        iter_begin = [0,0, *new_input_stride]
        iter_end = [0, 0, in_w - new_input_layout[0] + 1, in_h - new_input_layout[1] + 1]
        iter_strides = [1,1, *new_input_stride]
        iterative_output = iter_func(iter_begin, iter_end, iter_strides,[slice_begin, initial_var,*conv_chain_params], iteratee_func)
                
        if is_reshape_before_dense:
            output = relay.reshape(iterative_output, reshape_node.attrs.newshape)

        dense_weight = dense_node.args[1]
        output = relay.nn.dense(output, dense_weight)
        print(output.astext())


        return output

# Step 5: Rewrite the Graph and Define the External Function
def rewrite_conv_chain_to_function(mod):
    # Apply the pattern
    pattern = ConvChainPattern()
    
    # Run the pattern matcher and rewriter
    func = dfp.rewrite(pattern, mod["main"])

    # external_funcs = pattern.external_funcs

    # Add the external function to the IRModule
    mod["main"] = func
    # for gv, func in external_funcs:
    #     mod[gv] = func
    
    # Print the recorded shapes and attributes
    for info in pattern.op_info:
        print(f"Operation: {info['op_name']}")
        print(f"  Input Shape: {info['input_shape']}")
        print(f"  Output Shape: {info['output_shape']}")
        if info['kernel_size'] is not None:
            print(f"  Kernel Size: {info['kernel_size']}")
        if info['padding'] is not None:
            print(f"  Padding: {info['padding']}")
        if info['strides'] is not None:
            print(f"  Strides: {info['strides']}")
        print()

    return mod

def AnnotateUsedMemory():
    return relay.transform._ffi_api.AnnotateUsedMemory()

def AnnotateMemoryScope():
    return relay.transform._ffi_api.AnnotateMemoryScope()

# Example usage
if __name__ == "__main__":
    # Define a Relay module with a computation graph
    data = relay.var("data", shape=(1, 1, 64, 64))
    # data = relay.var("data", shape=(1, 3, 15, 15))
    weight1 = relay.var("weight1", shape=(64, 1, 7, 7))
    weight2 = relay.var("weight2", shape=(64, 64, 3, 3))
    weight3 = relay.var("weight3", shape=(64, 64, 3, 3))
    dense_weight = relay.var("dense_weight", shape=(10, 64))

    conv1 = relay.nn.conv2d(data, weight1, kernel_size=(7, 7))
    conv2 = relay.nn.conv2d(conv1, weight2, kernel_size=(3, 3))
    conv3 = relay.nn.conv2d(conv2, weight3, kernel_size=(3, 3))
    avg_pool = relay.nn.avg_pool2d(conv3, pool_size=(54, 54))
    reshape = relay.reshape(avg_pool, (1,64,))
    dense = relay.nn.dense(reshape, dense_weight)

    func = relay.Function([data, weight1, weight2, weight3, dense_weight], dense)
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
    TARGET = tvm.target.target.micro('nrf52840')

    from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll


    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                    "tir.usmp.algorithm": "hill_climb",
                                                    # "relay.backend.use_auto_scheduler": True, # Keep that for Primitive Function with multiple heavy ops (like Convs)
                                                    },
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()]
                                                    ): 
        # print(params.keys())
        # opt_module, _ = relay.optimize(mod, target=TARGET)
        module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)
    export_model_library_format(module, "./default.tar")
        # breakpoint()
    # func_mod = tvm.IRModule.from_expr(opt_module["main"])
    # func_mod = relay.transform.InferType()(func_mod)

    # func_mod = tvm.transform.Sequential([
    # # relay.transform.PartialEvaluate(),
    # # relay.transform.DeadCodeElimination(),
    # relay.transform.InferType(),
    # relay.transform.ToANormalForm(),
    # relay.transform.InferType(),
    # AnnotateUsedMemory(),
    # # relay.transform.RemoveUnusedFunctions(),
    # ])(opt_module)
    # print(func_mod['main'].params)
    # print(func_mod['main'].checked_type)
    # print(type(func_mod['main'].body))        
        # module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)
    # export_model_library_format(module, "./default.tar")
    
