import sys
import os
from typing import Any
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay

class IterativeGlobalAvgPool:
    def __init__(self, elem_shape, dividend=1.0) -> None:
        dividend = relay.const(dividend, dtype="float32")
        input_elem = relay.var("input_elem", shape=elem_shape)
        prev_output = relay.var("prev_output", shape=elem_shape)
        divided_input = relay.divide(input_elem, dividend)
        new_output = relay.add(prev_output, divided_input)
        self.func = relay.Function([prev_output, input_elem], new_output, 
                                    ret_type=relay.TensorType(elem_shape, "float32")
                                    ).with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    def __call__(self, prev_output, input_elem):
        return relay.Call(self.func, [prev_output, input_elem])

def iterative_global_avg_pool_step(input_elem, prev_output, dividend):
    """
    Iteratively computes one step of the global average pooling operation.
    
    Parameters:
    - input_elem: The current element from the input tensor (scalar).
    - prev_output: The previous cumulative sum.
    
    Returns:
    - The updated cumulative sum after the current step.
    """

    divided_input = relay.divide(input_elem, relay.const(dividend, dtype="float32"))
    # Add the current element to the cumulative sum
    new_output = relay.add(prev_output, divided_input)
    
    return new_output

def iterative_global_max_pool_step(input_elem, prev_output):
    """
    Iteratively computes one step of the global max pooling operation.
    
    Parameters:
    - input_elem: The current element from the input tensor (scalar).
    - prev_output: The previous maximum value.
    
    Returns:
    - The updated maximum value after the current step.
    """
    # Take the maximum of the current element and the previous maximum
    new_output = relay.maximum(prev_output, input_elem)
    
    return new_output

def iterative_dense_step(input_elem, weight, index, prev_output):
    """
    Iteratively computes one step of the dense operation.
    
    Parameters:
    - input_elem: The current element from the input tensor (scalar).
    - weight: The weight tensor for the dense operation.
    - index: The current index being processed.
    - prev_output: The previous cumulative result of the dense operation.
    
    Returns:
    - The updated cumulative result after the current step.
    """
    
    # Take the current row from the weight tensor using the index
    weight_row = relay.take(weight, index, axis=1)
    
    # Multiply the current input element with the weight row (element-wise)
    current_output = relay.multiply(input_elem, weight_row)
    
    # Add the current output to the previous cumulative output
    new_output = relay.add(prev_output, current_output)
    
    return new_output

# Example Relay Function to Execute Iterative Dense Operation
def iterative_dense(input_tensor, weight):
    """
    Executes the iterative dense operation for `num_steps` steps.
    
    Parameters:
    - input_tensor: The original input tensor for the dense operation.
    - weight: The weight tensor for the dense operation.
    - num_steps: The number of iterations to perform (usually the size of the input).
    
    Returns:
    - The final result of the iterative dense operation.
    """
    
    # Initialize an empty output tensor (zeros)
    shape = (weight.type_annotation.shape[0],)  # Get the number of output neurons (weight columns)
    initial_output = relay.zeros(shape=shape, dtype="float32")
    num_steps = int(weight.type_annotation.shape[1])
    # Loop through the input tensor iteratively
    output = initial_output
    for i in range(num_steps):
        # Extract the current input element
        input_elem = relay.take(input_tensor, relay.const(i), axis=0)
        
        # Perform one iteration of the dense operation
        output = iterative_dense_step(input_elem, weight, relay.const(i), output)
    
    return output

if __name__ == "__main__":
    # Example Usage
    input_tensor = relay.var("input", shape=(50,), dtype="float32")  # Example input tensor with shape (10,)
    weight = relay.var("weight", shape=(5, 50), dtype="float32")  # Example weight matrix with shape (10, 5)

    # Create the Relay function
    iterative_dense_func = iterative_dense(input_tensor, weight)
    mod = tvm.IRModule.from_expr(iterative_dense_func)
    mod = relay.transform.InferType()(mod)
    # breakpoint()

    import numpy as np
    from tvm.relay import create_executor

    # Example input data
    input_data = np.random.rand(50).astype("float32")
    weight_data = np.random.rand(5, 50).astype("float32")

    # Create the executor
    ex = create_executor(mod=mod, kind="debug")

    # Evaluate the function
    result = ex.evaluate()(input_data, weight_data)
    print(result)

    from tvm.ir.instrument import PrintAfterAll, PrintBeforeAll

    # with tvm.transform.PassContext(opt_level=3, instruments=[PrintBeforeAll(), PrintAfterAll()]):
    #     tvm.lower(mod)


    from tvm.micro import export_model_library_format
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

# Keep opt_level to 0, otherwise the `take` op will gather in the begining of the main function.
    with tvm.transform.PassContext(opt_level=0, config={
                                                    "tir.disable_vectorize": True, 
                                                    "tir.usmp.enable": True, # what is usmp? -> Enable Unified Static Memory Planning
                                                    "tir.usmp.algorithm": "hill_climb",
                                                    }, 
                                                    # instruments=[PrintBeforeAll(),PrintAfterAll()], 
                                                    # disabled_pass=["FuseOps"]
                                                    ): 
        # print(params.keys())
        module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=None, executor=EXECUTOR)

    # breakpoint()
    export_model_library_format(module, "./iterative_dense.tar")
