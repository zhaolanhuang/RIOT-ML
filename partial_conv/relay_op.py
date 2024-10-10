import sys
import os
sys.path.append(os.getenv("TVM_HOME") + '/python')

import tvm
from tvm import relay
import tvm.te


# Define the new operator in Relay
relay.op.op.register("my_add")
def my_add(lhs, rhs):
    return relay.Call(relay.op.get("my_add"), [lhs, rhs])

# Define the compute function for the my_add operator
@relay.op.op.register_compute("my_add")
def my_add_compute(attrs, inputs, output_type):
    lhs, rhs = inputs
    return [tvm.te.compute(lhs.shape, lambda *i: lhs(*i) + rhs(*i), name="my_add")]

relay.op.op.register_shape_func("my_add", False, relay.op._tensor.elemwise_shape_func)


# Define the new operator in Relay
relay.op.op.register("iter_avg_pool")
def iter_avg_pool(input_elem, prev_output, dividend):
    return relay.Call(relay.op.get("iter_avg_pool"), [input_elem, prev_output, relay.const(dividend, dtype="float32")])

# Define the compute function for the my_add operator
@relay.op.op.register_compute("iter_avg_pool")
def iter_avg_pool_compute(attrs, inputs, output_type):
    lhs, rhs, d = inputs
    return [tvm.te.compute(lhs.shape, lambda *i: lhs(*i) + (rhs(*i) / d), name="iter_avg_pool")]

relay.op.op.register_shape_func("iter_avg_pool", False, relay.op._tensor.elemwise_shape_func)

# # Register the compute function for my_add
# @tvm.ir.register_op_attr("my_add", "FTVMCompute")
# def my_add_compute_strategy(attrs, inputs, out_type):
#     return my_add_compute(attrs, inputs, out_type)

# Define the shape function for the my_add operator
# def my_add_shape_func(attrs, inputs, out_type):
#     # Return the same shape as the inputs (since element-wise addition)
#     return [inputs[0].shape]

# Register the shape function for my_add
# @tvm.ir.register_op_attr("my_add", "FInferShape")
# def my_add_shape_strategy(attrs, inputs, out_type):
#     print("my_add_shape_strategy")
#     return my_add_shape_func(attrs, inputs, out_type)

# Define the type relation for the my_add operator
# @tvm.ir.register_op_attr("my_add", "FInferType")
# def my_add_type_rel(attrs, inputs, out_type):
#     print("my_add_type_rel")
#     return relay.ty.TensorType(inputs[0].shape, inputs[0].dtype)

if __name__=="__main__":
    import numpy as np
    from tvm import relay
    from tvm.relay import create_executor

    # Define the Relay expression using my_add
    x = relay.var("x", shape=(3,))
    y = relay.var("y", shape=(3,))
    z = my_add(x, y)

    # Create a function and module
    func = relay.Function([x, y], z)

    mod = tvm.IRModule.from_expr(func)
    breakpoint()
    mod = relay.transform.InferType()(mod)

    # Create an executor to run the Relay function
    ex = create_executor(mod=mod, kind="debug")

    # Input data
    x_data = np.array([1.0, 2.0, 3.0], dtype="float32")
    y_data = np.array([4.0, 5.0, 6.0], dtype="float32")

    # Evaluate the function
    result = ex.evaluate(func)(x_data, y_data)
    print("Result:", result)
