from string import Template
import re
import numpy as np

def _shape_to_size(shape, dtype):
    bits_per_item = int(
        re.match(r"((float)|(int)|(uint))(?P<width_bits>[0-9]+)", dtype).group("width_bits")
    )
    assert bits_per_item is not None, f"don't know how to compute size of type {dtype}"
    total_bits = bits_per_item
    for s in shape:
        total_bits *= s

    return (total_bits + 7) // 8


def generate_model_io_vars_header(input_vars : dict, output_vars : dict, output_path='./model_io_vars.h'):
    HEADER_TEMPLATE = """ /* U-TOE Auto-generated File */
    static char input[$input_size];
    static char output[$output_size];
    struct tvmgen_default_inputs default_inputs = {
            $input_fields
        };
    struct tvmgen_default_outputs default_outputs = {
            $output_fields
        };
    """
    def get_fields_and_size(vars, pool_var_name=None):
        fields = ""
        size = 0
        for v in vars:
            fields += f'.{v["name"]}' + f' = &{pool_var_name}[{size}], \n'
            size += v['size']
        return fields, size
    
    input_fields, input_size = get_fields_and_size(input_vars, 'input')
    output_fields, output_size = get_fields_and_size(output_vars, 'output')
    io_vars_header = Template(HEADER_TEMPLATE).substitute(input_size=input_size, output_size=output_size,
                                                          input_fields=input_fields, output_fields=output_fields)
    with open(output_path, "w") as f:
        f.write(io_vars_header)
    return io_vars_header

def extract_io_vars_from_module(mod):
    metadata = mod.executor_codegen_metadata
    input_names = metadata.inputs
    output_names = metadata.outputs
    
    input_vars = [{'name' : name, 'size': _shape_to_size(tensor_type.shape, tensor_type.dtype)}
                  for name,tensor_type in zip(input_names, metadata.input_tensor_types)]
    
    output_vars = [{'name' : name, 'size': _shape_to_size(tensor_type.shape, tensor_type.dtype)}
                  for name,tensor_type in zip(output_names, metadata.output_tensor_types)]
    return input_vars, output_vars


def _npy_dtype_to_ctype(data: np.ndarray) -> str:
    if data.dtype == "int8":
        return "int8_t"
    elif data.dtype == "int32":
        return "int32_t"
    elif data.dtype == "uint8":
        return "uint8_t"
    elif data.dtype == "float32":
        return "float"
    else:
        raise ValueError(f"Data type {data.dtype} not expected.")
    

def generate_model_params_files(
    params: dict, output_path: str = "./", const: bool = True
):

    HEADER_FILE_NAME = "model_params.h"
    C_FILE_NAME = "model_params.c"
    BYTE_ALIGNMENT = 4  
    AUTOGEN_STATEMENT = "/* U-TOE Generated File */ \n"
    npy_data = None
    tensor_name = None
    const = "const " if const else ""
    c_file = open(output_path + C_FILE_NAME, "w")
    c_file.write(AUTOGEN_STATEMENT)
    c_file.write("#include <stddef.h>\n")
    c_file.write("#include <stdint.h>\n")
    c_file.write("#include \"mlmci.h\"\n")

    # Generate Const Params array 
    for tensor_name, v in params.items():
        npy_data = v.numpy()
        c_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")
        c_file.write(f"__attribute__ ((aligned({BYTE_ALIGNMENT})))\n")
        c_file.write(f"{const}{_npy_dtype_to_ctype(npy_data)} {tensor_name}[] =")
        c_file.write("{")
        for i in np.ndindex(npy_data.shape):
            c_file.write(f"{npy_data[i]}, ")
        c_file.write("};\n")
    c_file.write("static const mlmodel_param_t _model_params[] = { \n")
    for tensor_name, v in params.items():
        npy_data = v.numpy()
        c_file.write("{")
        c_file.write(f".name = \"{tensor_name}\",")
        c_file.write(f".values = (uint8_t*){tensor_name},")
        c_file.write(f".num_bytes = sizeof({tensor_name}),")
        c_file.write(f".persistent_values = (uint8_t*){tensor_name},")
        c_file.write(f".volatile_values = NULL,")
        c_file.write("},\n")
    c_file.write("};\n")
    c_file.write("const mlmodel_param_t* get_model_params(void) {\n")
    c_file.write("return _model_params;\n}\n")
    c_file.close()

    h_file = open(output_path + HEADER_FILE_NAME, "w")
    h_file.write(AUTOGEN_STATEMENT)
    h_file.write("#ifndef MODEL_PARAMS_H \n")
    h_file.write("#define MODEL_PARAMS_H\n")
    h_file.write(f"#define MODEL_PARAMS_NUM {len(params)}\n")
    h_file.write("const mlmodel_param_t* get_model_params(void);\n")
    h_file.write("#endif")
    h_file.close()


def generate_model_io_vars_files(
    input_vars: list, output_vars: list,output_path: str = "./"
):
    HEADER_FILE_NAME = "model_iovars.h"
    C_FILE_NAME = "model_iovars.c"
    BYTE_ALIGNMENT = 4  
    AUTOGEN_STATEMENT = "/* U-TOE Generated File */ \n"

    c_file = open(output_path + C_FILE_NAME, "w")
    c_file.write(AUTOGEN_STATEMENT)
    c_file.write("#include <stddef.h>\n")
    c_file.write("#include <stdint.h>\n")
    c_file.write("#include \"mlmci.h\"\n")

    for v in [*input_vars, *output_vars]:
        c_file.write(f"__attribute__ ((aligned({BYTE_ALIGNMENT})))\n")
        c_file.write(f"static uint8_t {str(v['name'])}[{v['size']}]; \n")

    c_file.write("static const mlmodel_iovar_t _model_input_vars[] = { \n")
    for v in input_vars:
        c_file.write("{")
        c_file.write(f".name = \"{str(v['name'])}\",")
        c_file.write(f".values = (uint8_t*){str(v['name'])},")
        c_file.write(f".num_bytes = sizeof({str(v['name'])}),")
        c_file.write("},\n")
    c_file.write("};\n")

    c_file.write("static const mlmodel_iovar_t _model_output_vars[] = { \n")
    for v in output_vars:
        c_file.write("{")
        c_file.write(f".name = \"{str(v['name'])}\",")
        c_file.write(f".values = (uint8_t*){str(v['name'])},")
        c_file.write(f".num_bytes = sizeof({str(v['name'])}),")
        c_file.write("},\n")
    c_file.write("};\n")

    c_file.write("const mlmodel_iovar_t* get_model_input_vars(void) {\n")
    c_file.write("return _model_input_vars;\n}\n")

    c_file.write("const mlmodel_iovar_t* get_model_output_vars(void) {\n")
    c_file.write("return _model_output_vars;\n}\n")
    c_file.close()

    h_file = open(output_path + HEADER_FILE_NAME, "w")
    h_file.write(AUTOGEN_STATEMENT)
    h_file.write("#ifndef MODEL_IOVARS_H \n")
    h_file.write("#define MODEL_IOVARS_H\n")
    h_file.write(f"#define MODEL_INPUTS_NUM {len(input_vars)}\n")
    h_file.write(f"#define MODEL_OUTPUTS_NUM {len(output_vars)}\n")

    h_file.write("const mlmodel_iovar_t* get_model_input_vars(void);\n")
    h_file.write("const mlmodel_iovar_t* get_model_output_vars(void);\n")
    h_file.write("#endif")
    h_file.close()

def generate_model_binding_files(
        params: dict, input_vars: list, output_vars: list,output_path: str = "./"
):
    HEADER_FILE_NAME = "model_binding.h"
    AUTOGEN_STATEMENT = "/* U-TOE Generated File */ \n"
    h_file = open(output_path + HEADER_FILE_NAME, "w")
    h_file.write(AUTOGEN_STATEMENT)
    h_file.write("#ifndef MODEL_BINDING_H \n")
    h_file.write("#define MODEL_BINDING_H \n")
    h_file.write("static inline void model_bind_tvm_iovars(const mlmodel_t *model, "
                 "struct tvmgen_default_inputs *inputs, struct tvmgen_default_outputs *outputs) { \n")
    idx = 0
    for k in params.keys():
        h_file.write(f"inputs->{k} = model->params[{idx}].values; \n")
        idx += 1
    
    idx = 0
    for v in input_vars:
        h_file.write(f"inputs->{str(v['name'])} = model->input_vars[{idx}].values; \n")
        idx += 1
    
    idx = 0
    for v in output_vars:
        h_file.write(f"outputs->{str(v['name'])} = model->output_vars[{idx}].values; \n")
        idx += 1
    
    h_file.write("}\n")

    h_file.write("#endif")
    h_file.close()
