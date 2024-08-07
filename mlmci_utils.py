from utils import extract_io_vars_from_module, generate_model_params_files, generate_model_io_vars_files, generate_model_binding_files

def generate_mlmci_files(relay_module, mod_params, output_path="./"):
    module = relay_module
    params = mod_params
    tvm_input_vars, tvm_output_vars = extract_io_vars_from_module(module)
    
    # opt_params: some parameters will be merged into code, needs to be filtered out
    metadata = module.executor_codegen_metadata
    input_names = [str(i) for i in metadata.inputs]
    opt_params = {k:v for k, v in params.items() if k in input_names}

    input_vars = filter(lambda x: str(x['name']) not in params, tvm_input_vars)
    input_vars = list(input_vars)
    generate_model_io_vars_files(input_vars, tvm_output_vars, output_path)
    generate_model_params_files(opt_params, output_path)
    generate_model_binding_files(opt_params, input_vars, tvm_output_vars, output_path)
