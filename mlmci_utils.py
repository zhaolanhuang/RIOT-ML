from utils import extract_io_vars_from_module, generate_model_params_files, generate_model_io_vars_files, generate_model_binding_files

def generate_mlmci_files(relay_module, mod_params):
    module = relay_module
    params = mod_params
    input_vars, output_vars = extract_io_vars_from_module(module)
    input_vars = filter(lambda x: str(x['name']) not in params, input_vars)
    input_vars = list(input_vars)
    generate_model_io_vars_files(input_vars, output_vars)
    generate_model_params_files(params)
    generate_model_binding_files(params, input_vars, output_vars)
