from model_converter import generate_model_c_code
from string import Template
import shutil
import os
from pathlib import Path


MIN_REQ_FILES = [
    "models/default/Makefile",
    "models/default/Makefile.include",
    "mlmci",
    "model_registry.c"
]

def export_minimal(model_file_path, board, export_path, shape_dict=None):
    model_export_path = export_path + "/models/default/"
    os.makedirs(export_path, exist_ok=True)
    os.makedirs(model_export_path, exist_ok=True)


    generate_model_c_code(model_file_path, board, model_export_path + "default.tar", export_path, shape_dict)
    for src in MIN_REQ_FILES:
        if os.path.isdir(src):
            shutil.copytree(src, export_path + "/" + src, dirs_exist_ok=True)
        else:
            
            os.makedirs(export_path + "/" + os.path.dirname(src), exist_ok=True)
            shutil.copy(src, export_path + "/" + src)
            
    
    shutil.copy("template/standalone_minimal/main.c", export_path + "/main.c")
    shutil.copy("template/standalone_minimal/Makefile", export_path + "/Makefile")
    s = Path(export_path + "/Makefile").read_text()
    s = Template(s).safe_substitute({'board': board})
    Path(export_path + "/Makefile").write_text(s)


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="path to machine learning model file.",
                        type=str)
    parser.add_argument("export_path", help="export path of standalone instance folder",
                        type=str)
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--minimal", help="Minimal Standalone. (default)",
                            action="store_true")
    mode_group.add_argument("--suit", help="Standalone with SUIT support. (Not supported yet)",
                            action="store_true")
 
    parser.add_argument("--board", help="IoT board name", default="stm32f746g-disco",
                        type=str)
    parser.add_argument("--input-shape", default=None, type=lambda s: [int(i) for i in s.split(',')], 
                        help="specify the input shape, mandatory for pytorch model. format: N,C,W,H. default: None")
    parser.add_argument("--odt", help="Enable on-device training. Please specify the configuration in odt_config.yml. (Not supported yet)",
                        action="store_true")
    args = parser.parse_args()
    export_minimal(args.model_file, args.board, args.export_path + "/", {'input': args.input_shape} if args.input_shape is not None else None)
    
    # export_minimal("./model_zoo/mnist_0.983_quantized.tflite", "nrf52840dk", "../temp/")