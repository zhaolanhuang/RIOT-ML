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
    print(export_path)
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
    export_minimal("./model_zoo/mnist_0.983_quantized.tflite", "nrf52840dk", "../temp/")