import argparse
import os
from DeepSliding.tvm_pytorch_frontend import load_model, set_bf16
from model_converter import set_global_model_loader, generate_model_c_code

from DeepSliding.model.cECG_CNN import cECG_CNN
from DeepSliding.model.CET import CET_S
from DeepSliding.model.ResTCN import ResTCN
from DeepSliding.model.TEMPONet import TEMPONet
from DeepSliding.model.tinychirp_cnn_time import TinyChirpCNNTime
from DeepSliding.model.tinychirp_transformer_time import TinyChirpTransformerTime

from pathlib import Path



CLS_OF_MODELS = [
    cECG_CNN,
    CET_S,
    ResTCN,
    TEMPONet,
    TinyChirpTransformerTime,
    TinyChirpCNNTime
]

EXPORT_DIR = "./DeepSliding_TVM_model/stm32f746g-disco-bf16/"

PT_MODEL_DIR = "./DeepSliding/artifact/"

if __name__ == "__main__":
    set_global_model_loader(load_model)
    set_bf16(True)
    for i in range(0, 100, 10):
        overlap_r = i / 100
        for cls in CLS_OF_MODELS:
            SSM_PT_MODEL_PATH = PT_MODEL_DIR + f"r_{overlap_r}/ssm_{cls.__name__}.pth"
            EXPORT_PATH = EXPORT_DIR + f"r_{overlap_r}/{cls.__name__}/"
            Path(EXPORT_PATH).mkdir(parents=True, exist_ok=True)
            generate_model_c_code(SSM_PT_MODEL_PATH, "stm32f746g-disco", EXPORT_PATH + "default.tar", EXPORT_PATH, {'input': list(cls.DEFAULT_INPUT_SHAPE[:-1]) + [1]} )


