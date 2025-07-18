from connector import get_local_controller, get_fit_iotlab_controller
import json
import os
import copy
import glob
import shutil
import re

from DeepSliding.model.cECG_CNN import cECG_CNN
from DeepSliding.model.CET import CET_S
from DeepSliding.model.ResTCN import ResTCN
from DeepSliding.model.TEMPONet import TEMPONet
from DeepSliding.model.tinychirp_cnn_time import TinyChirpCNNTime
from DeepSliding.model.tinychirp_transformer_time import TinyChirpTransformerTime

from pathlib import Path



CLS_OF_MODELS = [
    # cECG_CNN,
    # CET_S,
    ResTCN,
    # TEMPONet,
    # TinyChirpTransformerTime,
    # TinyChirpCNNTime
]

def parse_per_model_output(raw_output : str, heading=""):
    pattern = re.compile(f'{heading}trial: ([0-9]+), usec: ([0-9]+), ret: ([0-9]+)')
    results_list = pattern.findall(raw_output)
    return { 'trial' : [int(x[0]) for x in results_list],
             'usec' : [int(x[1]) for x in results_list],
             'ret' :  [int(x[2]) for x in results_list]  }

def main():
    MODELS_PATH = "./DeepSliding_TVM_model/stm32f746g-disco-bf16/"
    board = "stm32f746g-disco"
    os.environ['UTOE_ONLY'] = '1'
    os.environ['DS_SSM_MODEL'] = '1'
    env = {'BOARD': board, 'UTOE_TRIAL_NUM': str(1), 'UTOE_RANDOM_SEED': str(42),
        #    'PORT': "/dev/ttyACM0",
           }    

    hil_rslts = []

    for i in range(0, 100, 10):
        overlap_r = i / 100
        for cls in CLS_OF_MODELS:
            model_dir_name = f"r_{overlap_r}/{cls.__name__}"
            model_dir_path = os.path.join(MODELS_PATH, model_dir_name)
            for file in glob.glob(model_dir_path + r'/model_*'):
                shutil.copy(file, "./")
            shutil.copy(model_dir_path + "/default.tar", "./models/default/default.tar")
            riot_ctrl = get_local_controller(env)
            process_obj = riot_ctrl.flash(stdout=None, stderr=None)
            if process_obj.returncode == 0:
                term_retry_times = 2
                with riot_ctrl.run_term(reset=True): #reset should be false for risc v and native?
                    while term_retry_times > 0 :
                        try:
                            # riot_ctrl.term.expect_exact('start >')
                            riot_ctrl.term.sendline('s')
                            riot_ctrl.term.expect_exact('finished >',timeout=30)
                            break
                        except:
                            print("Exception Occured, term buffer:")
                            print(riot_ctrl.term.before)
                            term_retry_times -= 1
                            print("Retrying...")
                    raw_output = riot_ctrl.term.before
                riot_ctrl.stop_exp()
            else:
                raw_output = ""
            
            init_w_eval_rslt = parse_per_model_output(raw_output, "initial window ")
            overlap_w_eval_rslt = parse_per_model_output(raw_output, "overlap window ")
            
            new_rslt = {"board": board, "overlap_rate": overlap_r, "model": cls.__name__}
            new_rslt['initial_window_eval_record'] = init_w_eval_rslt
            new_rslt['overlap_window_eval_record'] = overlap_w_eval_rslt
            new_rslt['initial_window_usec'] = init_w_eval_rslt['usec'][-1] if len(init_w_eval_rslt['usec']) > 0 else -1
            new_rslt['overlap_window_usec'] = overlap_w_eval_rslt['usec'][-1] if len(overlap_w_eval_rslt['usec']) > 0 else -1

            hil_rslts.append(new_rslt)
            with open(f"./DS_SSM_eval_result_{board}_ResTCN_bf16.json", "w") as f:
                json.dump(hil_rslts, f)
    

if __name__ == "__main__":
    main()