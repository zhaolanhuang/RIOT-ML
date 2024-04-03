from riotctrl.ctrl import RIOTCtrl
import re
import analysis
from tabulate import tabulate
from datetime import datetime
from connector import get_local_controller, get_fit_iotlab_controller
import json
from model_converter import load_model, compile_per_model_eval, load_from_tflite, compile_per_ops_eval
from utils import generate_model_io_vars_header, extract_io_vars_from_module, _shape_to_size, generate_model_params_files, generate_model_io_vars_files, generate_model_binding_files
from microtvm_transport import UTOETransport
import tvm
from functools import reduce
from mlmci_utils import generate_mlmci_files

LOG_DIR = './logs'

def evaluate_per_model(model_path, board='stm32f746g-disco', trials_num=10, use_iotlab=False,
                       iotlab_node=None, random_seed=42,
                       shape_dict=None):
    print("Load Model and Code Gen...")
    mod, params = load_model(model_path, shape_dict)
    moudle = compile_per_model_eval(mod, params, board, './models/default/default.tar')
    generate_mlmci_files(moudle, params)
    print("Load Model and Code Gen...done")

    env = {'BOARD': board, 'UTOE_TRIAL_NUM': str(trials_num), 'UTOE_RANDOM_SEED': str(random_seed)}
    print('Compile and Flashing...')

    if use_iotlab or iotlab_node is not None:
        riot_ctrl = get_fit_iotlab_controller(env, iotlab_node=iotlab_node)
        if iotlab_node is not None:
            riot_ctrl.flash(stdout=None)
    else:
        riot_ctrl = get_local_controller(env)
        riot_ctrl.flash(stdout=None, stderr=None)

    print('Compile and Flashing...done')
    term_retry_times = 2
    with riot_ctrl.run_term(reset=True): #reset should be false for risc v
        while term_retry_times > 0 :
            try:
                # riot_ctrl.term.expect_exact('start >')
                riot_ctrl.term.sendline('s')
                riot_ctrl.term.expect_exact('finished >',timeout=25)
                break
            except:
                print("Exception Occured, term buffer:")
                print(riot_ctrl.term.before)
                term_retry_times -= 1
                print("Retrying...")
        raw_output = riot_ctrl.term.before
        riot_ctrl.stop_exp()

    evaluation_record = {'board' : env['BOARD'], 'datetime': datetime.now().strftime("%Y%m%d-%H%M%S"),
                         'memory': 0, 'storage': 0,
                         'trials_record': None, 'trials_stats': None,
                         'model_path': model_path, 'random_seed': random_seed, 'mode': 'per-model'}

    evaluation_record['trials_record'] = parse_per_model_output(raw_output)
    evaluation_record['trials_stats_in_usec'] = analysis.analysis_compute_latency(evaluation_record['trials_record'])
    
    memory, storage = get_memory_storage_from_buildsize(board)
    evaluation_record['memory'] = memory
    evaluation_record['storage'] = storage
    
    print_per_model_evaluation(evaluation_record.copy())
    save_evaluation_record(evaluation_record)

def parse_per_model_output(raw_output : str):
    pattern = re.compile('trial: ([0-9]+), usec: ([0-9]+), ret: ([0-9]+)')
    results_list = pattern.findall(raw_output)
    return { 'trial' : [int(x[0]) for x in results_list],
             'usec' : [int(x[1]) for x in results_list],
             'ret' :  [int(x[2]) for x in results_list]  }

import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def save_evaluation_record(rec, log_dir=LOG_DIR):
    import os
    os.makedirs(log_dir, exist_ok=True)
    file_name = rec['board'] + '_' + str(rec['datetime']) + '.json'
    file_path = LOG_DIR + '/' + file_name
    with open(file_path, 'w') as f:
        json.dump(rec, f, cls=NpEncoder)


def get_memory_storage_from_buildsize(board):
    import subprocess
    output = subprocess.check_output(f"make BOARD={board} info-buildsize", shell=True)
    output = output.split(b'\n')
    output = output[1].split(b'\t')
    text = int(output[0].strip())
    data = int(output[1].strip())
    bss = int(output[2].strip())
    memory = data + bss
    storage = text + data
    return memory, storage


def print_per_model_evaluation(rec):
    headers = ['Board', 'Memory (KB)', 'Storage (KB)', 
               '95-CI (ms)', 'Mean (ms)', 'Median (ms)', 'Min. (ms)', 'Max. (ms)']
    
    if not isinstance(rec, list):
        rec = [rec]
    
    output_list = []
    for r in rec:
        stats = r['trials_stats_in_usec']
        ci = [0, 0]
        ci[0] = int(stats['95ci'][0]) / 1e3
        ci[1] = int(stats['95ci'][1]) / 1e3

        mean = int(stats['mean']) / 1e3
        median = int(stats['median']) / 1e3
        min = int(stats['min']) / 1e3
        max = int(stats['max']) / 1e3
        output_list.append([r['board'], r['memory'] / 1e3, r['storage'] / 1e3,
                                ci, mean, median, min ,max])
    
    tabular_output = tabulate(output_list, headers=headers)
    print(tabular_output)

def load_logs_from_folder(dir_path):
    import glob
    json_dict = []
    for filepath in glob.iglob(f'{dir_path}/*'):
        with open(filepath, 'r') as f:
            json_dict.append(json.load(f))
    print_per_model_evaluation(json_dict)


def evaluate_per_operator(model_path, board='stm32f746g-disco', use_iotlab=False, iotlab_node=None,
                          shape_dict=None):
    print("Load Model and Code Gen...")
    mod, params = load_model(model_path, shape_dict)
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    module = compile_per_ops_eval(mod, params, board, './models/default/default.tar')
    dummy_mod = compile_per_ops_eval(mod, params, board,link_params=False)
    print("Load Model and Code Gen...done")
    env = {'BOARD': board, 'UTOE_GRANULARITY' : '1'}
    
    print('Compile and Flashing...')
    if use_iotlab or iotlab_node is not None:
        riot_ctrl = get_fit_iotlab_controller(env, iotlab_node=iotlab_node)
        if iotlab_node is not None:
            riot_ctrl.flash(stdout=None)
    else:
        riot_ctrl = get_local_controller(env)
        riot_ctrl.flash(stdout=None, stderr=None)
    print('Compile and Flashing...done')
    with tvm.micro.Session(UTOETransport(riot_ctrl=riot_ctrl)) as session:
        debug_module = tvm.micro.create_local_debug_executor(
            module.get_graph_json(), session.get_system_lib(), session.device
        )
        debug_module.run()
        nodes_list = debug_module.debug_datum.get_graph_nodes()
        time_list = debug_module.debug_datum._time_list

    eval_data = parse_per_ops_result(nodes_list, time_list)

    from tvm.contrib.debugger.debug_result import DebugResult

    dbg_rslt = DebugResult(dummy_mod.get_graph_json(), '.')

    nodes_info = dbg_rslt.get_graph_nodes()
    ops_rec = {}

    params_info = get_params_info(nodes_info)
    for node_data in eval_data:
        op = node_data[1]
        if op == "__nop":
            continue

        op_node = [n for n in nodes_info if n["op"] == op][0]
        op_params = list(filter(lambda x : x != "reshape_nop",op_node["inputs"]))
        ops_rec[op] = {'time_us': node_data[2], 'time_percent': node_data[3],
                                    'params': op_params, 'memory': None, 'storage': None}
        
        workspace_sizes = module.function_metadata[op].workspace_sizes
        workspace_sizes = reduce(lambda x,y: x + y, workspace_sizes.values())
        io_sizes = module.function_metadata[op].io_sizes
        io_sizes = reduce(lambda x,y: x + y, io_sizes.values())

        #TODO: seperate const and input / output var
        ops_rec[op]['memory'] = workspace_sizes + io_sizes
        ops_rec[op]['storage'] = sum(map(lambda x: params_info[x]['bytes'] 
                                         if params_info.get(x) is not None else 0, 
                                         op_params))

    
    print_per_ops_evaluation(ops_rec)

    rec = {'board' : env['BOARD'], 'datetime': datetime.now().strftime("%Y%m%d-%H%M%S"),
        'ops_record': ops_rec, 'params_info': params_info,
        'model_path': model_path, 'mode': 'per-ops',
        }
    
    # save_evaluation_record(rec)

def print_per_ops_evaluation(rec):
    headers = ['Ops', 'Time (us)', 'Time (%)', 
               'Params', 'Memory (KB)', 'Storage (KB)'] 
    output_list = []
    for k,v in rec.items():
        output = [k, v['time_us'], v['time_percent'], v['params'], v['memory'] / 1e3 , v['storage'] / 1e3]
        output_list.append(output)
    tabular_output = tabulate(output_list, headers=headers)
    print(tabular_output)

def parse_per_ops_result(nodes_list, time_list):
    eid = 0
    data = []
    total_time = sum([np.mean(time) for time in time_list])
    for node, time in zip(nodes_list, time_list):
        time_mean = np.mean(time)
        num_outputs = 1 if node["op"] == "param" else int(node["attrs"]["num_outputs"])
        for j in range(num_outputs):
            op = node["op"]
            if node["op"] == "param":
                eid += 1
                continue
            name = node["name"]
            shape = None
            time_us = round(time_mean * 1e6, 3)
            time_percent = round(((time_mean / total_time) * 100), 3)
            inputs = str(node["attrs"]["num_inputs"])
            outputs = str(node["attrs"]["num_outputs"])
            measurements = str([round(repeat_data * 1e6, 3) for repeat_data in time])
            node_data = [name, op, time_us, time_percent, shape, inputs, outputs, measurements]
            data.append(node_data)
            eid += 1
    return data

def get_params_info(nodes_info):
    params_info = {}
    for n in nodes_info:
        if n['op'] != 'param':
            continue
        name = n['name']
        dtype = n['attrs']['T'][6:]
        shape = n['shape']
        params_info[name] = {'dtype': dtype, 'shape': shape,
                            'bytes': _shape_to_size(shape, dtype),
                            }
    return params_info

def memory_analysis(model_path, board='stm32f746g-disco',
                       shape_dict=None):
    print("Load Model and Code Gen...")
    mod, params = load_model(model_path, shape_dict)
    moudle = compile_per_model_eval(mod, params, board, './models/default/default.tar')
    generate_mlmci_files(moudle, params)
    print("Load Model and Code Gen...done")

    env = {'BOARD': board}
    print('Compiling...')
    riot_ctrl = get_local_controller(env)
    riot_ctrl.cosy()


    

if __name__ == '__main__':
    # from model_converter import RIOT_BOARD_TO_TARGET
    # for board in RIOT_BOARD_TO_TARGET.keys():
    #     try:
    #         evaluate_per_model(model_path='./model_zoo/mnist_0.983_quantized.tflite', 
    #                         board=board, use_iotlab=True, iotlab_node=None)
    #     except:
    #         print(f'Evaluation Failed: {board}')
    evaluate_per_model(model_path='./model_zoo/mnist_0.983_quantized.tflite', 
                            board='stm32f746g-disco', use_iotlab=False, iotlab_node=None)
        
    # evaluate_per_operator(model_path='./model_zoo/sinus_float.tflite', 
    #                         board='stm32f746g-disco', use_iotlab=False, iotlab_node=None)