from connector import get_local_controller, get_fit_iotlab_controller
from riotctrl.ctrl import RIOTCtrl
from model_converter import generate_model_c_code
import os
import time
import numpy as np
import tempfile
import argparse
import json

MODEL_C_LIB_PATH = './models/default/default.tar'
COAP_SERVER_ROOT = './coaproot'
SUIT_COAP_MODEL_PARAMS_BASEPATH = 'model_params'

MAKE_FLASH_ONLY = 'flash-only'
MAKE_SUIT_KEYGEN = 'suit/genkey'
MAKE_SUIT_MANIFEST = 'suit/manifest'
MAKE_SUIT_PUBLISH = 'suit/publish'
MAKE_SUIT_NOTIFY = 'suit/notify'
MAKE_SUIT_NOTIFY_UPDATE_PARAMS = 'suit/notify/model_params'
MAKE_RIOTBOOT = 'riotboot'


def build_suit_firmware_payload(board):
    env = {'BOARD': board, 'UTOE_GRANULARITY' : '0', 
           'USE_SUIT': '1', 'RIOTBOOT_SKIP_COMPILE': '0'}
    make_ctrl = get_local_controller(env)
    make_ctrl.make_run((MAKE_RIOTBOOT,))

def build_suit_full_update_manifest(board, coap_server_ip):
    env = {'BOARD': board, 'UTOE_GRANULARITY' : '0', 
           'USE_SUIT': '1', 'RIOTBOOT_SKIP_COMPILE': '1', 
           'SUIT_COAP_SERVER': coap_server_ip}
    make_ctrl = get_local_controller(env)
    make_ctrl.make_run((MAKE_SUIT_MANIFEST,))
    

def build_suit_parmas_payload(params_json):
    pass

def suit_genkey():
    pass

def compile_suit_able_firmware(board, env=None):
    if env is None:
        env = {'BOARD': board, 'UTOE_GRANULARITY' : '0', 'USE_SUIT': '1'}
    make_ctrl = get_local_controller(env)
    make_ctrl.make_run(('-j8',))

def preprovision_suit_able_firmware(connector: RIOTCtrl, model_file_path, board, shape_dict=None):
    print("Load Model and Code Gen...")
    generate_model_c_code(model_file_path, board, MODEL_C_LIB_PATH, shape_dict)
    print("Load Model and Code Gen...done")
    print("Compile Firmware...")
    compile_suit_able_firmware(board, connector.env)
    print("Compile Firmware...done")
    print("Flash to Device...")
    connector.make_run((MAKE_FLASH_ONLY,))
    print("Flash to Device...done")

# By now Only support CoAP server on the local host
# Publish to $(server_root)/fw/$(APPLICATION)/$(BOARD)
def push_payload_and_manifest(server_root):
    env = {'UTOE_GRANULARITY' : '0', 
           'USE_SUIT': '1', 'RIOTBOOT_SKIP_COMPILE': '1', 
           'SUIT_COAP_FSROOT': server_root}
    make_ctrl = get_local_controller(env)
    make_ctrl.make_run((MAKE_SUIT_PUBLISH,))

def notify_full_update(client_ip, coap_server_ip):
    env = {'UTOE_GRANULARITY' : '0', 
           'USE_SUIT': '1', 'RIOTBOOT_SKIP_COMPILE': '1', 
           'SUIT_CLIENT': client_ip, 'SUIT_COAP_SERVER': coap_server_ip}
    make_ctrl = get_local_controller(env)
    make_ctrl.make_run((MAKE_SUIT_NOTIFY,))
    
# User should setup CoAP file server manually in advance
def full_update(client_ip, coap_server_ip, model_file_path, board, shape_dict=None):
    os.environ['BOARD'] = board
    os.environ['APP_VER'] = str(int(time.time()))
    os.environ['SUIT_CLIENT'] = client_ip
    os.environ['SUIT_COAP_SERVER'] = coap_server_ip

    print("Load Model and Code Gen...")
    generate_model_c_code(model_file_path, board, MODEL_C_LIB_PATH, shape_dict)
    print("Load Model and Code Gen...done")
    build_suit_firmware_payload(board)
    build_suit_full_update_manifest(board, coap_server_ip)
    push_payload_and_manifest(COAP_SERVER_ROOT)
    notify_full_update(client_ip, coap_server_ip)
    

def partial_update(client_ip, coap_server_ip, 
                   params_json: dict[str, np.ndarray], board):
    SEQ_NR = os.environ['APP_VER'] = str(int(time.time()))
    # params_path = os.path.join(COAP_SERVER_ROOT, SUIT_COAP_MODEL_PARAMS_BASEPATH)
    # if not os.path.exists(params_path):
    #     os.makedirs(params_path)
    params_path = tempfile.mkdtemp()   
    
    SUIT_MANIFEST_PAYLOADS = []
    SUIT_MANIFEST_SLOTFILES = []
    for k,v in params_json.items():
        filename = f'{k}.{SEQ_NR}.bin'
        filepath = os.path.join(params_path, filename)
        v.tofile(filepath)
        SUIT_MANIFEST_PAYLOADS.append(filepath)
        SUIT_MANIFEST_SLOTFILES.append(f'{filepath}:0:model:{k}')
    SUIT_MANIFEST_PAYLOADS = " ".join(SUIT_MANIFEST_PAYLOADS)
    SUIT_MANIFEST_SLOTFILES = " ".join(SUIT_MANIFEST_SLOTFILES)
    env = {'USE_SUIT': '1', 'RIOTBOOT_SKIP_COMPILE': '1', 
           'SUIT_CLIENT': client_ip, 'SUIT_COAP_SERVER': coap_server_ip,
           'SUIT_MANIFEST_PAYLOADS': SUIT_MANIFEST_PAYLOADS, 
           'SUIT_MANIFEST_SLOTFILES': SUIT_MANIFEST_SLOTFILES, 
           'SUIT_COAP_BASEPATH': SUIT_COAP_MODEL_PARAMS_BASEPATH,
           'SUIT_COAP_FSROOT': COAP_SERVER_ROOT, 'BOARD': board}

    make_ctrl = get_local_controller(env)
    make_ctrl.make_run((MAKE_SUIT_MANIFEST,))
    make_ctrl.make_run((MAKE_SUIT_PUBLISH,))
    make_ctrl.make_run((MAKE_SUIT_NOTIFY_UPDATE_PARAMS,))

#Input: Json file with: param names -> value lists
def read_json_params(file_path):
    with open(file_path, 'rb') as f:
        params_dict = json.load(f)
    return {k: np.array(v, dtype=np.byte) for k,v in params_dict.items()}

        


if __name__ == "__main__":
    # model_update.py --preprovision/--full/--partial --board --iotlab-node --client --server model_artifact
    # model_artifact: model file when preprovision or full update, 
    #                 json dict file(params name -> values list) when partial update
    # iotlab-node: only use for preprovision to iotlab
    # client, server: ipv6 addresses only use in full and partial update
    # USE_ETHOS and DEFAULT_CHANNEL should be pre-defined outside the python file
    # examples:
    # USE_ETHOS=0 DEFAULT_CHANNEL=26 python model_update.py --preprovision --board nrf52840dk --iotlab-node nrf52840dk-10.saclay.iot-lab.info ./model_zoo/mnist_0.983_quantized.tflite
    # USE_ETHOS=1 python model_update.py --full --board nrf52840dk --client [2001:db8::2] --server [2001:db8::1] ./model_zoo/mnist_0.983_quantized.tflite
    # python model_update.py --partial --board nrf52840dk --client [2001:db8::2] --server [2001:db8::1] ./params.json

    parser = argparse.ArgumentParser()
    parser.add_argument("model_artifact", help="path to model file for preprovision or full update;"
                        "path to json file with parameter dict (param name -> values in list) for partial update.",
                        type=str)
    func_group = parser.add_mutually_exclusive_group()
    func_group.add_argument("--preprovision", 
                            help="Preprovision device with SUIT-able firmware.",
                            action="store_true")
    func_group.add_argument("--full-update", 
                            help="Run full update of model.",
                            action="store_true")
    func_group.add_argument("--partial-update", 
                            help="Run partial update of model.",
                            action="store_true")
    parser.add_argument("--board", help="IoT board name", default="nrf52840dk",
                        type=str)
    parser.add_argument("--iotlab-node", help="remote node url. Only used when preprovisioning firmware to IoT lab node",
                        default=None)
    parser.add_argument("--client", help="IPv6 address of client received model update",
                        type=str)
    parser.add_argument("--server", help="IPv6 address of CoAP server as SUIT-artifacts repository.",
                        type=str)
    args = parser.parse_args()
    
    artifact_path = args.model_artifact
    BOARD = args.board
    env = {'BOARD': BOARD, 'UTOE_GRANULARITY' : '0', 'USE_SUIT': '1'}
    if args.iotlab_node is not None:
        conn = get_fit_iotlab_controller(env, iotlab_node=args.iotlab_node)
    else:
        conn = get_local_controller(env)
    
    if args.preprovision:
        preprovision_suit_able_firmware(conn, artifact_path, BOARD)
    elif args.full_update:
        full_update(args.client, args.server, artifact_path, BOARD)
    elif args.partial_update:
        partial_update(args.client, args.server, 
                       read_json_params(artifact_path), BOARD)
    else:
        print("Unsupport Operation!")

    DEBUG = 0
    if DEBUG == 1:

        BOARD = 'nrf52840dk'
        model_path = './model_zoo/mnist_0.983_quantized.tflite'
        # env = {'BOARD': BOARD, 'UTOE_GRANULARITY' : '0', 
        #        'USE_SUIT': '1', }
        # conn = get_local_controller(env)
        env = {'BOARD': BOARD, 'UTOE_GRANULARITY' : '0', 'USE_SUIT': '1'}
        os.environ['USE_ETHOS'] = '0'
        os.environ['DEFAULT_CHANNEL'] = '26'
        conn = get_fit_iotlab_controller(env, iotlab_node='nrf52840dk-10.saclay.iot-lab.info')

        # preprovision_suit_able_firmware(conn, model_path, BOARD)
        
        # full_update('[2001:db8::64fa:5ffe:7879:4ad9]','[2001:db8::1]', model_path, BOARD)
        dummy_params = {'_param_1': np.array(list(range(150)), dtype=np.byte)}
        partial_update('[2001:db8::64fa:5ffe:7879:4ad9]','[2001:db8::1]', dummy_params, BOARD)