
__FUSION_WORKER_ID__ = 0
__FUSION_OP_NUM__ = 0

def get_fusion_worker_id():
    global __FUSION_WORKER_ID__
    return __FUSION_WORKER_ID__

def increase_fusion_worker_id():
    global __FUSION_WORKER_ID__
    __FUSION_WORKER_ID__ += 1

def clear_fusion_worker_id():
    global __FUSION_WORKER_ID__
    __FUSION_WORKER_ID__ = 0

def get_fusion_op_num():
    global __FUSION_OP_NUM__
    return __FUSION_OP_NUM__

def increase_fusion_op_num():
    global __FUSION_OP_NUM__
    __FUSION_OP_NUM__ += 1

def clear_fusion_op_num():
    global __FUSION_OP_NUM__
    __FUSION_OP_NUM__ = 0