import tvm
from tvm.ir import register_intrin_lowering, Op, register_op_attr

# GlobalVar = tvm.ir.GlobalVar

# DEFAULT_TYPE = tvm.ir.FuncType([], None)
# According to the following C-Macro

#define STUB_LABEL(id, num) STUB_##id##_##num

#define INSERT_STUB(worker_id, op_num) STUB_LABEL(worker_id, op_num):

#define GOTO_STUB(worker_id, op_num) goto STUB_LABEL(worker_id, op_num)
INSERT_STUB = "INSERT_STUB"
GOTO_STUB = "GOTO_STUB"
  

#define WAIT_FOR_VAR(worker_id) fusion_worker_##worker_id##_wait_for_op

#define DECLEAR_EXTERN_WAIT_FOR_VAR(worker_id) extern int WAIT_FOR_VAR(worker_id)

#define SET_WAIT_FOR_OP_FROM(worker_id, op_num) WAIT_FOR_VAR(worker_id) = op_num

#define CLEAR_WAIT_FOR_OP_FROM(worker_id, op_num) WAIT_FOR_VAR(worker_id) = -1
DECLEAR_EXTERN_WAIT_FOR_VAR = "DECLEAR_EXTERN_WAIT_FOR_VAR"
SET_WAIT_FOR_OP_FROM = "SET_WAIT_FOR_OP_FROM"
CLEAR_WAIT_FOR_OP_FROM = "CLEAR_WAIT_FOR_OP_FROM"
  

#define CHECK_IF_SKIP_COMPUTE(worker_id, cur_op_num, goto_op_num) if(cur_op_num < WAIT_FOR_VAR(worker_id)) GOTO_STUB(worker_id, goto_op_num)
CHECK_IF_SKIP_COMPUTE = "CHECK_IF_SKIP_COMPUTE"
  

#define OUTPUT_DIRECTION_VAR(worker_id) fusion_worker_##worker_id##_output_direction

#define DECLEAR_EXTERN_OUTPUT_DIRECTION_VAR(worker_id) extern int OUTPUT_DIRECTION_VAR(worker_id)

#define OUTPUT_DIR_H 1

#define OUTPUT_DIR_V 2

#define SET_OUTPUT_DIRECTION_HORIZON(worker_id) OUTPUT_DIRECTION_VAR(worker_id) = OUTPUT_DIR_H

#define SET_OUTPUT_DIRECTION_VERTICAL(worker_id) OUTPUT_DIRECTION_VAR(worker_id) = OUTPUT_DIR_V

DECLEAR_EXTERN_OUTPUT_DIRECTION_VAR = "DECLEAR_EXTERN_OUTPUT_DIRECTION_VAR"
OUTPUT_DIRECTION_VAR = "OUTPUT_DIRECTION_VAR"

SET_OUTPUT_DIRECTION_HORIZON = "SET_OUTPUT_DIRECTION_HORIZON"
SET_OUTPUT_DIRECTION_VERTICAL = "SET_OUTPUT_DIRECTION_VERTICAL"

__reg_c_macro__ = {INSERT_STUB, GOTO_STUB, 
                   DECLEAR_EXTERN_WAIT_FOR_VAR, SET_WAIT_FOR_OP_FROM, CLEAR_WAIT_FOR_OP_FROM,
                   CHECK_IF_SKIP_COMPUTE,
                   DECLEAR_EXTERN_OUTPUT_DIRECTION_VAR, OUTPUT_DIRECTION_VAR, SET_OUTPUT_DIRECTION_HORIZON, SET_OUTPUT_DIRECTION_VERTICAL}

# auto-reg
for r in __reg_c_macro__:
    register_op_attr(f'tir.{r}', "TCallEffectKind", tvm.tir.CallEffectKind.UpdateState) # Use UpdateState mark to prevent optimized out
    register_op_attr(f'tir.{r}', "TGlobalSymbol", r)

# TODO: proxy op for no need insert stub and goto for worker with only one layer cache 

_RECORDED_WORKER_ID_OP_NUM_ = {}

def invoke_c_macro(macro_var, *args):

    global _RECORDED_WORKER_ID_OP_NUM_

    tvm_args = [tvm.runtime.const(a, dtype="int32") for a in args]
    
    # return tvm.tir.call_extern("void", macro_var.name_hint, *tvm_args)
    # return tvm.tir.call_tir(CHECK_IF_SKIP_COMPUTE, *tvm_args)
    if macro_var == INSERT_STUB:
        worker_id = args[0]
        op_num = args[1]
        if worker_id in _RECORDED_WORKER_ID_OP_NUM_:
            _RECORDED_WORKER_ID_OP_NUM_[worker_id].append(op_num)
        else:
            _RECORDED_WORKER_ID_OP_NUM_[worker_id] = [op_num]

    return tvm.tir.call_intrin("void", f'tir.{macro_var}', *tvm_args)


def get_recorded_worker_id_op_num():
    global _RECORDED_WORKER_ID_OP_NUM_

    return {k: set(v) for k,v in _RECORDED_WORKER_ID_OP_NUM_.items()}

    