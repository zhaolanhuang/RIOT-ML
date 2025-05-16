#ifndef MSF_CNN_MACRO_H
#define MSF_CNN_MACRO_H

#define STUB_LABEL(id, num) STUB_##id##_##num

#define INSERT_STUB(worker_id, op_num) STUB_LABEL(worker_id, op_num):

#define GOTO_STUB(worker_id, op_num) goto STUB_LABEL(worker_id, op_num)

// #define WAIT_FOR_VAR(worker_id) fusion_worker_##worker_id##_wait_for_op

// #define DECLEAR_EXTERN_WAIT_FOR_VAR(worker_id) extern int WAIT_FOR_VAR(worker_id)
#define DECLEAR_EXTERN_WAIT_FOR_VAR(worker_id)

// #define SET_WAIT_FOR_OP_FROM(worker_id, op_num) WAIT_FOR_VAR(worker_id) = op_num

// #define CLEAR_WAIT_FOR_OP_FROM(worker_id, op_num) WAIT_FOR_VAR(worker_id) = -1

  

// #define CHECK_IF_SKIP_COMPUTE(worker_id, cur_op_num, goto_op_num) if(cur_op_num < WAIT_FOR_VAR(worker_id)) GOTO_STUB(worker_id, goto_op_num)
#define CHECK_IF_SKIP_COMPUTE(worker_id, cur_op_num, goto_op_num)  

// #define OUTPUT_DIRECTION_VAR(worker_id) fusion_worker_##worker_id##_output_direction

// #define DECLEAR_EXTERN_OUTPUT_DIRECTION_VAR(worker_id) extern int OUTPUT_DIRECTION_VAR(worker_id)
#define DECLEAR_EXTERN_OUTPUT_DIRECTION_VAR(worker_id)

// #define OUTPUT_DIR_H 1
// 
// #define OUTPUT_DIR_V 2

// #define SET_OUTPUT_DIRECTION_HORIZON(worker_id) OUTPUT_DIRECTION_VAR(worker_id) = OUTPUT_DIR_H

// #define SET_OUTPUT_DIRECTION_VERTICAL(worker_id) OUTPUT_DIRECTION_VAR(worker_id) = OUTPUT_DIR_V

#endif