/*
 * Copyright (c) 2023 Koen Zandberg, Zhaolan Huang
 * 
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3.0. See the file LICENSE in the top level
 * directory for more details.
 */

#ifndef MLMCI_H
#define MLMCI_H
#include <stdint.h>
#include <stddef.h>

/* model control interface */
typedef size_t mlmodel_location_t; /* Easy to ajust later when needed */
typedef struct mlmodel_t mlmodel_t;
typedef void (*mlmodel_complete_callback_t)(mlmodel_t *model, void *ctx);

/* Deprecated */
// typedef enum {
//     MLMODEL_PARAM_CLASS_WEIGHTS  = 0x01,
//     MLMODEL_PARAM_CLASS_BIAS     = 0x02,
//     MLMODEL_PARAM_CLASS_RMEAN     = 0x04,
//     MLMODEL_PARAM_CLASS_RVARIANCE = 0x08,
// } mlmodel_param_class_t;

typedef enum {
    MLMODEL_PARAM_PERMISSION_NO = 0x00,
    MLMODEL_PARAM_PERMISSION_READ = 0x01,
    MLMODEL_PARAM_PERMISSION_WRITE = 0x02,
    MLMODEL_PARAM_PERMISSION_RW = 0x03,

} mlmodel_param_permission_t;

typedef enum {
    MLMODEL_STATUS_STOP = 0x00,
    MLMODEL_STATUS_INFERENCE = 0x01,
    MLMODEL_STATUS_TRAINING = 0x02,
    MLMODEL_STATUS_ERROR = 0x03,

} mlmodel_status_t;

typedef struct{
    uint8_t* values;
    size_t num_bytes;
    mlmodel_param_permission_t permission;
    char *name;
    uint8_t* volatile_values;
    uint8_t* persistent_values;
} mlmodel_param_t;

typedef struct {
    mlmodel_param_t *params;
    size_t num_params;
    char *name;
} mlmodel_operator_t;

typedef struct {
    uint8_t *values;
    size_t num_bytes;
    char *name;
} mlmodel_iovar_t;

typedef struct mlmodel_model_driver {
    int (*init)(mlmodel_t *model);
    int (*inference)(mlmodel_t *model);

    int (*backward_pass)(mlmodel_t *model, mlmodel_iovar_t input_gradients[], mlmodel_iovar_t output_gradients[]);
    
    
} mlmodel_model_driver_t;

struct mlmodel_t{
    mlmodel_param_t *params;
    size_t num_params;
    mlmodel_operator_t *operators;
    size_t num_operators;
    mlmodel_status_t status;
    mlmodel_complete_callback_t complete_cb;

    mlmodel_iovar_t *input_vars;
    size_t num_input_vars;

    mlmodel_iovar_t *output_vars;
    size_t num_output_vars;

    mlmodel_model_driver_t *driver;
    char *name;

};

/* Model Interface */
int mlmodel_start_inference(mlmodel_t *model);
int mlmodel_start_training(mlmodel_t *model);
int mlmodel_stop(mlmodel_t *model, mlmodel_complete_callback_t cb);
mlmodel_status_t mlmodel_get_status(const mlmodel_t *model);

static inline const char* mlmodel_get_name(const mlmodel_t *model) {
    return model->name;
}

size_t mlmodel_get_num_opts(const mlmodel_t *model);
mlmodel_operator_t *mlmodel_get_operator(const mlmodel_t *model, size_t location);

size_t mlmodel_get_num_params(const mlmodel_t *model);
mlmodel_param_t *mlmodel_get_parameter(const mlmodel_t *model, size_t location);
mlmodel_param_t *mlmodel_get_parameter_by_name(const mlmodel_t *model, const char name[]);

size_t mlmodel_get_num_input_vars(const mlmodel_t *model);
mlmodel_iovar_t *mlmodel_get_input_variable(const mlmodel_t *model, size_t location);

size_t mlmodel_get_num_output_vars(const mlmodel_t *model);
mlmodel_iovar_t *mlmodel_get_output_variable(const mlmodel_t *model, size_t location);

/* Operator Interface */
size_t mlmodel_opt_get_num_params(const mlmodel_operator_t *operator);
mlmodel_param_t *mlmodel_opt_get_param(const mlmodel_operator_t *operator, size_t location);

/* Parameter Interface */
size_t mlmodel_param_get_values_bytes(const mlmodel_param_t *parameter);
mlmodel_param_permission_t mlmodel_param_get_permission(const mlmodel_param_t *parameter);
int mlmodel_param_update_values(mlmodel_param_t *parameter, size_t num_values, size_t offset, const uint8_t *values);

/* IO Variable Interface */
size_t mlmodel_iovar_get_values_bytes(const mlmodel_iovar_t *var);
uint8_t* mlmodel_iovar_get_values(const mlmodel_iovar_t *var);
char* mlmodel_iovar_get_name(const mlmodel_iovar_t *var);

/* General Impl of Driver */
static inline int mlmodel_init(mlmodel_t *model) {
    return model->driver->init(model);
}

static inline int mlmodel_inference(mlmodel_t *model) {
    return model->driver->inference(model);
}

static inline int mlmodel_backward_pass(mlmodel_t *model, mlmodel_iovar_t input_gradients[], mlmodel_iovar_t output_gradients[]) {
    return model->driver->backward_pass(model, input_gradients, output_gradients);
}

/* Global Model Interface */
void mlmodel_set_global_model(mlmodel_t *model);
mlmodel_t *mlmodel_get_global_model(void);
const mlmodel_t *mlmodel_get_global_model_const(void);




#endif