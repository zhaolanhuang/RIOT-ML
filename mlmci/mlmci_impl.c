/*
 * Copyright (c) 2023 Zhaolan Huang
 * 
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3.0. See the file LICENSE in the top level
 * directory for more details.
 */

#include "mlmci.h"
#include <stddef.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


int mlmodel_start_inference(mlmodel_t *model) {
    model->status = MLMODEL_STATUS_INFERENCE;
    return 0;
}

int mlmodel_start_training(mlmodel_t *model) {
    model->status = MLMODEL_STATUS_TRAINING;
    return 0;
}

int mlmodel_stop(mlmodel_t *model, mlmodel_complete_callback_t cb) {
    model->status = MLMODEL_STATUS_STOP;
    model->complete_cb = cb;
    return 0;    
}

mlmodel_status_t mlmodel_get_status(const mlmodel_t *model) {
    return model->status;
}
size_t mlmodel_get_num_opts(const mlmodel_t *model) {
    return model->num_operators;
}
mlmodel_operator_t *mlmodel_get_operator(const mlmodel_t *model, size_t location) {
    if (location >= model->num_operators) return NULL;
    return &(model->operators[location]);

}
size_t mlmodel_get_num_params(const mlmodel_t *model) {
    return model->num_params;
}
mlmodel_param_t *mlmodel_get_parameter(const mlmodel_t *model, size_t location) {
    if (location >= model->num_params) return NULL;
    return &(model->params[location]);
}

mlmodel_param_t *mlmodel_get_parameter_by_name(const mlmodel_t *model, const char name[]) {
    for(size_t i = 0; i < model->num_params; i++) {
        if(0 == strcmp(model->params[i].name, name)) return &(model->params[i]);
    }
    return NULL;
}

size_t mlmodel_get_num_input_vars(const mlmodel_t *model) {
    return model->num_input_vars;
}
mlmodel_iovar_t *mlmodel_get_input_variable(const mlmodel_t *model, size_t location) {
    if (location >= model->num_input_vars) return NULL;
    return &(model->input_vars[location]);
}

size_t mlmodel_get_num_output_vars(const mlmodel_t *model) {
    return model->num_output_vars;
}
mlmodel_iovar_t *mlmodel_get_output_variable(const mlmodel_t *model, size_t location){
    if (location >= model->num_output_vars) return NULL;
    return &(model->output_vars[location]);
}


/* Operator Interface */
size_t mlmodel_opt_get_num_params(const mlmodel_operator_t *operator) {
    return operator->num_params;
}
mlmodel_param_t *mlmodel_opt_get_param(const mlmodel_operator_t *operator, size_t location) {
    if (location >= operator->num_params) return NULL;
    return &(operator->params[location]);
}

/* Parameter Interface */
size_t mlmodel_param_get_values_bytes(const mlmodel_param_t *parameter) {
    return parameter->num_bytes;
}
mlmodel_param_permission_t mlmodel_param_get_permission(const mlmodel_param_t *parameter) {
    return parameter->permission;
}

#ifdef CONFIG_PARAMS_PERSISTENT_ON_FLASH

#include "mtd.h"
#include "mtd_flashpage.h"
#define PAGES_PER_SECTOR    8
static mtd_flashpage_t _dev = MTD_FLASHPAGE_INIT_VAL(PAGES_PER_SECTOR);
static mtd_dev_t *dev = &_dev.base;

#endif

int mlmodel_param_update_values(mlmodel_param_t *parameter, size_t num_values, size_t offset, const uint8_t *values) {
    if (num_values + offset > parameter->num_bytes) return -1;

#ifdef CONFIG_PARAMS_PERSISTENT_ON_FLASH
    static uint8_t is_mtd_init = 0;
    int res;
    if(0 == is_mtd_init) {
        //TODO: exception handle
        res = mtd_init(dev);
        printf("mtd_init res=%d\n", res);

        is_mtd_init = 1;
    }
    printf("mtd_write dst=%ld\n", (uint32_t)dst);
    res = mtd_write(dev, values, (uint32_t)dst, num_values);
    printf("mtd_write res=%d\n", res);


#else
    
    
    if (parameter->volatile_values == NULL) {
        parameter->volatile_values = malloc(parameter->num_bytes);
        memcpy(parameter->volatile_values, parameter->persistent_values, parameter->num_bytes);
    }
    
    uint8_t *dst = parameter->volatile_values + offset;

    memcpy(dst, values, num_values);
    parameter->values = parameter->volatile_values;

#endif
    return 0;
}

/* IO Variable Interface */
size_t mlmodel_iovar_get_values_bytes(const mlmodel_iovar_t *var){
    return var->num_bytes;
}
uint8_t* mlmodel_iovar_get_values(const mlmodel_iovar_t *var){
    return var->values;
}
char* mlmodel_iovar_get_name(const mlmodel_iovar_t *var) {
    return var->name;
}


