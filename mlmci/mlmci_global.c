/*
 * Copyright (c) 2023 Zhaolan Huang
 * 
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3.0. See the file LICENSE in the top level
 * directory for more details.
 */

#include "mlmci.h"

static mlmodel_t *global_model_ptr = NULL;

/* Global Model Interface */
void mlmodel_set_global_model(mlmodel_t *model) {
    global_model_ptr = model;
}

mlmodel_t *mlmodel_get_global_model(void) {
    return global_model_ptr;
}

const mlmodel_t *mlmodel_get_global_model_const(void) {
    return global_model_ptr;
}