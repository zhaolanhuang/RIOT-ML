/*
 * Copyright (C) 2023 Zhaolan Huang <zhaolan.huang@fu-berlin.de>
 *
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3. See the file LICENSE in the top level
 * directory for more details.
 */

/**
 * @ingroup     apps
 * @{
 *
 * @file
 * @brief       RIOT-ML Standalone Minimal Application
 *
 * @author      Zhaolan Huang <zhaolan.huang@fu-berlin.de>
 *
 * @}
 */

#include <stdio.h>
#include <string.h>
#include "random.h"
#include "stdio_base.h"
#include "mlmci.h"

#include <tvmgen_default.h>
extern mlmodel_t *model_ptr;

int main(void)
{


    mlmodel_init(model_ptr);
    mlmodel_set_global_model(model_ptr);

    random_init(42);

    for(int j = mlmodel_get_num_input_vars(model_ptr); j > 0; j--) {
            mlmodel_iovar_t *input = mlmodel_get_input_variable(model_ptr, j - 1);
            random_bytes(input->values, input->num_bytes);
    }
        
    int ret_val = mlmodel_inference(model_ptr);
    (void) ret_val;

    return 0;
}
