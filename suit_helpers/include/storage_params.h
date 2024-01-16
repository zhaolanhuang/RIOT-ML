/*
 * Copyright (c) 2023 Zhaolan Huang
 * 
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3.0. See the file LICENSE in the top level
 * directory for more details.
 * 
 * Adapted from sys/include/suit/storage/ram.h
 * 
 */


#ifndef STORAGE_PARAMS_H
#define STORAGE_PARAMS_H

#include <stdint.h>

#include "suit.h"
#include "mlmci.h"

#ifdef __cplusplus
extern "C" {
#endif



/**
 * @brief Extra attributes for allocating the RAM struct
 */
#ifndef CONFIG_SUIT_STORAGE_MODEL_PARAMS_ATTR
#define CONFIG_SUIT_STORAGE_MODEL_PARAMS_ATTR
#endif

/**
 * @brief Storage location string separators
 */
#ifndef CONFIG_SUIT_STORAGE_MODEL_PARAMS_LOCATION_SEPARATOR
#define CONFIG_SUIT_STORAGE_MODEL_PARAMS_LOCATION_SEPARATOR '.'
#endif

/**
 * @brief Storage location string prefix
 *
 * Must include the leading and trailing separators
 */
#ifndef CONFIG_SUIT_STORAGE_MODEL_PARAMS_LOCATION_PREFIX
#define CONFIG_SUIT_STORAGE_MODEL_PARAMS_LOCATION_PREFIX  ".model."
#endif


/**
 * @brief memory storage state
 */
typedef struct CONFIG_SUIT_STORAGE_MODEL_PARAMS_ATTR {
    suit_storage_t storage;       /**< parent struct */
    /**
     * @brief ram storage regions
     */
    uint32_t sequence_no; /**< Ephemeral sequence number */
    mlmodel_t *current_model;
    mlmodel_param_t *active_params;
} suit_storage_model_params_t;

#ifdef __cplusplus
}
#endif

#endif /* SUIT_STORAGE_MODEL_PARAMS_H */
/** @} */
