/*
 * Copyright (c) 2023 Zhaolan Huang
 * 
 * This file is subject to the terms and conditions of the GNU Lesser
 * General Public License v3.0. See the file LICENSE in the top level
 * directory for more details.
 * 
 * Adapted from sys/suit/storage/ram.c
 * 
 */


#include <string.h>
#include <inttypes.h>

#include "fmt.h"
#include "kernel_defines.h"
#include "log.h"
#include "xfa.h"

#include "suit.h"
#include "suit/storage.h"
#include "include/storage_params.h"

XFA_USE(suit_storage_t, suit_storage_reg);

static inline suit_storage_model_params_t *_get_ram(suit_storage_t *storage)
{
    return container_of(storage, suit_storage_model_params_t, storage);
}

static inline const suit_storage_model_params_t *_get_ram_const(
    const suit_storage_t *storage)
{
    return container_of(storage, suit_storage_model_params_t, storage);
}

static inline mlmodel_param_t *_get_active_params(
    suit_storage_model_params_t *ram)
{
    return ram->active_params;
}

static bool _get_params_by_string(const suit_storage_model_params_t *ram, const char *location, mlmodel_param_t **val)
{
    /* Matching on .ram.### */
    static const char prefix[] = CONFIG_SUIT_STORAGE_MODEL_PARAMS_LOCATION_PREFIX;
    static const size_t prefix_len = sizeof(prefix) - 1;

    /* Check for prefix */
    if (strncmp(prefix, location, prefix_len) == 0 &&
        location[prefix_len] != '\n') {
        /* Advance to the number */
        location += prefix_len;
        /* Check if the rest of the string is a number */
        if (fmt_is_number(location)) {
            /* grab the number */
            uint32_t index = scn_u32_dec(location, 5);
            *val = mlmodel_get_parameter(ram->current_model, index);
            /* Number must be smaller than the number of regions */
            if (*val != NULL) {
                return true;
            }
        } 
        /* Find Params by name */
        else {
            *val = mlmodel_get_parameter_by_name(ram->current_model, location);
            if (*val != NULL) {
                return true;
            }
        }
    }

    return false;
}

/* will be called before main func*/
static int _ram_init(suit_storage_t *storage)
{

    suit_storage_model_params_t *ram = _get_ram(storage);
    ram->sequence_no = 0;
    ram->current_model = mlmodel_get_global_model();
    ram->active_params = NULL;
    
    return SUIT_OK;
}

static int _ram_start(suit_storage_t *storage, const suit_manifest_t *manifest,
                      size_t len)
{
    (void)manifest;
    (void)storage;
    (void)len;
    // suit_storage_model_params_t *ram = _get_ram(storage);
    
    return SUIT_OK;
}

static int _ram_write(suit_storage_t *storage, const suit_manifest_t *manifest,
                      const uint8_t *buf, size_t offset, size_t len)
{
    (void)manifest;
    suit_storage_model_params_t *ram = _get_ram(storage);
    mlmodel_param_update_values(ram->active_params, len, offset, buf);
    

    return SUIT_OK;
}

static int _ram_finish(suit_storage_t *storage, const suit_manifest_t *manifest)
{
    (void)storage;
    (void)manifest;
    return SUIT_OK;
}

static int _ram_install(suit_storage_t *storage, const suit_manifest_t *manifest)
{
    suit_storage_set_seq_no(storage, manifest->seq_number);
    return SUIT_OK;
}

static int _ram_erase(suit_storage_t *storage)
{
    (void)storage;
    return SUIT_ERR_NOT_SUPPORTED;
}

static int _ram_read(suit_storage_t *storage, uint8_t *buf, size_t offset,
                     size_t len)
{
    suit_storage_model_params_t *ram = _get_ram(storage);
    mlmodel_param_t *region = _get_active_params(ram);

    if (offset + len > region->num_bytes) {
        return SUIT_ERR_STORAGE_EXCEEDED;
    }

    memcpy(buf, &region->values[offset], len);

    return SUIT_OK;
}

static int _ram_read_ptr(suit_storage_t *storage,
                         const uint8_t **buf, size_t *len)
{
    suit_storage_model_params_t *ram = _get_ram(storage);
    mlmodel_param_t *region = _get_active_params(ram);

    *buf = region->values;
    *len = region->num_bytes;
    return SUIT_OK;
}

static bool _ram_has_location(const suit_storage_t *storage,
                              const char *location)
{
    mlmodel_param_t *val = NULL;
    suit_storage_model_params_t *ram = _get_ram(storage);
    ram->current_model = mlmodel_get_global_model();

    return _get_params_by_string(ram, location, &val);
}

static int _ram_set_active_location(suit_storage_t *storage,
                                    const char *location)
{
    suit_storage_model_params_t *ram = _get_ram(storage);
    mlmodel_param_t *region = NULL;

    if (!_get_params_by_string(ram, location, &region)) {
        return -1;
    }

    ram->active_params = region;
    return SUIT_OK;
}

static int _ram_get_seq_no(const suit_storage_t *storage, uint32_t *seq_no)
{
    const suit_storage_model_params_t *ram = _get_ram_const(storage);

    *seq_no = ram->sequence_no;
    LOG_INFO("Retrieved sequence number: %" PRIu32 "\n", *seq_no);
    return SUIT_OK;
}

static int _ram_set_seq_no(suit_storage_t *storage, uint32_t seq_no)
{
    suit_storage_model_params_t *ram = _get_ram(storage);

    if (ram->sequence_no < seq_no) {
        LOG_INFO("Stored sequence number: %" PRIu32 "\n", seq_no);
        ram->sequence_no = seq_no;
        return SUIT_OK;
    }

    return SUIT_ERR_SEQUENCE_NUMBER;
}

static const suit_storage_driver_t suit_storage_model_params_driver = {
    .init = _ram_init,
    .start = _ram_start,
    .write = _ram_write,
    .finish = _ram_finish,
    .read = _ram_read,
    .read_ptr = _ram_read_ptr,
    .install = _ram_install,
    .erase = _ram_erase,
    .has_location = _ram_has_location,
    .set_active_location = _ram_set_active_location,
    .get_seq_no = _ram_get_seq_no,
    .set_seq_no = _ram_set_seq_no,
    .separator = CONFIG_SUIT_STORAGE_MODEL_PARAMS_LOCATION_SEPARATOR,
};

suit_storage_model_params_t suit_storage_model_params = {
    .storage = {
        .driver = &suit_storage_model_params_driver,
    },
};

XFA(suit_storage_reg, 0) suit_storage_t* suit_storage_model_params_ptr = &suit_storage_model_params.storage;