
#include "cmsis_nn_wrapper.h"
#include <arm_nnfunctions.h>
#include <arm_nn_types.h>
#include <arm_nn_math_types.h>

#define CTX_BUF_SIZE 5120
static char* cmsis_nn_ctx_buf[CTX_BUF_SIZE];

int32_t arm_convolve_wrapper_s8_(int8_t* input_data, int8_t* filter_data, int8_t *output_data, 
                            int32_t input_n, int32_t input_h, int32_t input_w, int32_t input_c,
                            int32_t filter_n, int32_t filter_h, int32_t filter_w, int32_t filter_c,
                            int32_t output_n, int32_t output_h, int32_t output_w, int32_t output_c,
                            int32_t stride, int32_t padding, int32_t dilation// parts of cmsis_nn_conv_params
                        ) {
                            cmsis_nn_context ctx = {cmsis_nn_ctx_buf, CTX_BUF_SIZE};
                            cmsis_nn_conv_params conv_params = {
                                .input_offset = 0, .output_offset = 0,
                                .stride = stride, .dilation = dilation, padding = padding,
                                .activation = {.max = 127, .min = -128}
                            };
                            cmsis_nn_per_channel_quant_params quant_params = {.multiplier=1, .shift = 0};
                            
                            cmsis_nn_dims input_dims = {input_n, input_h, input_w, input_c};
                            cmsis_nn_dims filter_dims = {filter_n, filter_h, filter_w, filter_c};
                            cmsis_nn_dims output_dims = {output_n, output_h, output_w, output_c};

                            cmsis_nn_dims bias_dims = {1, 1, 1, output_c};

                            arm_status status = arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_params, 
                                                                        &input_dims, input_data,
                                                                        &filter_dims, filter_data,
                                                                        &bias_dims, NULL,
                                                                        &output_dims, output_data);
                            if (status == ARM_MATH_SUCCESS) {
                                return 0;
                            } else {
                                return -1;
                            }                                                       

}

int32_t arm_depthwise_conv_wrapper_s8_(int8_t* input_data, int8_t* filter_data, int8_t *output_data, 
                            int32_t input_n, int32_t input_h, int32_t input_w, int32_t input_c,
                            int32_t filter_n, int32_t filter_h, int32_t filter_w, int32_t filter_c,
                            int32_t output_n, int32_t output_h, int32_t output_w, int32_t output_c,
                            int32_t stride, int32_t padding, int32_t dilation, int32_t ch_mult // parts of cmsis_nn_conv_params
                        ) {
                            cmsis_nn_context ctx = {cmsis_nn_ctx_buf, CTX_BUF_SIZE};
                            cmsis_nn_dw_conv_params dw_conv_params = {
                                .input_offset = 0, .output_offset = 0,
                                .stride = stride, .dilation = dilation, padding = padding,
                                .activation = {.max = 127, .min = -128},
                                .ch_mult = ch_mult
                            };
                            cmsis_nn_per_channel_quant_params quant_params = {.multiplier=1, .shift = 0};
                            
                            cmsis_nn_dims input_dims = {input_n, input_h, input_w, input_c};
                            cmsis_nn_dims filter_dims = {filter_n, filter_h, filter_w, filter_c};
                            cmsis_nn_dims output_dims = {output_n, output_h, output_w, output_c};

                            cmsis_nn_dims bias_dims = {1, 1, 1, output_c};

                            arm_status status = arm_depthwise_conv_wrapper_s8(&ctx, &dw_conv_params, &quant_params, 
                                                                        &input_dims, input_data,
                                                                        &filter_dims, filter_data,
                                                                        &bias_dims, NULL,
                                                                        &output_dims, output_data);
                            if (status == ARM_MATH_SUCCESS) {
                                return 0;
                            } else {
                                return -1;
                            }

}
    

