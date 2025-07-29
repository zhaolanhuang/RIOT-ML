#ifndef CMSIS_NN_WRAPPER
#define CMSIS_NN_WRAPPER

// #include <arm_nnfunctions.h>
// #include <arm_nn_types.h>
// #include <arm_nn_math_types.h>
#include <stdint.h>

extern int32_t arm_convolve_wrapper_s8_(int8_t* input_data, int8_t* filter_data, int8_t *output_data, 
                            int32_t input_n, int32_t input_h, int32_t input_w, int32_t input_c,
                            int32_t filter_o, int32_t filter_h, int32_t filter_w, int32_t filter_i,
                            int32_t output_n, int32_t output_h, int32_t output_w, int32_t output_c,
                            int32_t stride, int32_t padding, int32_t dilation// parts of cmsis_nn_conv_params
                        );

extern int32_t arm_depthwise_conv_wrapper_s8_(int8_t* input_data, int8_t* filter_data, int8_t *output_data, 
                            int32_t input_n, int32_t input_h, int32_t input_w, int32_t input_c,
                            int32_t filter_o, int32_t filter_h, int32_t filter_w, int32_t filter_i,
                            int32_t output_n, int32_t output_h, int32_t output_w, int32_t output_c,
                            int32_t stride, int32_t padding, int32_t dilation, int32_t ch_mult // parts of cmsis_nn_conv_params
                        );
    
// arm_convolve_wrapper_s8	(	const cmsis_nn_context * 	ctx,
// const cmsis_nn_conv_params * 	conv_params,
// const cmsis_nn_per_channel_quant_params * 	quant_params,
// const cmsis_nn_dims * 	input_dims,
// const int8_t * 	input_data,
// const cmsis_nn_dims * 	filter_dims,
// const int8_t * 	filter_data,
// const cmsis_nn_dims * 	bias_dims,
// const int32_t * 	bias_data,
// const cmsis_nn_dims * 	output_dims,
// int8_t * 	output_data 
// )		


#endif