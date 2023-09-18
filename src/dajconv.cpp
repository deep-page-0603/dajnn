
#include "dajdense.h"
#include "dajtensor.h"
#include "dajutil.h"
#include "dajgemm.h"

#ifdef PADDLE
#include "paddle_api_2.h"
using namespace paddle::lite_api;
#endif

namespace dajnn {
namespace conv {

#ifndef PADDLE
float im2col_get_pixel(float* im, int height, int width, int channels,
		int row, int col, int channel, int pad_h, int pad_w) {

    row -= pad_h;
    col -= pad_w;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

void im2col_cpu(float* data_im, int channels, int height, int width,
		int kernel_h, int kernel_w, int stride_h, int stride_w,
		int pad_h, int pad_w, float* data_col) {

    int c, h, w;
    int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

	int ksize = kernel_h * kernel_w;
    int channels_col = channels * ksize;

    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % kernel_w;
        int h_offset = (c / kernel_w) % kernel_h;
        int c_im = c / ksize;

        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride_h;
                int im_col = w_offset + w * stride_w;
                int col_index = (c * height_col + h) * width_col + w;

                data_col[col_index] = im2col_get_pixel(
					data_im, height, width, channels,
                    im_row, im_col, c_im, pad_h, pad_w);
            }
        }
    }
}
#endif

FTensor* conv2d(FTensor* input, FTensor* kernel, FTensor* bias,
		int padding_h, int padding_w, int stride_h, int stride_w,
		int dilation_h, int dilation_w) {
	
	exit_if(input->shape.size() != 4, "input dim of conv2d expects to be 4, but got %d", input->shape.size());
	exit_if(kernel->shape.size() != 4, "kernel dim of conv2d expects to be 4, but got %d", kernel->shape.size());
	exit_if(bias && (bias->shape.size() != 1), "bias dim of conv2d expects to be null or 1, but got %d", bias->shape.size());

	int num_batches = input->shape[0];
	int num_channels = input->shape[1];
	int h = input->shape[2];
	int w = input->shape[3];
	int num_filters = kernel->shape[0];
	int kernel_h = kernel->shape[2];
	int kernel_w = kernel->shape[3];

	exit_if(kernel->shape[1] != num_channels, "second dim of conv2d kernel (# of channels) expects to be %d, but got %d", num_channels, kernel->shape[1]);
	exit_if(bias->span != num_filters, "span of conv2d bias (# of filters) expects to be %d, but got %d", num_filters, bias->span);

	if (padding_h < 0) padding_h = (kernel_h - 1) * dilation_h / 2;
	if (padding_w < 0) padding_w = (kernel_w - 1) * dilation_w / 2;

	int _h_ = (h + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
	int _w_ = (w + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

	FTensor* output = new FTensor(num_batches, num_filters, _h_, _w_, END_DIM);
		
	int chw = input->span / num_batches;
	int _chw_ = output->span / num_batches;

	float* ip = input->val;
	float* op = output->val;

#ifdef PADDLE
	exit_if(num_filters == 1, "single conv2d filter is not available for mobile forward");

	paddle_conv2d(num_batches, h, w, num_channels, ip, num_filters, kernel_h, kernel_w, kernel->val,
		op, bias ? bias->val : nullptr, padding_h, padding_w, dilation_h, dilation_w,
		stride_h, stride_w, 0, 0, PADDLE_CLS, PADDLE_THREADS);
#else
	exit_if((dilation_h != 1) || (dilation_w != 1), "only single dilation is available for win32 forward");
	memset(output->val, 0, 4 * output->span);
	
	int _hw_ = _h_ * _w_;
	int _kkc_ = kernel_h * kernel_w * num_channels;
	float* workspace = (float*) malloc(4 * _kkc_ * _hw_);

	for (int i = 0; i < num_batches; ++i, ip += chw, op += _chw_) {
        im2col_cpu(ip, num_channels, h, w, kernel_h, kernel_w, stride_h, stride_w,
			padding_h, padding_w, workspace);
        gemm(0, 0, num_filters, _hw_, _kkc_, 1, kernel->val, _kkc_, workspace, _hw_, 1, op, _hw_);
    }
	if (bias) {
		op = output->val;

		for (int i = 0; i < num_batches; ++i) {
			for (int j = 0; j < num_filters; ++j) {
				float b = bias->val[j];

				for (int k = 0; k < _hw_; ++k) {
					op[i * _chw_ + j * _hw_ + k] += b;
				}
			}
		}
	}
	free(workspace);
#endif
	return output;
}

}
}
