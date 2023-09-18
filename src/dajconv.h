
#pragma once

#include "dajnn.h"

namespace dajnn {
namespace conv {

/*
	2-d convolutional layer
	@param input: 4-d tensor with shape (n, c, h, w)
	@param kernel: 4-d tensor with shape (f, c, k_h, k_w)
	@param bias: null or 1-d tensor with shape (f)
	@param padding_x: padding sizes (-1 for auto, 0 for no padding)
	@param stride_x: strides
	@param dilation_x: dilations
	@return: 4-d tensor with shape (n, f, _h_, _w_)

	CAUTION:
		f (# of filters) must be >1 for mobile forward (paddle-lite's bug)
		dilation_x must be =1 for win32 forward (darknet's limitance)
*/
FTensor* conv2d(FTensor* input, FTensor* kernel, FTensor* bias = nullptr,
	int padding_h = -1, int padding_w = -1, int stride_h = 1, int stride_w = 1,
	int dilation_h = 1, int dilation_w = 1);

}
}
