
#pragma once

#include "dajnn.h"

namespace dajnn {
namespace dense {

/*
	full-connected layer
	@param input: 2-d tensor with shape (n, m)
	@param kernel: 2-d tensor with shape (m, p)
	@param bias: null or 1-d tensor with shape (p)
	@return: 2-d tensor with shape (n, p)
*/
FTensor* dense(FTensor* input, FTensor* kernel, FTensor* bias);

}
}
