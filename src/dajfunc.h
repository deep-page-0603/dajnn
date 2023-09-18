
#pragma once

#include "dajnn.h"

namespace dajnn {
namespace func {

void relu(FTensor* tensor);
void tanh(FTensor* tensor);

void scale(FTensor* tensor, FTensor* weight, FTensor* bias, bool is_first_batch_dim);
void add(FTensor* dst, FTensor* oprd);

}
}
