
#pragma once

#include "dajnn.h"

namespace dajnn {
namespace norm {

void batch_norm_with_precomputed(FTensor* tensor, FTensor* pc_gamma, FTensor* pc_beta);

}
}
