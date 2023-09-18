
#include "dajnorm.h"
#include "dajtensor.h"
#include "dajfunc.h"

#ifdef PADDLE
#include "paddle_api_2.h"
using namespace paddle::lite_api;
#endif

namespace dajnn {
namespace norm {

void batch_norm_with_precomputed(FTensor* tensor, FTensor* pc_gamma, FTensor* pc_beta) {
	func::scale(tensor, pc_gamma, pc_beta, true);	
}

}
}
