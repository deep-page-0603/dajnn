
#include "dajdense.h"
#include "dajtensor.h"
#include "dajutil.h"
#include "dajgemm.h"

#ifdef PADDLE
#include "paddle_api_2.h"
using namespace paddle::lite_api;
#endif

namespace dajnn {
namespace dense {

FTensor* dense(FTensor* input, FTensor* kernel, FTensor* bias) {
	exit_if(input->shape.size() != 2, "input dim of dense expects to be 2, but got %d", input->shape.size());
	exit_if(kernel->shape.size() != 2, "kernel dim of dense expects to be 2, but got %d", kernel->shape.size());
	exit_if(bias && (bias->shape.size() != 1), "bias dim of dense expects to be null or 1, but got %d", bias->shape.size());

	uint n = input->shape[0];
	uint m = input->shape[1];
#ifdef PADDLE
	uint p = kernel->shape[1];
	bool shape_ok = kernel->shape[0] == m;
#else
	uint p = kernel->shape[0];
	bool shape_ok = kernel->shape[1] == m;
#endif

	exit_if(!shape_ok, "dense input and kernel shapes mismatch : %s and %s",
		get_shape_str(&input->shape).c_str(),
		get_shape_str(&kernel->shape).c_str());
	exit_if(bias && (bias->span != p), "dense kernel and bias shapes mismatch : %s and %s",
		get_shape_str(&kernel->shape).c_str(),
		get_shape_str(&bias->shape).c_str());

	FTensor* output = new FTensor(n, p, END_DIM);	
#ifdef PADDLE
	paddle_matmul(n, p, m, input->val, kernel->val, output->val);
	
	if (bias) {
		float* op = output->val;

		for (uint i = 0; i < n; ++i, op += p) {
			paddle_elementwise_add(op, bias->val, op, p);
		}
	}
#else
	memset(output->val, 0, 4 * output->span);
	gemm(0, 1, n, p, m, 1, input->val, m, kernel->val, m, 1, output->val, p);

	if (bias) {
		float* op = output->val;

		for (uint i = 0; i < n; ++i) {
			for (float* bp = bias->val; bp < bias->val + p; ++bp, ++op) {
				*op += *bp;
			}
		}
	}
#endif
	return output;
}

}
}