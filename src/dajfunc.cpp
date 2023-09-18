
#include "dajfunc.h"
#include "dajtensor.h"
#include "dajutil.h"

#ifdef PADDLE
#include "paddle_api_2.h"
using namespace paddle::lite_api;
#endif

namespace dajnn {
namespace func {

void relu(FTensor* tensor) {
#ifdef PADDLE
	paddle_act_relu(tensor->val, tensor->val, tensor->span, PADDLE_THREADS);
#else
	for (float* v = tensor->val; v < tensor->val + tensor->span; ++v) {
		if (*v < 0) *v = 0;
	}
#endif
}

void tanh(FTensor* tensor) {
#ifdef PADDLE
	act_tanh(tensor->val, tensor->val, tensor->span, PADDLE_THREADS);
#else
	for (float* v = tensor->val; v < tensor->val + tensor->span; ++v) {
		*v = 2.f / (1.f + expf(-2.f * *v)) - 1;
	}
#endif
}

void scale(FTensor* tensor, FTensor* weight, FTensor* bias, bool is_first_batch_dim) {
	uint span = tensor->span;
	int num_batches = is_first_batch_dim ? tensor->shape[0] : 1;
	int num_channels = tensor->shape[is_first_batch_dim ? 1 : 0];
	int num_features = span / num_batches / num_channels;

	exit_if((weight->shape.size() != 1) || (weight->span != num_channels),
		"invalid scale weight shape with tensor shape : %s and %s",
		get_shape_str(&weight->shape).c_str(),
		get_shape_str(&tensor->shape).c_str());
	exit_if(bias && ((bias->shape.size() != 1) || (bias->span != num_channels)),
		"invalid scale bias shape with weight shape : %s and %s",
		get_shape_str(&bias->shape).c_str(),
		get_shape_str(&weight->shape).c_str());

#ifdef PADDLE
	paddle::lite_api::scale(tensor->val, tensor->val, num_batches, num_channels, num_features, weight->val, bias ? bias->val : nullptr);
#else
	float* v = tensor->val;
		
	if (bias) {
		for (int i = 0; i < num_batches; ++i) {
			float* w = weight->val;
			float* b = bias->val;

			for (int j = 0; j < num_channels; ++j, ++w, ++b) {
				for (int k = 0; k < num_features; ++k, ++v) {
					*v = *v * *w + *b;
				}
			}
		}
	} else {
		for (int i = 0; i < num_batches; ++i) {
			float* w = weight->val;

			for (int j = 0; j < num_channels; ++w) {
				for (int k = 0; k < num_features; ++k, ++v) {
					*v = *v * *w;
				}
			}
		}
	}
#endif
}

void add(FTensor* dst, FTensor* oprd) {
	int span = dst->span;

	exit_if(!dst->is_shape(&oprd->shape), "shapes mismatch for add operation : %s and %s",
		get_shape_str(&dst->shape).c_str(),
		get_shape_str(&oprd->shape).c_str());

#ifdef PADDLE
	paddle_elementwise_add(dst->val, oprd->val, dst->val, span);
#else
	float* v = dst->val;
	float* o = oprd->val;

	for (int i = 0; i < span; ++i, ++v, ++o) {
		*v += *o;
	}
#endif
}

}
}
