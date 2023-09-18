
#include "dajnn.h"
#include "dajutil.h"

#ifdef PADDLE
#include "paddle_api_2.h"
using namespace paddle::lite_api;
#endif

namespace dajnn {

#ifdef TRACE_MEMORY_LEAK
vector<Tensor*> _tensor_trace_pool_;
vector<uint> _tensor_unique_indice_;
uint _tensor_trace_len_ = 0;

void push_tensor_trace(Tensor* tensor) {
	if (_tensor_trace_pool_.empty()) _tensor_trace_len_ = 0;

	_tensor_trace_pool_.push_back(tensor);
	_tensor_unique_indice_.push_back(_tensor_trace_len_++);
}

uint pop_tensor_trace(Tensor* tensor) {
	uint i = 0;

	for (vector<Tensor*>::iterator ti = _tensor_trace_pool_.begin();
		ti != _tensor_trace_pool_.end(); ++ti, ++i) {
		if (tensor == *ti) break;
	}
	exit_if(i == _tensor_trace_pool_.size(), "cannot find tensor to pop from trace");

	uint idx = _tensor_unique_indice_[i];
	_tensor_unique_indice_.erase(_tensor_unique_indice_.begin() + i);
	_tensor_trace_pool_.erase(_tensor_trace_pool_.begin() + i);
	return idx;
}

vector<uint> get_leaked_tensor_indice() {
	return _tensor_unique_indice_;
}
#endif

void init_dajnn() {
#ifdef PADDLE
	paddle_DeviceInit();
#endif
}

void finish_dajnn() {
#ifdef TRACE_MEMORY_LEAK
	if (!_tensor_unique_indice_.empty()) {
		string msg = "leaked tensors : ";

		for (vector<uint>::iterator idx = _tensor_unique_indice_.begin(); idx != _tensor_unique_indice_.end(); ++idx) {
			msg += format_str("%d ", *idx);
		}
		log_e("%s", msg);
	}
#endif
}

}
