
#include "dajmodel.h"
#include "dajutil.h"

namespace dajnn {

Model::Model() {
	weights = nullptr;
	weights_len = 0;
}

Model::~Model() {
	if (weights) {
		for (uint i = 0; i < weights_len; ++i) {
			delete weights[i];
		}
		delete[] weights;
	}
}

Model::Model(ByteStream* stream) : Model() {
	string header = stream->read_str();
	
	if (header.compare(MODEL_HEADER)) {
		log_w("invalid model header : %s", header);
		return;
	}
	vector<Tensor*> tensors;

	while (true) {
		string mode = stream->read_str();

		if (!mode.compare("f")) {
			tensors.push_back(new FTensor(stream));
		} else if (!mode.compare("i")) {
			tensors.push_back(new ITensor(stream));
		} else if (!mode.compare(MODEL_FOOTER)) {
			break;
		} else {
			log_w("invalid tensor mode (%s) from model", mode.c_str());
		}
	}
	weights_len = tensors.size();
	weights = new Tensor*[weights_len];

	for (uint i = 0; i < weights_len; ++i) {
		weights[i] = tensors[i];
	}
}

uint Model::length() {
	return weights_len;
}

FTensor* Model::get_f(uint idx) {
	return (FTensor*) weights[idx];
}

ITensor* Model::get_i(uint idx) {
	return (ITensor*) weights[idx];
}

}
