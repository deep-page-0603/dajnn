
#pragma once

#include "dajtensor.h"

namespace dajnn {

class Model {
public:
	Model();
	Model(ByteStream* stream);
	virtual ~Model();

public:
	uint length();

	FTensor* get_f(uint idx);
	ITensor* get_i(uint idx);

protected:
	Tensor** weights;
	uint weights_len;
};

}
