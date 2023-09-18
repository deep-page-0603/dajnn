
#include "dajtensor.h"
#include "dajutil.h"

namespace dajnn {

Tensor::Tensor() {
	_val_ = nullptr;
	span = 0;
	releasable = true;

#ifdef TRACE_MEMORY_LEAK
	push_tensor_trace(this);
#endif
}

Tensor::~Tensor() {
	if (releasable && _val_) free(_val_);

#ifdef TRACE_MEMORY_LEAK
	pop_tensor_trace(this);
#endif
}

void Tensor::reshape(vector<uint>* shape) {
	exit_if(span != get_span(shape), "unable to reshape from %s to %s",
		get_shape_str(&this->shape).c_str(), get_shape_str(shape).c_str());
	this->shape = *shape;
}

void Tensor::reshape(uint dim1, ...) {
	if (dim1 == END_DIM) return;

	vector<uint> running_shape;
	running_shape.push_back(dim1);

	va_list ap;
	va_start(ap, dim1);
	uint adim = va_arg(ap, uint);

	while (adim != END_DIM) {
		running_shape.push_back(adim);
		adim = va_arg(ap, uint);
	}
	va_end(ap);
	reshape(&running_shape);
}

bool Tensor::is_shape(vector<uint>* shape) {
	if (this->shape.size() != shape->size()) return false;

	for (uint i = 0; i < shape->size(); ++i) {
		if (this->shape[i] != shape->at(i)) return false;
	}
	return true;
}

bool Tensor::is_shape(uint dim1, ...) {
	if (dim1 != END_DIM) {
		if (shape.empty()) return false;
		if (shape[0] != dim1) return false;
	} else if (!shape.empty()) return false;

	va_list ap;
	va_start(ap, dim1);
	uint adim = va_arg(ap, uint);

	for (uint i = 1; i < shape.size(); ++i) {
		if (shape[i] != adim) return false;
		adim = va_arg(ap, uint);
	}
	if (adim != END_DIM) return false;
	va_end(ap);
	return true;
}

void Tensor::set_releasable(bool releasable) {
	this->releasable = releasable;
}

void* Tensor::_init_(Tensor* tensor, bool copy_val) {
	exit_if(!tensor, "cannot clone tensor from empty tensor");
	shape = tensor->shape;
	span = get_span(&shape);

	if (copy_val) {
		_val_ = malloc(4 * span);
		memcpy(_val_, tensor->_val_, 4 * span);
	} else {
		_val_ = tensor->_val_;
	}
	return _val_;
}

void* Tensor::_init_(vector<uint>* shape, void* val, bool copy_val) {
	this->shape = *shape;
	span = get_span(shape);
	
	if (val && copy_val) {
		_val_ = malloc(4 * span);
		memcpy(_val_, val, 4 * span);
	} else if (!val) {
		_val_ = malloc(4 * span);
	} else {
		_val_ = val;
	}
	return _val_;
}

void* Tensor::_init_(void* val, bool copy_val, uint dim1, va_list ap) {
	if (dim1 == END_DIM) return nullptr;
	vector<uint> running_shape;

	running_shape.push_back(dim1);
	uint adim = va_arg(ap, uint);

	while (adim != END_DIM) {
		running_shape.push_back(adim);
		exit_if(running_shape.size() == MAX_TENSOR_DIM,
			"tensor shape with too many dimensions (%s) : did you forget to end with END_DIM?",
			get_shape_str(&running_shape).c_str());
		adim = va_arg(ap, uint);
	}
	return _init_(&running_shape, val, copy_val);
}

void* Tensor::_init_(ByteStream* stream) {
	_read_meta_(stream);
	_val_ = malloc(4 * span);
	stream->read(_val_, 4, span);
	return _val_;
}

void Tensor::_read_meta_(ByteStream* stream) {
	unsigned char len = 0;
	stream->read(&len, 1, 1);

	for (unsigned char i = 0; i < len; ++i) {
		unsigned int dim = 0;
		stream->read(&dim, 4, 1);
		shape.push_back(dim);
	}
	span = get_span(&shape);
}

void Tensor::_write_meta_(ByteStream* stream) {
	unsigned char len = (unsigned char) shape.size();
	stream->write(&len, 1, 1);

	for (unsigned char i = 0; i < len; ++i) {
		stream->write(&shape[i], 4, 1);
	}
}

void Tensor::_write_val_(ByteStream* stream) {
	stream->write(_val_, 4, span);
}

void Tensor::_save_(ByteStream* stream) {
	_write_meta_(stream);
	_write_val_(stream);
}

ITensor::ITensor() : Tensor() {
	val = nullptr;
}

ITensor::ITensor(ITensor* tensor, bool copy_val) : ITensor() {
	this->val = (int*) _init_(tensor, copy_val);
}

ITensor::ITensor(vector<uint>* shape, int* val, bool copy_val) : ITensor() {
	this->val = (int*) _init_(shape, val, copy_val);
}

ITensor::ITensor(int* val, bool copy_val, uint dim1, ...) : ITensor() {
	va_list ap;
	va_start(ap, dim1);
	this->val = (int*) _init_(val, copy_val, dim1, ap);
	va_end(ap);
}

ITensor::ITensor(uint dim1, ...) : ITensor() {
	va_list ap;
	va_start(ap, dim1);
	this->val = (int*) _init_(nullptr, false, dim1, ap);
	va_end(ap);
};

ITensor::ITensor(ByteStream* stream) : ITensor() {
	char compressed = 0;
	stream->read(&compressed, 1, 1);

	if (compressed) {
		_read_meta_(stream);
		short sh = 0;
		_val_ = val = (int*) malloc(span * 4);

		for (int* vp = val; vp < val + span; ++vp) {
			stream->read(&sh, 2, 1);
			*vp = sh;
		}
	} else {
		val = (int*) _init_(stream);
	}
}

void ITensor::save(ByteStream* stream, bool compressed) {
	char flag = compressed ? 1 : 0;
	stream->write(&flag, 1, 1);

	if (compressed) {
		_write_meta_(stream);
		short sh = 0;

		for (int* vp = val; vp < val + span; ++vp) {
			sh = (short) *vp;
			stream->write(&sh, 2, 1);
		}
	} else {
		_save_(stream);
	}
}

int ITensor::compare(ITensor* tensor) {
	return compare(tensor->val, tensor->span);
}

int ITensor::compare(int* val, uint len) {
	uint comp_len = MIN(span, len);
	int max_abs = 0;
	int* vp1 = this->val;
	int* vp2 = val;

	for (uint i = 0; i < comp_len; ++i, ++vp1, ++vp2) {
		int d = abs(*vp1 - *vp2);
		if (d > max_abs) max_abs = d;
	}
	return max_abs;
}

int ITensor::get_max() {
	return dajnn::get_max(val, span);
}

int ITensor::get_min() {
	return dajnn::get_min(val, span);
}

FTensor::FTensor() : Tensor() {
	val = nullptr;
}

FTensor::FTensor(ITensor* tensor) {
	shape = tensor->shape;
	span = get_span(&shape);
	
	_val_ = val = (float*) malloc(span * 4);
	int* tp = tensor->val;

	for (float* vp = val; vp < val + span; ++vp, ++tp) {
		*vp = (float) *tp;
	}
}

FTensor::FTensor(FTensor* tensor, bool copy_val) : FTensor() {
	this->val = (float*) _init_(tensor, copy_val);
}

FTensor::FTensor(vector<uint>* shape, float* val, bool copy_val) : FTensor() {
	this->val = (float*) _init_(shape, val, copy_val);
}

FTensor::FTensor(float* val, bool copy_val, uint dim1, ...) : FTensor() {
	va_list ap;
	va_start(ap, dim1);
	this->val = (float*) _init_(val, copy_val, dim1, ap);
	va_end(ap);
}

FTensor::FTensor(uint dim1, ...) : FTensor() {
	va_list ap;
	va_start(ap, dim1);
	this->val = (float*) _init_(nullptr, false, dim1, ap);
	va_end(ap);
}

FTensor::FTensor(ByteStream* stream) : FTensor() {
	char compressed = 0;
	stream->read(&compressed, 1, 1);

	if (compressed) {
		_read_meta_(stream);

		float min_v = 0, max_v = 0;
		short sh = 0;

		stream->read(&min_v, 4, 1);
		stream->read(&max_v, 4, 1);
		_val_ = val = (float*) malloc(span * 4);

		for (float* vp = val; vp < val + span; ++vp) {
			stream->read(&sh, 2, 1);
			*vp = min_v + (max_v - min_v) * (1 + (float) sh / SHRT_MAX) / 2;
		}
	} else {
		val = (float*) _init_(stream);
	}
}

void FTensor::save(ByteStream* stream, bool compressed) {
	char flag = compressed ? 1 : 0;
	stream->write(&flag, 1, 1);

	if (compressed) {
		_write_meta_(stream);

		float min_v = get_min();
		float max_v = get_max();
		short sh = 0;

		stream->write(&min_v, 4, 1);
		stream->write(&max_v, 4, 1);

		for (float* vp = val; vp < val + span; ++vp) {
			sh = (short) ((2 * (*vp - min_v) / (max_v - min_v) - 1) * SHRT_MAX);
			stream->write(&sh, 2, 1);
		}
	} else {
		_save_(stream);
	}
}

void FTensor::print(uint start, uint end) {
	if (start == END_DIM) start = 0;
	if (end == END_DIM) end = span;

	for (uint i = start; i < end; ++i) {
		printf("%.8f,", val[i]);
	}
}

float FTensor::compare(FTensor* tensor) {
	return compare(tensor->val, tensor->span);
}

float FTensor::compare(float* val, uint len) {
	uint comp_len = MIN(span, len);
	float max_abs = 0;
	float* vp1 = this->val;
	float* vp2 = val;

	for (uint i = 0; i < comp_len; ++i, ++vp1, ++vp2) {
		float d = fabsf(*vp1 - *vp2);
		if (d > max_abs) max_abs = d;
	}
	return max_abs;
}

float FTensor::get_max() {
	return dajnn::get_max(val, span);
}

float FTensor::get_min() {
	return dajnn::get_min(val, span);
}

ByteStream::ByteStream() {
	buff = nullptr;
	fp = nullptr;
	pointer = 0;
}

ByteStream::ByteStream(const void* buff) : ByteStream() {
	this->buff = (const char*) buff;
}

ByteStream::ByteStream(FILE* fp) : ByteStream() {
	this->fp = fp;
}

string ByteStream::read_str() {
	string str;
	char t = 0;

	for (uint i = 0; i < MAX_MODEL_STR; ++i) {
		if (!read(&t, 1, 1)) break;
		if (!t) break;
		str += t;
	}
	return str;
}

uint ByteStream::read(void* dst, int ele_size, int ele_count) {
	if (buff) {
		int len = ele_size * ele_count;
		memcpy(dst, &buff[pointer], len);
		pointer += len;
		return ele_count;
	} else if (fp) {
		return (uint) fread(dst, ele_size, ele_count, fp);
	} else {
		return 0;
	}
}

void ByteStream::write(void* src, int ele_size, int ele_count) {
	if (buff) {
		int len = ele_size * ele_count;
		memcpy((char*) &buff[pointer], src, len);
		pointer += len;
	} else if (fp) {
		fwrite(src, ele_size, ele_count, fp);
	}
}

int ByteStream::seek() {
	return pointer;
}

}
