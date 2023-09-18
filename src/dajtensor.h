
#pragma once

#include "dajnn.h"

namespace dajnn {

class ByteStream {
public:
	ByteStream();
	ByteStream(const void* buff);
	ByteStream(FILE* fp);

	string read_str();
	uint read(void* dst, int ele_size, int ele_count);
	void write(void* src, int ele_size, int ele_count);
	int seek();

private:
	const char* buff;
	FILE* fp;
	int pointer;
};

class Tensor {
public:
	Tensor();
	virtual ~Tensor();

	void reshape(vector<uint>* shape);
	void reshape(uint dim1, ...);
	bool is_shape(vector<uint>* shape);
	bool is_shape(uint dim1, ...);
	void set_releasable(bool releasable);

public:
	vector<uint> shape;
	uint span;
	bool releasable;

protected:
	void* _init_(Tensor* tensor, bool copy_val = true);
	void* _init_(vector<uint>* shape, void* val = nullptr, bool copy_val = true);
	void* _init_(void* val, bool copy_val, uint dim1, va_list ap);
	void* _init_(ByteStream* stream);

	void _read_meta_(ByteStream* stream);
	void _write_meta_(ByteStream* stream);
	void _write_val_(ByteStream* stream);
	void _save_(ByteStream* stream);
	
protected:
	void* _val_;
};

class ITensor : public Tensor {
public:
	ITensor();
	ITensor(ITensor* tensor, bool copy_val = true);
	ITensor(vector<uint>* shape, int* val = nullptr, bool copy_val = true);
	ITensor(int* val, bool copy_val, uint dim1, ...);
	ITensor(uint dim1, ...);
	ITensor(ByteStream* stream);

	void save(ByteStream* stream, bool compressed = false);
	int compare(ITensor* tensor);
	int compare(int* val, uint len);
	
	int get_max();
	int get_min();

public:
	int* val;
};

class FTensor : public Tensor {
public:
	FTensor();
	FTensor(FTensor* tensor, bool copy_val = true);
	FTensor(ITensor* tensor);
	FTensor(vector<uint>* shape, float* val = nullptr, bool copy_val = true);
	FTensor(float* val, bool copy_val, uint dim1, ...);
	FTensor(uint dim1, ...);
	FTensor(ByteStream* stream);

	void save(ByteStream* stream, bool compressed = false);
	void print(uint start = END_DIM, uint end = END_DIM);

	float compare(FTensor* tensor);
	float compare(float* val, uint len);

	float get_max();
	float get_min();
	
public:
	float* val;
};

}
