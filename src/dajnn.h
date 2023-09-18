
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <memory.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <random>
#include <vector>
#include <algorithm>
#include <string>
#include "dajdef.h"

#ifdef PADDLE
#include <jni.h>
#include <android/log.h>
#endif

using namespace std;

namespace dajnn {

#ifndef uchar
#define uchar unsigned char
#endif

#ifndef ushort
#define ushort unsigned short
#endif

#ifndef uint
#define uint unsigned int
#endif

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#ifndef INT_MIN
#define INT_MIN -2147483648
#define INT_MAX 2147483647
#endif

#ifndef FLOAT_MIN
#define FLOAT_MIN -1e10f
#define FLOAT_MAX 1e10f
#endif

#ifndef SHRT_MIN
#define SHRT_MIN -32768
#define SHRT_MAX 32767
#endif

#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif

#define END_DIM 0xFFFFFFFF
#define MAX_TENSOR_DIM 16
#define MAX_MODEL_STR 256

#define MODEL_HEADER "MRB_NN_DAJ_MODEL_V1_BEGIN"
#define MODEL_FOOTER "MRB_NN_DAJ_MODEL_END"

class Tensor;
class ITensor;
class FTensor;

void init_dajnn();
void finish_dajnn();

#ifdef TRACE_MEMORY_LEAK
void push_tensor_trace(Tensor* tensor);
uint pop_tensor_trace(Tensor* tensor);

vector<uint> get_leaked_tensor_indice();
#endif

}
