
#pragma once

#include "dajtensor.h"

namespace dajnn {

void log_i(const char* format, ...);
void log_w(const char* format, ...);
void log_d(const char* format, ...);
void log_e(const char* format, ...);
void log_x(const char* type_str, const char* format, va_list ap);

void exit_if(bool condition, const char* format = nullptr, ...);

float get_max(float* arr, uint len);
float get_min(float* arr, uint len);

int get_max(int* arr, uint len);
int get_min(int* arr, uint len);

uint get_span(vector<uint>* shape);
string get_shape_str(vector<uint>* shape);
string format_str(const char* format, ...);

}
