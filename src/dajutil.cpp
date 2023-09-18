
#include "dajutil.h"

namespace dajnn {

void log_i(const char* format, ...) {
	va_list ap;
    va_start(ap, format);
    log_x("[info]", format, ap);
	va_end(ap);
}

void log_w(const char* format, ...) {
	va_list ap;
    va_start(ap, format);
    log_x("[warn]", format, ap);
	va_end(ap);
}

void log_d(const char* format, ...) {
	va_list ap;
    va_start(ap, format);
    log_x("[debug]", format, ap);
	va_end(ap);
}

void log_e(const char* format, ...) {
	va_list ap;
    va_start(ap, format);
    log_x("[error]", format, ap);
	va_end(ap);
}

void log_x(const char* type_str, const char* format, va_list ap) {
	char msg[1024];
#ifdef _WIN32
	sprintf_s(msg, 1024, format, ap);
	printf_s("%s %s\n", type_str, msg);
#else
	vsprintf(msg, format, ap);
	__android_log_print(ANDROID_LOG_ERROR, type_str, msg)
#endif	
}

void exit_if(bool condition, const char* format, ...) {
	if (condition) {
		if (format) {
			va_list ap;
			va_start(ap, format);
			log_e(format, ap);
			va_end(ap);
		}
		exit(-1);
	}
}

float get_max(float* arr, uint len) {
	float m = FLOAT_MIN;

	for (float* ap = arr; ap < arr + len; ++ap) {
		if (m < *ap) m = *ap;
	}
	return m;
}

float get_min(float* arr, uint len) {
	float m = FLOAT_MAX;

	for (float* ap = arr; ap < arr + len; ++ap) {
		if (m > *ap) m = *ap;
	}
	return m;
}

int get_max(int* arr, uint len) {
	int m = INT_MIN;

	for (int* ap = arr; ap < arr + len; ++ap) {
		if (m < *ap) m = *ap;
	}
	return m;
}

int get_min(int* arr, uint len) {
	int m = INT_MAX;

	for (int* ap = arr; ap < arr + len; ++ap) {
		if (m > *ap) m = *ap;
	}
	return m;
}

uint get_span(vector<uint>* shape) {
	uint span = 1;

	for (vector<uint>::iterator dim = shape->begin(); dim != shape->end(); ++dim) {
		span *= *dim;
	}
	return span;
}

string get_shape_str(vector<uint>* shape) {
	string str = "(";

	for (uint i = 0; i < shape->size(); ++i) {
		if (i > 0) str += ",";
		str += format_str("%d", shape->at(i));
	}
	str += ")";
	return str;
}

string format_str(const char* format, ...) {
	char str[1024];
	va_list ap;
	va_start(ap, format);
#ifdef _WIN32
	sprintf_s(str, 1024, format, ap);	
#else
	vsprintf(str, format, ap);
#endif
	va_end(ap);
	return string(str);
}

}
