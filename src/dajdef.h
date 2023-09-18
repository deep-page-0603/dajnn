
#pragma once

#ifdef _WIN32

#ifdef _DEBUG
#define TRACE_MEMORY_LEAK
#endif

#else // _WIN32

#define LITE_WITH_ARM
#define PADDLE
#define PADDLE_THREADS 2
#define PADDLE_CLS 1

#endif
