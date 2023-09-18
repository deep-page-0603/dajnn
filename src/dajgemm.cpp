
#include "dajgemm.h"
#include "dajutil.h"

namespace dajnn {

void gemm_bin(int M, int N, int K, float ALPHA, char* A, int lda, float* B, int ldb, float* C, int ldc) {
	int i, j, k;

	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			char A_PART = A[i * lda + k];

			if (A_PART) {
				for (j = 0; j < N; ++j) {
					C[i * ldc + j] += B[k * ldb + j];
				}
			} else {
				for (j = 0; j < N; ++j) {
					C[i * ldc + j] -= B[k * ldb + j];
				}
			}
		}
	}
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float BETA, float* C, int ldc) {
	gemm_cpu(TA, TB, M, N, K, ALPHA, A, lda, B, ldb, BETA, C, ldc);
}

//#if (defined(__AVX__) && defined(__x86_64__)) || defined(_WIN64)
#if defined(__AVX__) || defined(_WIN64)
#define OSXSAVEFlag (1UL << 27)
#define AVXFlag     ((1UL << 28) | OSXSAVEFlag)
#define FMAFlag     ((1UL << 12) | AVXFlag | OSXSAVEFlag)
#define CLMULFlag   ((1UL << 1) | AVXFlag | OSXSAVEFlag)
#define VAESFlag    ((1UL << 25) | AVXFlag | OSXSAVEFlag)

#include <stdint.h>

//#ifdef _WIN64
#ifdef _WIN32
#include <intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#else	// Linux GCC/Clang
#include <x86intrin.h>
#include <ammintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <cpuid.h>

void asm_cpuid(uint32_t* abcd, uint32_t eax) {
	uint32_t ebx = 0, edx = 0, ecx = 0;

	// EBX is saved to EDI and later restored
	__asm__("movl %%ebx, %%edi;"
		"cpuid;"
		"xchgl %%ebx, %%edi;"
		: "=D"(ebx),
		"+a"(eax), "+c"(ecx), "=d"(edx));

	abcd[0] = eax;
	abcd[1] = ebx;
	abcd[2] = ecx;
	abcd[3] = edx;
}
#endif

int simd_detect_x86(unsigned int idFeature) {
	uint32_t regs[4];	// EAX, EBX, ECX, EDX;
#ifdef _WIN32
	__cpuid((int*) regs, 0);
	if (regs[0] > 1U) __cpuid((int*) regs, 1);
#else
	__get_cpuid(0, &regs[0], &regs[1], &regs[2], &regs[3]);
	if (regs[0] > 1U) __get_cpuid(1, &regs[0], &regs[1], &regs[2], &regs[3]);
#endif
	if ((regs[2] & idFeature) != idFeature) return 0;
	return 1;
}

int is_fma_avx() {
	static int result = -1;

	if (result == -1) {
		result = simd_detect_x86(AVXFlag);
		
		if (result == 1) {
			log_i(" used AVX");
		} else {
			log_i(" not used AVX");
		}
	}
	return result;
}

void gemm_nn(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc) {
	int i, j, k;

	if (is_fma_avx() == 1) {	// AVX
		for (i = 0; i < M; ++i) {
			for (k = 0; k < K; ++k) {
				float A_PART = ALPHA * A[i * lda + k];
				__m256 a256, b256, c256, result256;	// AVX
				a256 = _mm256_set1_ps(A_PART);

				for (j = 0; j < N - 8; j += 8) {
					b256 = _mm256_loadu_ps(&B[k * ldb + j]);
					c256 = _mm256_loadu_ps(&C[i * ldc + j]);
					
					// FMA - Intel Haswell (2013), AMD Piledriver (2012)
					result256 = _mm256_fmadd_ps(a256, b256, c256);
					//result256 = _mm256_mul_ps(a256, b256);
					//result256 = _mm256_add_ps(result256, c256);
					
					_mm256_storeu_ps(&C[i * ldc + j], result256);
				}
				int prev_end = (N % 8 == 0) ? (N - 8) : (N / 8) * 8;

				for (j = prev_end; j < N; ++j)
					C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	} else {
		for (i = 0; i < M; ++i) {
			for (k = 0; k < K; ++k) {
				register float A_PART = ALPHA * A[i * lda + k];
				
				for (j = 0; j < N; ++j) {
					C[i * ldc + j] += A_PART * B[k * ldb + j];
				}
				/* // SSE
				__m128 a128, b128, c128, result128;	// SSE
				a128 = _mm_set1_ps(A_PART);
				for (j = 0; j < N - 4; j += 4) {
				b128 = _mm_loadu_ps(&B[k*ldb + j]);
				c128 = _mm_loadu_ps(&C[i*ldc + j]);
				//result128 = _mm_fmadd_ps(a128, b128, c128);
				result128 = _mm_mul_ps(a128, b128);
				result128 = _mm_add_ps(result128, c128);
				_mm_storeu_ps(&C[i*ldc + j], result128);
				}

				int prev_end = (N % 4 == 0) ? (N - 4) : (N / 4) * 4;
				for (j = prev_end; j < N; ++j){
				C[i*ldc + j] += A_PART*B[k*ldb + j];
				}
				*/
			}
		}
	}
}
#else

void gemm_nn(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc) {
	int i, j, k;

	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[i * lda + k];

			for (j = 0; j < N; ++j) {
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}

#endif	// __x86_64

void gemm_nt(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc) {
	int i, j, k;

	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;

			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
			}
			C[i * ldc + j] += sum;
		}
	}
}

void gemm_tn(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc) {
	int i, j, k;

	for (i = 0; i < M; ++i) {
		for (k = 0; k < K; ++k) {
			register float A_PART = ALPHA * A[k * lda + i];

			for (j = 0; j < N; ++j) {
				C[i * ldc + j] += A_PART * B[k * ldb + j];
			}
		}
	}
}

void gemm_tt(int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float* C, int ldc) {
	int i, j, k;

	for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			register float sum = 0;

			for (k = 0; k < K; ++k) {
				sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
			}
			C[i * ldc + j] += sum;
		}
	}
}

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, float* A, int lda, float* B, int ldb, float BETA, float* C, int ldc) {
	int i, j;

	/*for (i = 0; i < M; ++i) {
		for (j = 0; j < N; ++j) {
			C[i * ldc + j] *= BETA;
		}
	}*/
	int t;

#pragma omp parallel for
	for (t = 0; t < M; ++t) {
		if (!TA && !TB) {
			gemm_nn(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
		} else if (TA && !TB) {
			gemm_tn(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
		} else if (!TA && TB) {
			gemm_nt(1, N, K, ALPHA, A + t * lda, lda, B, ldb, C + t * ldc, ldc);
		} else {
			gemm_tt(1, N, K, ALPHA, A + t, lda, B, ldb, C + t * ldc, ldc);
		}
	}
}

}
