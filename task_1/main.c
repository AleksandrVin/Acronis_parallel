#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <error.h>
#include <sys/time.h>
#include <assert.h>
#include <immintrin.h>
#include <sys/param.h>

#define L1 32 * 1024
#define L2 256 * 1024
#define L3 2.5 * 1024 * 1024 // 45Mb per cpua

#define file_a "matrix_a"
#define file_b "matrix_b"
#define file_out "matrix"

struct  buf_t
{
    float * p;
    int n;

    struct buf_t(int size) : n(size), p((float*)_mm_malloc(size * 4, 64)) {}
    ~struct buf_t() { _mm_free(p); }
};

void micro_6x16(int K, const float *A, int lda, int step,
                const float *B, int ldb, float *C, int ldc)
{
    __m256 c00 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();
    const int offset0 = lda * 0;
    const int offset1 = lda * 1;
    const int offset2 = lda * 2;
    const int offset3 = lda * 3;
    const int offset4 = lda * 4;
    const int offset5 = lda * 5;
    __m256 b0, b1, a0, a1;
    for (int k = 0; k < K; k++)
    {
        b0 = _mm256_loadu_ps(B + 0);
        b1 = _mm256_loadu_ps(B + 8);
        a0 = _mm256_set1_ps(A[offset0]);
        a1 = _mm256_set1_ps(A[offset1]);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        a0 = _mm256_set1_ps(A[offset2]);
        a1 = _mm256_set1_ps(A[offset3]);
        c20 = _mm256_fmadd_ps(a0, b0, c20);
        c21 = _mm256_fmadd_ps(a0, b1, c21);
        c30 = _mm256_fmadd_ps(a1, b0, c30);
        c31 = _mm256_fmadd_ps(a1, b1, c31);
        a0 = _mm256_set1_ps(A[offset4]);
        a1 = _mm256_set1_ps(A[offset5]);
        c40 = _mm256_fmadd_ps(a0, b0, c40);
        c41 = _mm256_fmadd_ps(a0, b1, c41);
        c50 = _mm256_fmadd_ps(a1, b0, c50);
        c51 = _mm256_fmadd_ps(a1, b1, c51);
        B += ldb;
        A += step;
    }
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c00, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c01, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c10, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c11, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c20, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c21, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c30, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c31, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c40, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c41, _mm256_loadu_ps(C + 8)));
    C += ldc;
    _mm256_storeu_ps(C + 0, _mm256_add_ps(c50, _mm256_loadu_ps(C + 0)));
    _mm256_storeu_ps(C + 8, _mm256_add_ps(c51, _mm256_loadu_ps(C + 8)));
}

void reorder_b_16(int K, const float *B, int ldb, float *bufB)
{
    for (int k = 0; k < K; ++k, B += ldb, bufB += 16)
    {
        _mm256_storeu_ps(bufB + 0, _mm256_loadu_ps(B + 0));
        _mm256_storeu_ps(bufB + 8, _mm256_loadu_ps(B + 8));
    }
}

void macro_v7(int M, int N, int K, const float *A,
              const float *B, int ldb, float *bufB, bool reorderB, float *C, int ldc)
{
    for (int j = 0; j < N; j += 16)
    {
        if (reorderB)
            reorder_b_16(K, B + j, ldb, bufB + K * j);
        for (int i = 0; i < M; i += 6)
            micro_6x16(K, A + i * K, 1, 6, bufB + K * j, 16, C + i * ldc + j, ldc);
    }
}

void reorder_a_6(const float *A, int lda, int M, int K, float *bufA)
{
    for (int i = 0; i < M; i += 6)
    {
        for (int k = 0; k < K; k += 4)
        {
            const float *pA = A + k;
            __m128 a0 = _mm_loadu_ps(pA + 0 * lda);
            __m128 a1 = _mm_loadu_ps(pA + 1 * lda);
            __m128 a2 = _mm_loadu_ps(pA + 2 * lda);
            __m128 a3 = _mm_loadu_ps(pA + 3 * lda);
            __m128 a4 = _mm_loadu_ps(pA + 4 * lda);
            __m128 a5 = _mm_loadu_ps(pA + 5 * lda);
            __m128 a00 = _mm_unpacklo_ps(a0, a2);
            __m128 a01 = _mm_unpacklo_ps(a1, a3);
            __m128 a10 = _mm_unpackhi_ps(a0, a2);
            __m128 a11 = _mm_unpackhi_ps(a1, a3);
            __m128 a20 = _mm_unpacklo_ps(a4, a5);
            __m128 a21 = _mm_unpackhi_ps(a4, a5);
            _mm_storeu_ps(bufA + 0, _mm_unpacklo_ps(a00, a01));
            _mm_storel_pi((__m64 *)(bufA + 4), a20);
            _mm_storeu_ps(bufA + 6, _mm_unpackhi_ps(a00, a01));
            _mm_storeh_pi((__m64 *)(bufA + 10), a20);
            _mm_storeu_ps(bufA + 12, _mm_unpacklo_ps(a10, a11));
            _mm_storel_pi((__m64 *)(bufA + 16), a21);
            _mm_storeu_ps(bufA + 18, _mm_unpackhi_ps(a10, a11));
            _mm_storeh_pi((__m64 *)(bufA + 22), a21);
            bufA += 24;
        }
        A += 6 * lda;
    }
}

void init_c(int M, int N, float * C, int ldc)
{
    for (int i = 0; i < M; ++i, C += ldc)
        for (int j = 0; j < N; j += 8)
            _mm256_storeu_ps(C + j, _mm256_setzero_ps());
}

int main(int argc, char **argv)
{

    int M = 400;
    int K = M;
    int N = M;
    if (argc != 3)
    {
        fprintf(stderr, "provide method and threads as arg");
        return (-1);
    }
    int threads = atoi(argv[2]);

    assert(M % threads == 0 && N % threads == 0 && K % threads == 0);
    FILE *f_a = fopen(file_a, "r");
    FILE *f_b = fopen(file_b, "r");

    float *matrix_a = (float *)calloc(M * K, sizeof(float));
    float *matrix_b = (float *)calloc(K * N, sizeof(float));

    fread(matrix_a, sizeof(float), M * K, f_a);
    fread(matrix_b, sizeof(float), K * N, f_b);

    float *matrix_c = (float *)calloc(M * N, sizeof(float));

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            matrix_c[i * N + j] = 0;
        }
    }

    // time
    struct timeval tv_start, tv_end;

    switch (atoi(argv[1]))
    {
    case 0: // basic single thread
        printf("method 0 - \"basic\" executed in single thread\n");
        gettimeofday(&tv_start, NULL);
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                for (size_t p = 0; p < K; p++)
                    matrix_c[i * N + j] += matrix_a[i * K + p] * matrix_b[p * N + j];
            }
        }
        break;
    case 1: // optimised address, single thread
        printf("method 2 - \"optimized basic\" in single thread\n");
        gettimeofday(&tv_start, NULL);
        for (int i = 0; i < M; ++i)
        {
            float *c = matrix_c + i * N;
            for (int j = 0; j < N; ++j)
                c[j] = 0;
            for (int k = 0; k < K; ++k)
            {
                const float *b = matrix_b + k * N;
                float a = matrix_a[i * K + k];
                for (int j = 0; j < N; ++j)
                    c[j] += a * b[j];
            }
        }
        break;
    case 2: // max version
    {
        int mK = MIN(L1 / 4 / 16, K) / 4 * 4;
        int mM = MIN(L2 / 4 / mK, M) / 6 * 6;
        int mN = MIN(L3 / 4 / mK, N) / 16 * 16;
        struct buf_t bufB(mN * mK);
        struct buf_t bufA(mK * mM);
        for (int j = 0; j < N; j += mN)
        {
            int dN = MIN(N, j + mN) - j;
            for (int k = 0; k < K; k += mK)
            {
                int dK = MIN(K, k + mK) - k;
                for (int i = 0; i < M; i += mM)
                {
                    int dM = MIN(M, i + mM) - i;
                    if (k == 0)
                        init_c(dM, dN, matrix_c + i * N + j, N);
                    reorder_a_6(matrix_a + i * K + k, K, dM, dK, bufA.p);
                    macro_v7(dM, dN, dK, bufA.p, matrix_b + k * N + j, N, bufB.p, i == 0, matrix_c + i * N + j, N);
                }
            }
        }
    }

    default:
        printf("looks like there no such method\n");
        return -1;
        break;
    }

    // timeend

    gettimeofday(&tv_end, NULL);
    unsigned long time_in_micros_start = 1000000 * tv_start.tv_sec + tv_start.tv_usec;
    unsigned long time_in_micros_end = 1000000 * tv_end.tv_sec + tv_end.tv_usec;
    printf("time = %lu\n", time_in_micros_end - time_in_micros_start);

    /* FILE *f_out = fopen(file_out, "w");
    fwrite(matrix_c, sizeof(int), N1 * N2, f_out); */

    for (size_t i = 0; i < M; i++)
    {
        if (matrix_c[i * N + i] == 0)
        {
            fprintf(stderr, "test failed\n");
            return -1;
        }
    }

    return 0;
}