#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>

#define ARG_AMOUNT 1 + 6 + 1 + 1

int read_from_csv(const char *filename, double *matrix, int M, int N)
{
    FILE *file = fopen(filename, "r");

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            fscanf(file, "%le", &matrix[i * N + j]);
            if (j != N - 1)
            {
                fscanf(file, ",");
            }
        }
        fscanf(file, "\n");
    }
    fclose(file);
    return 0;
}

int save_to_csv(const char *filename, double *matrix, int M, int N)
{
    FILE *file = fopen(filename, "w");

    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            fprintf(file, "%1.18le", matrix[i * N + j]);
            if (j != N - 1)
            {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);
    return 0;
}

void obvious(int M, int N, int K, const double *A, const double *B, double *C)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C[i * N + j] = 0;
            for (int k = 0; k < K; ++k)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
    }
}

void linear_access(int M, int N, int K, const double *A, const double *B, double *C)
{
    for (int i = 0; i < M; ++i)
    {
        double *c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const double *b = B + k * N;
            double a = A[i * K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
}

int main(int argc, char **argv)
{
    assert(argc == ARG_AMOUNT);

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    double *A = malloc(M * K * sizeof(double));
    double *B = malloc(K * N * sizeof(double));
    double *C = malloc(M * N * sizeof(double));

    assert(read_from_csv(argv[4], A, M, K) == 0);
    assert(read_from_csv(argv[5], B, K, N) == 0);

    struct timeval time_before, time_after;

    gettimeofday(&time_before, NULL);

    switch (atoi(argv[7]))
    {
    case 0:
        obvious(M, N, K, A, B, C);
        break;
    case 1:
        linear_access(N, N, K, A, B, C);
        break;
    }

    gettimeofday(&time_after, NULL);

    printf("%ld\n microsec execution time\n", (time_after.tv_sec - time_before.tv_sec) * 1000 * 1000 + time_after.tv_usec - time_before.tv_usec);

    assert(save_to_csv(argv[6], C, M, N) == 0);
    return 0;
}