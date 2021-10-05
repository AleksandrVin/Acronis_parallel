#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <assert.h>
#include <sys/time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <immintrin.h>

#define _ISOC11_SOURCE
#define _GNU_SOURCE

#define ARG_AMOUNT 1 + 6 + 1 + 1

#define L1_CACHE_SIZE 4000
#define BLOCK_SIZE (L1_CACHE_SIZE / sizeof(double))

struct Matrix
{
    double *elements;
    size_t size_i, boosted_size_i;
    size_t size_j, boosted_size_j;
};

int read_from_csv(const char *filename, struct Matrix *matrix)
{
    FILE *file = fopen(filename, "r");
    int scanned;

    for (size_t i = 0; i < matrix->size_i; i++)
    {
        for (size_t j = 0; j < matrix->size_j; j++)
        {
            scanned = fscanf(file, "%le", matrix->elements + i * matrix->boosted_size_j + j);
            if (j != matrix->size_j - 1)
            {
                scanned = fscanf(file, ",");
            }
        }
        scanned = fscanf(file, "\n");
    }
    scanned+= 2;
    fclose(file);
    return 0;
}

int save_to_csv(const char *filename, struct Matrix *matrix)
{
    FILE *file = fopen(filename, "w");

    for (size_t i = 0; i < matrix->size_i; i++)
    {
        for (size_t j = 0; j < matrix->size_j; j++)
        {
            fprintf(file, "%1.18le", matrix->elements[i * matrix->boosted_size_j + j]);
            if (j != matrix->size_j - 1)
            {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }
    fclose(file);
    return 0;
}

void obvious(struct Matrix *mA, struct Matrix *mB, struct Matrix *mC)
{
    for (int i = 0; i < mC->size_i; ++i)
    {
        for (int j = 0; j < mC->size_j; ++j)
        {
            mC->elements[i * mC->boosted_size_j + j] = 0;
            for (int k = 0; k < mA->size_j; ++k)
                mC->elements[i * mC->boosted_size_j + j] += mA->elements[i * mA->boosted_size_j + k] * mB->elements[k * mB->boosted_size_j + j];
        }
    }
}

void linear_access(struct Matrix *mA, struct Matrix *mB, struct Matrix *mC)
{
    for (int i = 0; i < mA->size_i; ++i)
    {
        double *c = mC->elements + i * mC->boosted_size_j;
        for (int j = 0; j < mC->size_j; ++j)
            c[j] = 0;
        for (int k = 0; k < mA->size_j; ++k)
        {
            const double *b = mB->elements + k * mB->boosted_size_j;
            double a = mA->elements[i * mA->boosted_size_j + k];
            for (int j = 0; j < mC->size_j; ++j)
                c[j] += a * b[j];
        }
    }
}

/// blocked part
void matrix_alloc(int size_i, int size_j, struct Matrix *matrix)
{
    matrix->size_i = size_i;
    matrix->size_j = size_j;

    matrix->boosted_size_i = BLOCK_SIZE * (size_i / BLOCK_SIZE + ((size_i % BLOCK_SIZE == 0) ? 0 : 1));
    matrix->boosted_size_j = BLOCK_SIZE * (size_j / BLOCK_SIZE + ((size_j % BLOCK_SIZE == 0) ? 0 : 1));

    matrix->elements = aligned_alloc(L1_CACHE_SIZE, matrix->boosted_size_i * matrix->boosted_size_j * sizeof(double));
    assert(matrix->elements != NULL);
}

void calc_block(struct Matrix *mA, struct Matrix *mB, struct Matrix *mC, int block_pos_i, int block_pos_j)
{
    // zero matrix C block
    for (int i = block_pos_i; i < block_pos_i + BLOCK_SIZE; i++)
    {
        for (int j = block_pos_j; j < block_pos_j + BLOCK_SIZE; j++)
        {
            mC->elements[i * mC->boosted_size_j + j] = 0;
        }
    }

    // obvious access
    /*     for (int step = 0; step < mA->boosted_size_i; step += BLOCK_SIZE)
    {
        for (int c_i = 0; c_i < BLOCK_SIZE; c_i++)
        {
            for (int c_j = 0; c_j < BLOCK_SIZE; c_j++)
            {
                for (int p = 0; p < BLOCK_SIZE; p++)
                {
                    mC->elements[(block_pos_i + c_i) * mC->boosted_size_j + block_pos_j + c_j] += mA->elements[(c_i + block_pos_i) * mA->boosted_size_j + step + p] * mB->elements[(step + p) * mB->boosted_size_j + block_pos_j + c_j];
                }
            }
        }
    }  */

    // linearized access
    for (int step = 0; step < mA->boosted_size_i; step += BLOCK_SIZE)
    {
        for (int c_i = 0; c_i < BLOCK_SIZE; c_i++)
        {
            for (int c_j = 0; c_j < BLOCK_SIZE; c_j++)
            {
                double a = mA->elements[(c_i + block_pos_i) * mA->boosted_size_j + c_j + step];
                for (int p = 0; p < BLOCK_SIZE; p++)
                {
                    mC->elements[(block_pos_i + c_i) * mC->boosted_size_j + block_pos_j + p] += a * mB->elements[(step + c_j) * mB->boosted_size_j + block_pos_j + p];
                }
            }
        }
    }
}

void blocked(struct Matrix *mA, struct Matrix *mB, struct Matrix *mC)
{
    for (int block_pos_i = 0; block_pos_i < mC->boosted_size_i; block_pos_i += BLOCK_SIZE)
    {
        for (int block_pos_j = 0; block_pos_j < mC->boosted_size_j; block_pos_j += BLOCK_SIZE)
        {
            calc_block(mA, mB, mC, block_pos_i, block_pos_j);
        }
    }
}

struct ThreadArgs
{
    pthread_t thread_id;

    struct Matrix *mA;
    struct Matrix *mB;
    struct Matrix *mC;

    int thread;
    int threads_amount;
};

void *thread_job(void *arg)
{
    struct ThreadArgs *thread_args = (struct ThreadArgs *)arg;

    for (size_t block_pos_i = thread_args->thread * BLOCK_SIZE; block_pos_i < thread_args->mC->boosted_size_i; block_pos_i += thread_args->threads_amount * BLOCK_SIZE)
    {
        for (size_t block_pos_j = 0; block_pos_j < thread_args->mC->boosted_size_j; block_pos_j += BLOCK_SIZE)
        {
            calc_block(thread_args->mA, thread_args->mB, thread_args->mC, block_pos_i, block_pos_j);
        }
    }

    return NULL;
}

void run_threaded(struct Matrix *mA, struct Matrix *mB, struct Matrix *mC, int threads_amount)
{
    struct ThreadArgs *args = calloc(threads_amount, sizeof(struct ThreadArgs));
    assert(args != NULL);

    pthread_attr_t attr;
    if (pthread_attr_init(&attr) != 0)
    {
        fprintf(stderr, "can't init theading");
        exit(EXIT_FAILURE);
    }

    for (int thread = 0; thread < threads_amount; thread++)
    {
        args[thread] = (struct ThreadArgs){
            .mA = mA,
            .mB = mB,
            .mC = mC,
            .thread = thread,
            .threads_amount = threads_amount};

        if (pthread_create(&args[thread].thread_id, &attr, &thread_job, &args[thread]) != 0)
        {
            fprintf(stderr, "can't create threads");
            exit(EXIT_FAILURE);
        }
    }

    for (size_t thread = 0; thread < threads_amount; ++thread)
    {
        if (pthread_join(args[thread].thread_id, NULL) != 0)
        {
            fprintf(stderr, "[Error] Unable to join a thread");
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char **argv)
{
    assert(argc == ARG_AMOUNT);

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    int threads_amount = atoi(argv[8]);

    struct Matrix mA, mB, mC;

    matrix_alloc(M, K, &mA);
    matrix_alloc(K, N, &mB);
    matrix_alloc(M, N, &mC);

    assert(read_from_csv(argv[4], &mA) == 0);
    assert(read_from_csv(argv[5], &mB) == 0);

    struct timeval time_before, time_after;

    gettimeofday(&time_before, NULL);

    switch (atoi(argv[7]))
    {
    case 0:
        obvious(&mA, &mB, &mC);
        break;
    case 1:
        linear_access(&mA, &mB, &mC);
        break;
    case 2:
        if (threads_amount == 1)
        {
            blocked(&mA, &mB, &mC);
            break;
        }
        else if (threads_amount > 1)
        {
            run_threaded(&mA, &mB, &mC, threads_amount);
            break;
        }
        fprintf(stderr, "threads_amount less then zero");
        exit(-1);
    default:
        fprintf(stderr, "error during algorithm selection");
        break;
    }

    gettimeofday(&time_after, NULL);

    printf("%ld\n microsec execution time\n", (time_after.tv_sec - time_before.tv_sec) * 1000 * 1000 + time_after.tv_usec - time_before.tv_usec);

    assert(save_to_csv(argv[6], &mC) == 0);

    printf("Block size is %ld \n", BLOCK_SIZE);
    return 0;
}