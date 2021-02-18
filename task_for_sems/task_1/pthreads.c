/**
 * @file pthread.c
 * @author Vinogradov Aleksandr (vinogradov.alek@gmail.com)
 * @brief Just create threads and print IDs
 * @version 0.1
 * @date 2021-02-11
 * 
 * @copyright Copyright (c) 2021
 * 
 * @note Compile with { -pthread }
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// required for pthreads
#include <pthread.h>

void *thread_job(void *arg)
{
    printf("Hello i'm thread with ID: %ld and i %d in a row\n", pthread_self(), *(int *)arg);
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "bad args\n");
        return -1;
    }

    int thread_amount = atoi(argv[1]);

    if (thread_amount < 1)
    {
        fprintf(stderr, "thread amount less then 1\n");
        return -1;
    }

    // create array of threads
    pthread_t *tids = (pthread_t *)calloc(thread_amount, sizeof(pthread_t));
    assert(tids);

    // create array with agrs for threads
    int *targs = (int *)calloc(thread_amount, sizeof(int));
    assert(targs);

    // init attributes with default values
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    // creating threads
    for (size_t i = 0; i < thread_amount; i++)
    {
        targs[i] = i;
        int status = pthread_create(&(tids[i]), &attr, thread_job, &targs[i]);
        assert(!status);
    }

    // waiting for threads to terminate
    for (size_t i = 0; i < thread_amount; i++)
    {
        int status = pthread_join(tids[i], NULL);
        assert(!status);
    }

    return 0;
}