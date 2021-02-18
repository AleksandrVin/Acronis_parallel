/**
 * @file pthread.cpp
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
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

#include <thread>
#include <unistd.h>

void thread_job(int born_sequence)
{
    std::stringstream out_stream;
    out_stream << "Hello i'm thread with ID: " << std::this_thread::get_id() << " and i " << born_sequence << " in a row " << getpid() << std::endl;
    std::cout << out_stream.str();
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

    std::vector<std::thread> t_vector(thread_amount);

    int i = 0;
    for (auto &thread : t_vector)
    {
        thread = std::thread(thread_job, i);
        i++;
    }

    for (auto &thread : t_vector)
    {
        thread.join();
    }
    return 0;
}