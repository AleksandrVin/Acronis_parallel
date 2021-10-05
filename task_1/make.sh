#!/bin/bash

cc -std=c11 -Werror -Wall -g -pthread -O0 -o matrix_multiply.elf matrix_multiply.c