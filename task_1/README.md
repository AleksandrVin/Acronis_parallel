# CPU cache effective matrix multiplication

## Launch verifier.py to test with varaity of options

verifier will do all work for you

1. Compiling
2. Creating test matrixes
3. running
4. verifing result
5. printing time

### Algorithm

1) obvious - just basic math implementaion of formula
2) linear_access - linearazid access to b matrix( improved data localization )
3) reordered - reordered matrix B
4) blocked_1 - just blocked matrix multiplication with linearized access. Can be runned multithreaded. One block meaned to fit inside L1 cache

### Cache size and block

Assume double - 8 bite, L1 cache - 32k, then 60*60 block will fit perfectly with some space for 

### Blocked implementation

1) Boost matrix to be size % block_size = 0. Just not to calc side elements. Result matrix is just cutter matrix C.
2) Calculate parts of matrix C. Every part are localized in the cache.