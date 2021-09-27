import numpy as np
import subprocess



import inquirer


M = int(input("input M: M_k * k_n = M_n : "))
N = int(input("input N: m_k * k_N = m_N : "))
K = int(input("input K: m_K * K_n = m_n : "))

choices_temp = ['obvious', 'linear_access', 'blocked']

questions = [
    inquirer.List('algorithm',
                   message="Select algorithm to verify",
                   choices=choices_temp,
                 ),
]

answers = inquirer.prompt(questions)

if(choices_temp.index(answers["algorithm"]) > 1):
    Threads = int(input("Provide number of thread : "))
else:
    Threads = 1

A = np.random.rand(M, K)
B = np.random.rand(K, N)

np.savetxt("A.csv", A, delimiter=",")
np.savetxt("B.csv", B, delimiter=",")

print('\nMultiplying ' + str(M) + ' x ' + str(K) + ' * ' + str(K) + ' x ' + str(N) + ' = ' + str(M) + ' x ' +  str(N))

C_expected = np.matmul(A, B)

print("building source")

build_result = subprocess.run("cc -Werror -Wall -g -o matrix_multiply.elf matrix_multiply.c", shell=True)
if build_result.returncode != 0:
    print('build failed')
    exit()

print("Running C pthread implementation")

with open('out.txt','w+') as fout:
    with open('err.txt','w+') as ferr:
        result = subprocess.run(['./matrix_multiply.elf', str(M), str(N), str(K), "A.csv", "B.csv", "C.csv", str(choices_temp.index(answers["algorithm"])), str(Threads)], stdout=fout, stderr=ferr)
        # reset file to read from it
        fout.seek(0)
        # save output (if any) in variable
        output=fout.read()

        # reset file to read from it
        ferr.seek(0) 
        # save errors (if any) in variable
        errors = ferr.read()

if result.returncode != 0:
    print('Implementation failed with code' + str(result.returncode))
    exit()

C = np.loadtxt('C.csv', delimiter=',')

# fixing case, when array size is N*1 and numpy reads it as 1d array 
if C_expected.shape[-1] == 1: 
    C = np.array([C])
    C = np.transpose(C)
    print("answer transposed")


if np.allclose(C, C_expected, atol=1e-08):
    print("! Test passed")
else:
    print(" !!!!! TEST FAILED !!!!")

with open("out.txt", 'r') as file:
    print("Algorithm clean time is")
    print(int(file.readline()))

np.savetxt("C_expected.csv", C_expected, delimiter=",")

