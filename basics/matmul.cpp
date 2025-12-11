#include <hip/hip_runtime.h>
#include <iostream>
#include <string>

/*
P[r,c] = sum M[r,k] N[k,c]
*/

/*
let's simplify
M a b
N b c

P a c
*/

__device__ __inline__ int indexify(int m, int n, int row_len){
    return m*row_len + n;
}

__global__ void matmulK(float *P, const float *M, const float *N, int a, int b, int c){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < a && col < c){
        // access M[m,n] as M[m*row_len + n]
        float sum = 0;
        for (int k=0; k<b; ++k){
            sum += M[indexify(row,k,b)] * N[indexify(k,col,c)];
        }
        P[indexify(row,col,c)] = sum;
    }
}

// col_len M, row_len_M == col_len_N, row_len_N

void mmwrap(float *P, const float *M, const float *N, int a, int b, int c){
    float *M_d, *N_d, *P_d;
    std::size_t size_M = (std::size_t)(a)*b*sizeof(float);
    std::size_t size_N = (std::size_t)(b)*c*sizeof(float);
    std::size_t size_P = (std::size_t)(a)*c*sizeof(float);


    dim3 dimGrid(ceil(c/16.0),ceil(a/16.0),1);
    dim3 dimBlock(16,16,1);

    hipMalloc((void**)&M_d,size_M);
    hipMalloc((void**)&P_d,size_P);
    hipMalloc((void**)&N_d,size_N);

    hipMemcpy(M_d, M, size_M, hipMemcpyHostToDevice);
    hipMemcpy(N_d, N, size_N, hipMemcpyHostToDevice);

    matmulK<<<dimGrid,dimBlock>>>(P_d,M_d,N_d,a,b,c);
    hipMemcpy(P, P_d, size_P, hipMemcpyDeviceToHost);
    
    hipFree(P_d);
    hipFree(M_d);
    hipFree(N_d);

}