#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void vecaddK(float *A, float *B, float *C, int n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i<n) {
		C[i] = A[i] + B[i];
	}
}

void vecadd(float *A, float *B, float *C, int n){
	float *A_d, *B_d, *C_d;
	int size = n * sizeof(float);

	hipMalloc((void**) &A_d, size);
	hipMalloc((void**) &B_d, size);
	hipMalloc((void**) &C_d, size);

	hipMemcpy(A_d, A, size, hipMemcpyHostToDevice);
	hipMemcpy(B_d, B, size, hipMemcpyHostToDevice);

	vecaddK<<<ceil(n/256.0),256>>>(A_d,B_d,C_d,n);
	hipMemcpy(C, C_d, size, hipMemcpyDeviceToHost);

	hipFree(A_d);
	hipFree(B_d);
	hipFree(C_d);

}


// Forward declaration of your host wrapper that launches vecaddK
// Adjust the signature if yours is different.
void vecadd(float* A, float* B, float* C, int n);

int main() {
    // Size of the test vector
    const int N = 1 << 20;  // 1,048,576 elements

    // Host arrays
    std::vector<float> hA(N), hB(N), hC(N);

    // Initialize input data with some pattern
    for (int i = 0; i < N; ++i) {
        hA[i] = static_cast<float>(i);         // 0, 1, 2, ...
        hB[i] = static_cast<float>(2 * i);     // 0, 2, 4, ...
    }

    // Call your function that uses the HIP kernel vecaddK internally
    vecadd(hA.data(), hB.data(), hC.data(), N);

    // Verify results
    bool ok = true;
    const float eps = 1e-5f;

    for (int i = 0; i < N; ++i) {
        float expected = hA[i] + hB[i];
        float got = hC[i];
        if (std::fabs(got - expected) > eps) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << got
                      << ", expected " << expected << "\n";
            ok = false;
            break;  // Stop on first error; remove this if you want all mismatches
        }
    }

    if (ok) {
        std::cout << "vecadd test PASSED for N = " << N << "\n";
        std::cout << "Sample values:\n";
        for (int i = 0; i < 5; ++i) {
            std::cout << "  i=" << i
                      << "  A[i]=" << hA[i]
                      << "  B[i]=" << hB[i]
                      << "  C[i]=" << hC[i]
                      << " (A+B=" << (hA[i] + hB[i]) << ")\n";
        }
        return 0;
    } else {
        std::cerr << "vecadd test FAILED\n";
        return 1;
    }
}

