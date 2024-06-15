// header files declaration
#include <stdio.h>
#include <cuda.h> //standard cuda header file

// global variables
const int iNumberOfArrayElements = 5;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;

float *deviceInput1 = NULL;
float *deviceInput2 = NULL;
float *deviceOutput = NULL;

// global CUDA kernel function definition
__global__ void vecAddGPU(float *input1, float *input2, float *output, int length)
{
    // local variable declaration
    // row * width + column
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // code
    if (index < length)
    {
        output[index] = input1[index] + input2[index];
    }
}

// main() definition
int main(void)
{
    // local function declaration
    void cleanup(void);

    // code
    // host memory allocation
    hostInput1 = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if (hostInput1 == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostInput1 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if (hostInput2 == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostInput2 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float *)malloc(iNumberOfArrayElements * sizeof(float));
    if (hostOutput == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostOutput Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // filling values input host arrays
    hostInput1[0] = 1001.0f;
    hostInput1[1] = 1002.0f;
    hostInput1[2] = 1003.0f;
    hostInput1[3] = 1004.0f;
    hostInput1[4] = 1005.0f;

    hostInput2[0] = 2001.0f;
    hostInput2[1] = 2002.0f;
    hostInput2[2] = 2003.0f;
    hostInput2[3] = 2004.0f;
    hostInput2[4] = 2005.0f;

    // device memory allocation
    int size = iNumberOfArrayElements * sizeof(float);
    cudaError_t result_cudaError_t = cudaSuccess;
    result_cudaError_t = cudaMalloc((void **)&deviceInput1, size);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device Memory Allocation Failed For deviceInput1 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result_cudaError_t = cudaMalloc((void **)&deviceInput2, size);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device Memory Allocation Failed For deviceInput2 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result_cudaError_t = cudaMalloc((void **)&deviceOutput, size);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device Memory Allocation Failed For deviceOutput Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // copy data from host memory into device memory
    result_cudaError_t = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Host To Device Data Copy Failed For deviceInput1 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result_cudaError_t = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Host To Device Data Copy Failed For deviceInput2 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    dim3 dimGrid = dim3(iNumberOfArrayElements, 1, 1);
    dim3 dimBlock = dim3(1, 1, 1);

    // CUDA kernel for vector addtion
    vecAddGPU<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

    // copy data from device array into host array
    result_cudaError_t = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device To Host Data Copy Failed For hostOutput Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // vector addition on host
    printf("\n==================================================================\n");
    printf("+ DISPLAYING THE RESULT OF ADDITION FROM DEVICE TO HOST +\n");
    printf("==================================================================\n");

    for (int index = 0; index < iNumberOfArrayElements; index++)
    {
        printf("- Array Index '%d' >> %f + %f = %f\n", index, hostInput1[index], hostInput2[index], hostOutput[index]);
    }
    printf("==================================================================\n");

    // cleanup
    cleanup();

    return (0);
}

// cleanup() definition
void cleanup(void)
{
    // code
    if (deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput = NULL;
    }

    if (deviceInput2)
    {
        cudaFree(deviceInput2);
        deviceInput2 = NULL;
    }

    if (deviceInput1)
    {
        cudaFree(deviceInput1);
        deviceInput1 = NULL;
    }

    if (hostOutput)
    {
        free(hostOutput);
        hostOutput = NULL;
    }

    if (hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if (hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }
}
