// header files declaration
#include <stdio.h>

// cuda headers
#include <cuda.h>         //standard cuda header file
#include "helper_timer.h" //header for time calculation

// global variables
// const int iNumberOfArrayElements = 5;
const int iNumberOfArrayElements = 11444777;

float *hostInput1 = NULL;
float *hostInput2 = NULL;
float *hostOutput = NULL;
float *gold = NULL;

float *deviceInput1 = NULL;
float *deviceInput2 = NULL;
float *deviceOutput = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

// global CUDA kernel function definition
__global__ void vecAddGPU(float *input1, float *input2, float *output, int length)
{
    // local variable declaration
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // code
    if (index < length)
    {
        output[index] = input1[index] + input2[index];
    }
}

// main() definition
int main(int argc, char *argv[])
{
    // local function declaration
    void fillFloatArrayWithRandomNumbers(float *pFloatArray, int iSize);
    void vecAddCPU(const float *input1, const float *input2, float *output, int length);
    void cleanup();

    // local variable declaration
    int size = (iNumberOfArrayElements * sizeof(float));
    cudaError_t result_cudaError_t = cudaSuccess;

    // code
    // host memory allocation
    hostInput1 = (float *)malloc(size);
    if (hostInput1 == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostInput1 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float *)malloc(size);
    if (hostInput2 == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostInput2 Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float *)malloc(size);
    if (hostOutput == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostOutput Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (float *)malloc(size);
    if (gold == NULL)
    {
        printf("error>> Host Memory Allocation Is Failed For Gold Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // filling values into the host arrays
    fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
    fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

    // device memory allocation
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

    // copy data from host arrays into device arrays
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

    // CUDA kernel configuration
    dim3 dimGrid = dim3((int)ceil((float)iNumberOfArrayElements / 256.0f), 1, 1);
    dim3 dimBlock = dim3(256, 1, 1);

    // CUDA kernel for vector addtion
    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    vecAddGPU<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

    // stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    // copy data from device array into host array
    result_cudaError_t = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device To Host Data Copy Failed For hostOutput Array. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // vector addition on host
    vecAddCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements);

    // comparison
    const float epsilon = 0.000001f;
    int breakValue = 0;
    bool bAccuracy = true;
    int index;

    for (index = 0; index < iNumberOfArrayElements; index++)
    {
        float value1 = gold[index];
        float value2 = hostOutput[index];
        if (fabs(value1 - value2) > epsilon)
        {
            bAccuracy = false;
            breakValue = index;
            break;
        }
    }

    char stringMessage[125];
    if (bAccuracy == false)
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Vector Addition Is Not Within Accuracy Of 0.000001 At Array Index %d", breakValue);
    }
    else
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Vector Addition Is Within Accuracy Of 0.000001");
    }

    // output
    printf("\n==================================================================================\n");
    printf("+ DISPLAYING THE RESULT OF ADDITION FROM DEVICE TO HOST +\n");
    printf("==================================================================================\n");
    printf("- Array1 Begins From 0th Element Index %.6f to %dth Index %.6f\n", hostInput1[0], (iNumberOfArrayElements - 1), hostInput1[iNumberOfArrayElements - 1]);
    printf("- Array2 Begins From 0th Element Index %.6f to %dth Index %.6f\n\n", hostInput2[0], (iNumberOfArrayElements - 1), hostInput2[iNumberOfArrayElements - 1]);

    printf("- CUDA Kernel Grid Dimension = (%d, %d, %d) And Block Dimension = (%d, %d, %d)\n\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    printf("- Output Begins From 0th Element Index %.6f to %dth Index %.6f\n\n", hostOutput[0], (iNumberOfArrayElements - 1), hostOutput[iNumberOfArrayElements - 1]);

    printf("- The Time Taken To Do Above Addition On CPU = %.6f (ms)\n", timeOnCPU);
    printf("- The Time Taken To Do Above Addition On GPU = %.6f (ms)\n", timeOnGPU);
    printf("- %s\n", stringMessage);
    printf("==================================================================================\n");

    // cleanup
    cleanup();

    return (0);
}

// fillFloatArrayWithRandomNumbers() definition
void fillFloatArrayWithRandomNumbers(float *pfArray, int length)
{
    // local variable declaration
    int index;
    const float fScale = (1.0f / (float)RAND_MAX);

    // code
    for (index = 0; index < length; index++)
    {
        pfArray[index] = (fScale * rand());
    }
}

// vecAddCPU() definition
void vecAddCPU(const float *pfArray1, const float *pfArray2, float *pfOutput, int length)
{
    // local variable declaration
    int index;

    // code
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (index = 0; index < length; index++)
    {
        pfOutput[index] = pfArray1[index] + pfArray2[index];
    }

    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;
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

    if (gold)
    {
        free(gold);
        gold = NULL;
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
