// header files declaration
#include <stdio.h>

// cuda headers
#include <cuda.h>         //standard cuda header file
#include "helper_timer.h" //header for time calculation

// macros
#define BLOCK_WIDTH 32

// variable declaration
int *hostA = NULL;
int *hostB = NULL;
int *hostC = NULL;
int *gold = NULL;

int *deviceA = NULL;
int *deviceB = NULL;
int *deviceC = NULL;

float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;

// global kernel definition
__global__ void matrixMultiplyGPU(int *A, int *B, int *C, int numberOfARows, int numberOfAColumns, int numberOfBColumns, int numberOfCColumns)
{
    // local variable declaration
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int depth;

    // code
    if ((rowIndex < numberOfARows) && (colIndex < numberOfBColumns))
    {
        int value = 0.0f;
        for (depth = 0; depth < numberOfAColumns; depth++)
        {
            int a = A[rowIndex * numberOfAColumns + depth];
            int b = B[depth * numberOfBColumns + colIndex];
            value += (a * b);
        }

        C[rowIndex * numberOfCColumns + colIndex] = value;
    }
}

// main() definition
int main(int argc, char *argv[])
{
    // local function declaration
    void InitA(int *data, int row, int column);
    void InitB(int *data, int row, int column);
    void matMulCPU(int *A, int *B, int *C, int numberOfARows, int numberOfAColumns, int numberOfBColumns, int numberOfCColumns);
    void cleanup();

    // local variable declaration
    int numberOfARows = BLOCK_WIDTH;
    int numberOfAColumns = BLOCK_WIDTH;
    int numberOfBRows = BLOCK_WIDTH;
    int numberOfBColumns = BLOCK_WIDTH;

    int numberOfCRows = numberOfARows;
    int numberOfCColumns = numberOfBColumns;

    int numberOfGoldRows = numberOfARows;
    int numberOfGoldColumns = numberOfBColumns;

    cudaError_t result_cudaError_t = cudaSuccess;

    // code
    int sizeA = (numberOfARows * numberOfAColumns * sizeof(int));
    int sizeB = (numberOfBRows * numberOfBColumns * sizeof(int));
    int sizeC = (numberOfCRows * numberOfCColumns * sizeof(int));
    int sizeGold = (numberOfGoldRows * numberOfGoldColumns * sizeof(int));

    // allocate host-memory
    hostA = (int *)malloc(sizeA);
    if (hostA == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostA Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostB = (int *)malloc(sizeB);
    if (hostB == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostB Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostC = (int *)malloc(sizeC);
    if (hostC == NULL)
    {
        printf("error>> Host Memory Allocation Failed For hostC Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    gold = (int *)malloc(sizeGold);
    if (gold == NULL)
    {
        printf("error>> Host Memory Allocation Failed For gold Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // print matrix dimensions and sizes
    printf("\n==================================================================\n");
    printf("+ DISPLAYING THE MEXTRIX DIMENSIONS AND SIZES +\n");
    printf("==================================================================\n");
    printf("- The Dimensions Of Matrix 'hostA' Are : %d x %d\n", numberOfARows, numberOfAColumns);
    printf("  Size Of Matrix 'hostA'               : %d\n\n", sizeA);
    
    printf("- The Dimensions Of Matrix 'hostB' Are : %d x %d\n", numberOfBRows, numberOfBColumns);
    printf("  Size Of Matrix 'hostB'               : %d\n\n", sizeB);

    printf("- The Dimensions Of Matrix 'hostC' Are : %d x %d\n", numberOfCRows, numberOfCColumns);
    printf("  Size Of Matrix 'hostC'               : %d\n\n", sizeC);

    printf("- The Dimensions Of Matrix 'gold' Are  : %d x %d\n", numberOfGoldRows, numberOfGoldColumns);
    printf("  Size Of Matrix 'gold'               : %d\n\n", sizeGold);

    // fill source matrices
    InitA(hostA, numberOfARows, numberOfAColumns);
    InitA(hostB, numberOfBRows, numberOfBColumns);

    // allocate device memory
    result_cudaError_t = cudaMalloc((void **)&deviceA, sizeA);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device Memory Allocation Failed For deviceA Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result_cudaError_t = cudaMalloc((void **)&deviceB, sizeB);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device Memory Allocation Failed For deviceB Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result_cudaError_t = cudaMalloc((void **)&deviceC, sizeC);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device Memory Allocation Failed For deviceC  Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // copy data from host matrices into device matrices
    result_cudaError_t = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Host To Device Data Copy Failed For deviceA Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    result_cudaError_t = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Host To Device Data Copy Failed For deviceB Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // CUDA kernel configuration
    dim3 dimGrid = dim3(ceil((int)numberOfCColumns / (int)BLOCK_WIDTH), ceil((int)numberOfCRows / (int)BLOCK_WIDTH), 1);
    dim3 dimBlock = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    matrixMultiplyGPU<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numberOfARows, numberOfAColumns, numberOfBColumns, numberOfCColumns);

    // stop timer
    sdkStopTimer(&timer);
    timeOnGPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;

    // copy device memory to  host memory
    result_cudaError_t = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
    if (result_cudaError_t != cudaSuccess)
    {
        printf("error>> Device To Host Data Copy Failed For hostC Matrix. Terminating Now...\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // results
    matMulCPU(hostA, hostB, gold, numberOfARows, numberOfAColumns, numberOfBColumns, numberOfGoldColumns);

    // compare results for golden = host
    int breakValue = -1;
    bool bAccuracy = true;
    int index;
    for (index = 0; index < numberOfARows; index++)
    {
        int val1 = gold[index];
        int val2 = hostC[index];
        if (val1 != val2)
        {
            bAccuracy = false;
            breakValue = index;
            break;
        }
    }

    char stringMessage[125];
    if (bAccuracy == false)
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Matrix Multiplication Is Not Accurate At Array Index %d", breakValue);
    }
    else
    {
        sprintf(stringMessage, "%s", "# Comparison Of CPU And GPU Matrix Multiplication Is Accurate.");
    }

    printf("\n==================================================================\n");
    printf("+ DISPLAYING THE RESULT OF ADDITION FROM DEVICE TO HOST +\n");
    printf("==================================================================\n");

    printf("- Time Taken For Matrix Multiplication On CPU = %.6f (ms)\n", timeOnCPU);
    printf("- Time Taken For Matrix Multiplication On GPU = %.6f (ms)\n", timeOnGPU);
    printf("%s\n", stringMessage);
    printf("==================================================================\n");

    // total cleanup
    cleanup();

    return (0);
}

// InitA() definition
void InitA(int *data, int row, int column)
{
    // local variable declaration
    int number = 1;
    int rowIndex;
    int columnIndex;

    // code
    for (rowIndex = 0; rowIndex < row; rowIndex++)
    {
        for (columnIndex = 0; columnIndex < column; columnIndex++)
        {
            *(data + rowIndex * column + columnIndex) = number;
            number++;
        }
    }
}

// InitB() definition
void InitB(int *data, int row, int column)
{
    // local variable declaration
    int number = BLOCK_WIDTH;

    // code
    for (int rowIndex = 0; rowIndex < row; rowIndex++)
    {
        for (int columnIndex = 0; columnIndex < column; columnIndex++)
        {
            *(data + rowIndex * column + columnIndex) = number;
            number--;
        }
    }
}

// matMulCPU() definition
void matMulCPU(int *A, int *B, int *C, int numberOfARows, int numberOfAColumns, int numberOfBColumns, int numberOfCColumns)
{
    // code
    // start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    for (int row = 0; row < numberOfARows; row++)
    {
        for (int column = 0; column < numberOfBColumns; column++)
        {
            int value = 0.0f;
            for (int depth = 0; depth < numberOfAColumns; depth++)
            {
                int a = A[row * numberOfAColumns + depth];
                int b = B[depth * numberOfBColumns + column];

                value += a * b;
            }
            C[row * numberOfCColumns + column] = value;
        }
    }

    // stop timer
    sdkStopTimer(&timer);
    timeOnCPU = sdkGetTimerValue(&timer);
    sdkDeleteTimer(&timer);
    timer = NULL;
}

// cleanup() definition
void cleanup(void)
{
    // code
    if (deviceC)
    {
        cudaFree(deviceC);
        deviceC = NULL;
    }

    if (deviceB)
    {
        cudaFree(deviceB);
        deviceB = NULL;
    }

    if (deviceA)
    {
        cudaFree(deviceA);
        deviceA = NULL;
    }

    if (gold)
    {
        free(gold);
        gold = NULL;
    }

    if (hostC)
    {
        free(hostC);
        hostC = NULL;
    }

    if (hostB)
    {
        free(hostB);
        hostB = NULL;
    }

    if (hostA)
    {
        free(hostA);
        hostA = NULL;
    }
}
