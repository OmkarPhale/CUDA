// header files declaration
#include <stdio.h>

// main() definition
int main(void)
{
    // local function declaration
    void printCUDADeviceProperties(void);

    // code
    printCUDADeviceProperties();

    return (0);
}

// printCUDADeviceProperties() definition
void printCUDADeviceProperties(void)
{
    // local function declaration
    int ConvertSMVersionNumberToCores(int, int);

    // code
    printf("\n======================================================================================\n");
    printf("+ CUDA RELATED INFORMATION + \n");
    printf("======================================================================================\n");

    cudaError_t ret_cudaError_t = cudaSuccess;
    int device_count;
    ret_cudaError_t = cudaGetDeviceCount(&device_count);
    if (ret_cudaError_t != cudaSuccess)
    {
        printf("error>> CUDA Runtime API Error - cudaGetDeviceCount() Failed Due To %s. Terminating Now ...\n", cudaGetErrorString(ret_cudaError_t));
    }
    else if (device_count == 0)
    {
        printf("info>> There Is No CUDA Support Device On This System. Terminating Now ...\n");
        return;
    }
    else
    {
        printf("info>> Total Number of CUDA Supporting Device/Devices On This System : %d\n", device_count);
        for (int index = 0; index < device_count; index++)
        {
            cudaDeviceProp cuda_device_properties;
            int driverVersion = 0;
            int runtimeVersion = 0;

            ret_cudaError_t = cudaGetDeviceProperties(&cuda_device_properties, index);
            if (ret_cudaError_t != cudaSuccess)
            {
                printf("error>> CUDA Runtime API Error - cudaGetDeviceProperties() Failed in %s Due To %s at %d. Terminating Now ...\n", __FILE__, cudaGetErrorString(ret_cudaError_t), __LINE__);
                return;
            }

            printf("\n");
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);

            printf("### CUDA DRIVER AND RUNTIME INFORMATION ###\n");
            printf("======================================================================================\n");
            printf("- Driver Version                            : %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
            printf("- Runtime Version                           : %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

            printf("\n\n======================================================================================\n");
            printf("### GPU DEVICE GENERAL INFORMATION ###\n");
            printf("======================================================================================\n");
            printf("- Number                                    : %d\n", index);
            printf("- Name                                      : %s\n", cuda_device_properties.name);
            printf("- Compute Capability                        : %d.%d\n", cuda_device_properties.major, cuda_device_properties.minor);
            printf("- Clock Rate                                : %d\n", cuda_device_properties.clockRate);
            printf("- Type                                      :");
            if (cuda_device_properties.integrated)
            {
                printf(" Integrated (On-Board)\n");
            }
            else
            {
                printf(" Discrete (Card)\n");
            }
            printf("\n\n======================================================================================\n");
            printf("### GPU DEVICE MEMORY INFORMATION ###\n");
            printf("======================================================================================\n");
            printf("- Total Memory                              : %.0f GB = %.0f MB = %llu Bytes\n",
                   ((float)cuda_device_properties.totalGlobalMem / 1048576.0f) / 1024.0f,
                   ((float)cuda_device_properties.totalGlobalMem / 1048576.0f),
                   (unsigned long long)cuda_device_properties.totalGlobalMem);

            printf("- Constant Memory                           : %lu Bytes\n", (unsigned long)cuda_device_properties.totalConstMem);
            printf("- Shared Memory Per SMProcessor             : %lu\n", (unsigned long)cuda_device_properties.sharedMemPerBlock);

            printf("\n\n======================================================================================\n");
            printf("### GPU DEVICE MULTIPROCESSOR INFORMATION ###\n");
            printf("======================================================================================\n");
            printf("- Number Of SMProcessors                    : %d\n", cuda_device_properties.multiProcessorCount);
            printf("- Number Of Registers Per SMProcessor       : %d\n", cuda_device_properties.regsPerBlock);
            printf("\n");

            printf("\n\n======================================================================================\n");
            printf("### GPU DEVICE THREAD INFORMATION ###\n");
            printf("======================================================================================\n");
            printf("- Maximum Number Of Threads Per SMProcessor : %d\n", cuda_device_properties.maxThreadsPerMultiProcessor);
            printf("- Maximum Number Of Threads Per Block       : %d\n", cuda_device_properties.maxThreadsPerBlock);
            printf("- Threads In Warp                           : %d\n", cuda_device_properties.warpSize);
            printf("- Maximum Thread Dimensions                 : (%d, %d, %d)\n", cuda_device_properties.maxThreadsDim[0], cuda_device_properties.maxThreadsDim[1], cuda_device_properties.maxThreadsDim[2]);
            printf("- Maximum Grid Dimension                    : (%d, %d, %d)\n", cuda_device_properties.maxGridSize[0], cuda_device_properties.maxGridSize[1], cuda_device_properties.maxGridSize[2]);

            printf("\n\n======================================================================================\n");
            printf("### GPU DEVICE THREAD INFORMATION ###\n");
            printf("======================================================================================\n");
            printf("- Has ECC Support                           : %s\n", cuda_device_properties.ECCEnabled ? "Enabled" : "Disabled");

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
            printf("- CUDA Driver Mode (TCC or WDDM)            : %s\n", cuda_device_properties.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
            printf("======================================================================================\n");
        }
    }
}

// ConvertSMVersionNumberToCores() definition
int ConvertSMVersionNumberToCores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM)
    typedef struct
    {
        int SM; // 0xMm (hexadecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
        {
            {0x20, 32},  // Fermi Generation (SM 2.0) GF100 class
            {0x21, 48},  // Fermi Generation (SM 2.1) GF100 class
            {0x30, 192}, // Kepler Generation (SM 3.0) GF100 class
            {0x32, 192}, // Kepler Generation (SM 3.2) GF100 class
            {0x35, 192}, // Kepler Generation (SM 3.5) GF100 class
            {0x37, 192}, // Maxwell Generation (SM 3.7) GF100 class
            {0x50, 128}, // Maxwell Generation (SM 5.0) GF100 class
            {0x52, 128}, // Maxwell Generation (SM 5.2) GF100 class
            {0x53, 128}, // Maxwell Generation (SM 5.3) GF100 class
            {0x60, 64},  // Pascal Generation (SM 6.0) GF100 class
            {0x61, 128}, // Pascal Generation (SM 6.1) GF100 class
            {0x62, 128}, // Pascal Generation (SM 6.2) GF100 class
            {-1, -1}};

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor) + minor)
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we dont't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);

    return (nGpuArchCoresPerSM[index - 1].Cores);
}
