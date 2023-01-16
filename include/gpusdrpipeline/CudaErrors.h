//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_CUDAERRORS_H
#define GPUSDRPIPELINE_CUDAERRORS_H

#include <cuda_runtime.h>
#include <gpusdrpipeline/GSErrors.h>

#ifdef DEBUG
#define GPU_SYNC_DEBUG cudaDeviceSynchronize()
#else
#define GPU_SYNC_DEBUG cudaSuccess
#endif

#ifdef DEBUG
#define CHECK_CUDA(msgOnFail__)                                                                                 \
  do {                                                                                                          \
    cudaError_t checkCudaStatus__ = cudaDeviceSynchronize();                                                    \
    if (checkCudaStatus__ != cudaSuccess) {                                                                     \
      THROW(                                                                                                    \
          (msgOnFail__) << ": " << cudaGetErrorName(checkCudaStatus__) << " (" << checkCudaStatus__ << "). At " \
                        << __FILE__ << ':' << __LINE__);                                                        \
    }                                                                                                           \
  } while (false)
#else
#define CHECK_CUDA(__msg) (void)0
#endif

#define SAFE_CUDA(cudaCmd__)                                                                                  \
  do {                                                                                                        \
    CHECK_CUDA("Before: " #cudaCmd__);                                                                        \
    cudaError_t safeCudaStatus__ = (cudaCmd__);                                                               \
    if (safeCudaStatus__ != cudaSuccess) {                                                                    \
      THROW(                                                                                                  \
          "CUDA error " << cudaGetErrorName(safeCudaStatus__) << ": " << cudaGetErrorString(safeCudaStatus__) \
                        << ". At " << __FILE__ << ':' << __LINE__);                                           \
    }                                                                                                         \
    CHECK_CUDA("After: " #cudaCmd__);                                                                         \
  } while (false)

#endif  // GPUSDRPIPELINE_CUDAERRORS_H
