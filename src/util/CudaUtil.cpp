//
// Created by Rick Kern on 1/8/23.
//

#include "util/CudaUtil.h"

#include <cuda_runtime.h>

#include "CudaErrors.h"

int32_t getCurrentCudaDeviceOrThrow() {
  int32_t device = -1;
  SAFE_CUDA(cudaGetDevice(&device));

  return device;
}
