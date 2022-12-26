/*
 * Copyright 2022 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SDRTEST_SRC_CUDA_UTIL_H_
#define SDRTEST_SRC_CUDA_UTIL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>

/*
 * CLion shows a syntax error at the '<<<>>>' when invoking a kernel because
 * it doesn't have a declaration for this function. The error is cosmetic
 * because it compiles fine, but it's distracting.
 */
extern cudaError_t cudaConfigureCall(
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem = 0,
    cudaStream_t stream = nullptr);

#define SAFE_CUDA(__cmd)                                                 \
  do {                                                                   \
    cudaError_t __status = (__cmd);                                      \
    if (__status != cudaSuccess) {                                       \
      throw std::runtime_error(                                          \
          std::string("CUDA error ") + cudaGetErrorName(__status) + ": " \
          + cudaGetErrorString(__status) + ". At " + __FILE__ + ":"      \
          + std::to_string(__LINE__));                                   \
    }                                                                    \
  } while (false)

inline int32_t getCurrentCudaDevice() {
  int32_t device = -1;
  SAFE_CUDA(cudaGetDevice(&device));

  return device;
}

#endif  // SDRTEST_SRC_CUDA_UTIL_H_
