/*
 * Copyright 2022-2023 Rick Kern <kernrj@gmail.com>
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

#ifndef SDRTEST_SRC_CUDADEVICEPUSHPOP_H_
#define SDRTEST_SRC_CUDADEVICEPUSHPOP_H_

#include <cuda.h>
#include <gpusdrpipeline/CudaErrors.h>
#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/util/CudaUtil.h>

#include <cstdint>

class CudaDevicePushPop {
 public:
  CudaDevicePushPop(int32_t previousCudaDevice, bool hasPrevious) noexcept
      : mPreviousCudaDevice(previousCudaDevice),
        mIsSet(hasPrevious) {}

  CudaDevicePushPop() noexcept
      : mPreviousCudaDevice(0),
        mIsSet(false) {}

  ~CudaDevicePushPop() {
    if (mIsSet) {
      cudaSetDevice(mPreviousCudaDevice);
    }
  }

 private:
  const int32_t mPreviousCudaDevice;
  const bool mIsSet;
};

[[nodiscard]] inline Result<CudaDevicePushPop> pushCudaDevice(int32_t cudaDevice) noexcept {
  int32_t previousDevice;
  UNWRAP_OR_FWD_RESULT(previousDevice, gsGetCurrentCudaDevice());
  SAFE_CUDA_OR_RET_RESULT(cudaSetDevice(cudaDevice));

  return makeValResult(CudaDevicePushPop(previousDevice, /* hasPrevious= */ true));
}

#define CUDA_DEV_PUSH_POP_OR_RET(deviceIndex__, returnOnFailure__)           \
  Result<CudaDevicePushPop> pushPopResult__ = pushCudaDevice(deviceIndex__); \
  if (pushPopResult__.status != Status_Success) {                            \
    return returnOnFailure__;                                                \
  }

#define CUDA_DEV_PUSH_POP_OR_RET_STATUS(deviceIndex__)                       \
  Result<CudaDevicePushPop> pushPopResult__ = pushCudaDevice(deviceIndex__); \
  if (pushPopResult__.status != Status_Success) {                            \
    return pushPopResult__.status;                                           \
  }

#define CUDA_DEV_PUSH_POP_OR_RET_RESULT(deviceIndex__)                       \
  Result<CudaDevicePushPop> pushPopResult__ = pushCudaDevice(deviceIndex__); \
  if (pushPopResult__.status != Status_Success) {                            \
    return {.status = pushPopResult__.status, .value = {}};                  \
  }

#define CUDA_DEV_PUSH_POP_OR_THROW(deviceIndex__)                            \
  Result<CudaDevicePushPop> pushPopResult__ = pushCudaDevice(deviceIndex__); \
  if (pushPopResult__.status != Status_Success) {                            \
    gsloge("Failed to set CUDA device to [%d]", deviceIndex__);              \
    throw std::runtime_error("Failed to set CUDA device.");                  \
  }

#endif  // SDRTEST_SRC_CUDADEVICEPUSHPOP_H_
