/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
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

#ifndef GPUSDRPIPELINE_WAITER_H
#define GPUSDRPIPELINE_WAITER_H

#include <cuda_runtime.h>

#include <cstdint>

#include "CudaErrors.h"
#include "Status.h"
#include "util/CudaDevicePushPop.h"

/**
 * Keeps a CUDA stream full, only waiting on the previous iteration of work.
 *
 * For example:
 * Waiter waiter(mCudaDevice, mCudaStream);
 *
 * queue.push(doCudaWork());
 * waiter.recordNextAndWaitPrevious();
 *
 * while (doWork) {
 *   queue.push(doCudaWork());
 *   waiter.recordNextAndWaitPrevious();
 *   finishedWork = queue.front();
 *   queue.pop();
 * }
 */
class Waiter {
 public:
  Waiter(int32_t cudaDevice, cudaStream_t cudaStream) noexcept;
  ~Waiter();

  [[nodiscard]] Status recordNextAndWaitPrevious() noexcept;

 private:
  const int32_t mCudaDevice;
  cudaStream_t mCudaStream;
  cudaEvent_t mCudaEvent;
  cudaEvent_t mNextCudaEvent;
};

#endif  // GPUSDRPIPELINE_WAITER_H
