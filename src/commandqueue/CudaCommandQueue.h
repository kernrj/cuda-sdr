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

#ifndef GPUSDRPIPELINE_CUDACOMMANDQUEUE_H
#define GPUSDRPIPELINE_CUDACOMMANDQUEUE_H

#include "Result.h"
#include "commandqueue/ICudaCommandQueue.h"

class CudaCommandQueue final : public ICudaCommandQueue {
 public:
  static Result<ICudaCommandQueue> create(int32_t device, cudaStream_t stream);

  int32_t cudaDevice() const noexcept final;
  cudaStream_t cudaStream() const noexcept final;

 private:
  const int32_t mDevice;
  cudaStream_t mStream;

 private:
  CudaCommandQueue(int32_t device, cudaStream_t stream) noexcept : mDevice(device), mStream(stream) {}

  REF_COUNTED(CudaCommandQueue);
};

#endif  // GPUSDRPIPELINE_CUDACOMMANDQUEUE_H
