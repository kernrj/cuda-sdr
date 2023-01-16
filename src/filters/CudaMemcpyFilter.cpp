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

#include "CudaMemcpyFilter.h"

#include <cstring>
#include <stdexcept>
#include <string>

#include "util/CudaDevicePushPop.h"

using namespace std;

CudaMemcpyFilter::CudaMemcpyFilter(
    cudaMemcpyKind memcpyKind,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IFactories* factories)
    : BaseFilter(
        factories->getRelocatableCudaBufferFactory(cudaDevice, cudaStream, 32, /*useHostMemory=*/true),
        factories->getBufferSliceFactory(),
        1,
        factories->getCudaMemSetFactory()->create(cudaDevice, cudaStream)),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream),
      mMemCopier(factories->getCudaBufferCopierFactory()->createBufferCopier(cudaDevice, cudaStream, memcpyKind)) {}

size_t CudaMemcpyFilter::getOutputDataSize(size_t port) { return getPortInputBuffer(0)->range()->used(); }
size_t CudaMemcpyFilter::getOutputSizeAlignment(size_t port) { return 1; }

void CudaMemcpyFilter::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    throw runtime_error("One output port is required");
  }

  const shared_ptr<IBuffer> inputBuffer = getPortInputBuffer(0);
  const auto& outBuffer = portOutputs[0];
  size_t copyNumBytes = min(outBuffer->range()->remaining(), inputBuffer->range()->used());

  mMemCopier->copy(outBuffer->writePtr(), inputBuffer->readPtr(), copyNumBytes);
  outBuffer->range()->increaseEndOffset(copyNumBytes);

  consumeInputBytesAndMoveUsedToStart(0, copyNumBytes);
}
