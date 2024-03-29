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

#include <util/util.h>

#include <cstring>
#include <string>

#include "util/CudaDevicePushPop.h"

using namespace std;

static bool isInputHostMemory(cudaMemcpyKind memcpyKind) noexcept {
  return memcpyKind == cudaMemcpyHostToDevice || memcpyKind == cudaMemcpyHostToHost;
}

Result<Filter> CudaMemcpyFilter::create(
    cudaMemcpyKind memcpyKind,
    ICudaCommandQueue* commandQueue,
    IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  Ref<IMemSet> memSet;
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;
  Ref<IBufferCopier> cudaCopier;

  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(
          commandQueue,
          32,
          /*useHostMemory=*/isInputHostMemory(memcpyKind)));
  UNWRAP_OR_FWD_RESULT(memSet, factories->getCudaMemSetFactory()->create(commandQueue));
  UNWRAP_OR_FWD_RESULT(
      cudaCopier,
      factories->getCudaBufferCopierFactory()->createBufferCopier(commandQueue, memcpyKind));

  auto copierFactory = factories->getCudaBufferCopierFactory();
  Ref<IBufferCopier> copier;
  UNWRAP_OR_FWD_RESULT(copier, copierFactory->createBufferCopier(commandQueue, cudaMemcpyDeviceToDevice));
  vector<ImmutableRef<IBufferCopier>> outputBufferCopiers;
  UNWRAP_MOVE_OR_FWD_RESULT(outputBufferCopiers, createOutputBufferCopierVector(copier.get()));

  return makeRefResultNonNull<Filter>(new (nothrow) CudaMemcpyFilter(
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory.get(),
      memSet.get(),
      cudaCopier.get(),
      std::move(outputBufferCopiers)));
}

CudaMemcpyFilter::CudaMemcpyFilter(
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet,
    IBufferCopier* cudaCopier,
    std::vector<ImmutableRef<IBufferCopier>>&& portOutputCopiers) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, std::move(portOutputCopiers), memSet),
      mMemCopier(cudaCopier) {}

size_t CudaMemcpyFilter::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);

  Ref<IBuffer> inputBuffer;
  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(0), 0);

  return inputBuffer->range()->used();
}

size_t CudaMemcpyFilter::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return 1;
}

Status CudaMemcpyFilter::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");

  Ref<IBuffer> inputBuffer;
  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));
  const auto& outBuffer = portOutputBuffers[0];
  size_t copyNumBytes = min(outBuffer->range()->remaining(), inputBuffer->range()->used());

  FWD_IF_ERR(mMemCopier->copy(outBuffer->writePtr(), inputBuffer->readPtr(), copyNumBytes));
  FWD_IF_ERR(outBuffer->range()->increaseEndOffset(copyNumBytes));

  return consumeInputBytesAndMoveUsedToStart(0, copyNumBytes);
}

size_t CudaMemcpyFilter::preferredInputBufferSize(size_t port) noexcept { return 1 << 20; }
