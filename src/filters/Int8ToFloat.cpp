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

#include "Int8ToFloat.h"

#include <cuda_runtime_api.h>
#include <gsdr/conversion.h>
#include <util/util.h>

#include <cstdint>

#include "CudaErrors.h"
#include "Factories.h"

using namespace std;

const size_t Int8ToFloat::mAlignment = 32;

Result<Filter> Int8ToFloat::create(ICudaCommandQueue* commandQueue, IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  Ref<IMemSet> memSet;
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;

  auto copierFactory = factories->getCudaBufferCopierFactory();
  Ref<IBufferCopier> copier;
  UNWRAP_OR_FWD_RESULT(copier, copierFactory->createBufferCopier(commandQueue, cudaMemcpyDeviceToDevice));
  vector<ImmutableRef<IBufferCopier>> outputBufferCopiers;
  UNWRAP_MOVE_OR_FWD_RESULT(outputBufferCopiers, createOutputBufferCopierVector(copier.get()));

  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(commandQueue, mAlignment, false));
  UNWRAP_OR_FWD_RESULT(memSet, factories->getCudaMemSetFactory()->create(commandQueue));

  return makeRefResultNonNull<Filter>(new (nothrow) Int8ToFloat(
      commandQueue,
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory,
      memSet.get(),
      std::move(outputBufferCopiers)));
}

Int8ToFloat::Int8ToFloat(
    ICudaCommandQueue* commandQueue,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet,
    vector<ImmutableRef<IBufferCopier>>&& portOutputCopiers) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, std::move(portOutputCopiers), memSet),
      mCommandQueue(commandQueue) {}

size_t Int8ToFloat::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Port [%zu] is out of range", port);

  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(0), 0);

  return inputBuffer->range()->used() * sizeof(float) / sizeof(int8_t);
}

size_t Int8ToFloat::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mAlignment;
}

Status Int8ToFloat::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(0 == portCount, "One output port is required");

  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));
  const auto& outputBuffer = portOutputBuffers[0];

  const size_t elementCount = min(inputBuffer->range()->used(), outputBuffer->range()->remaining() / sizeof(float));

  SAFE_CUDA_OR_RET_STATUS(gsdrInt8ToNormFloat(
      inputBuffer->readPtr<int8_t>(),
      outputBuffer->writePtr<float>(),
      elementCount,
      mCommandQueue->cudaDevice(),
      mCommandQueue->cudaStream()));

  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(elementCount * sizeof(float)));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, elementCount * sizeof(int8_t)));

  return Status_Success;
}

size_t Int8ToFloat::preferredInputBufferSize(size_t port) noexcept { return 1 << 20; }
