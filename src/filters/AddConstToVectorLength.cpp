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

#include "AddConstToVectorLength.h"

#include <cuComplex.h>
#include <cuda.h>
#include <gsdr/gsdr.h>
#include <util/util.h>

#include "CudaErrors.h"

using namespace std;

const size_t AddConstToVectorLength::mAlignment = 32;

Result<Filter> AddConstToVectorLength::create(
    float addValueToMagnitude,
    ICudaCommandQueue* commandQueue,
    IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  Ref<IMemSet> memSet;
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;

  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(commandQueue, mAlignment, false));
  UNWRAP_OR_FWD_RESULT(memSet, factories->getCudaMemSetFactory()->create(commandQueue));

  auto copierFactory = factories->getCudaBufferCopierFactory();
  Ref<IBufferCopier> copier;
  UNWRAP_OR_FWD_RESULT(copier, copierFactory->createBufferCopier(commandQueue, cudaMemcpyDeviceToDevice));
  vector<ImmutableRef<IBufferCopier>> outputBufferCopiers;
  UNWRAP_MOVE_OR_FWD_RESULT(outputBufferCopiers, createOutputBufferCopierVector(copier.get()));

  return makeRefResultNonNull<Filter>(new (nothrow) AddConstToVectorLength(
      addValueToMagnitude,
      commandQueue,
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory,
      memSet.get(),
      std::move(outputBufferCopiers)));
}

AddConstToVectorLength::AddConstToVectorLength(
    float addValueToMagnitude,
    ICudaCommandQueue* commandQueue,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet,
    vector<ImmutableRef<IBufferCopier>>&& portOutputCopiers) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, std::move(portOutputCopiers), memSet),
      mAddValueToMagnitude(addValueToMagnitude),
      mCommandQueue(commandQueue) {}

size_t AddConstToVectorLength::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return getAvailableNumInputElements() * sizeof(cuComplex);
}

size_t AddConstToVectorLength::getAvailableNumInputElements() const noexcept {
  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(0), 0);

  return inputBuffer->range()->used() / sizeof(cuComplex);
}

size_t AddConstToVectorLength::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mAlignment * sizeof(cuComplex);
}

Status AddConstToVectorLength::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");
  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));

  const size_t numInputElements = getAvailableNumInputElements();
  const auto& outputBuffer = portOutputBuffers[0];
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(cuComplex);
  const size_t processNumElements = min(maxNumOutputElements, numInputElements);

  SAFE_CUDA_OR_RET_STATUS(gsdrAddToMagnitude(
      inputBuffer->readPtr<cuComplex>(),
      mAddValueToMagnitude,
      outputBuffer->writePtr<cuComplex>(),
      processNumElements,
      mCommandQueue->cudaDevice(),
      mCommandQueue->cudaStream()));

  const size_t writtenNumBytes = processNumElements * sizeof(cuComplex);

  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(writtenNumBytes));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, writtenNumBytes));

  return Status_Success;
}

size_t AddConstToVectorLength::preferredInputBufferSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(port == 0, Status_InvalidArgument, "Max port is 0, but port [%zu] was requested", port);
  return (1 << 20) * sizeof(cuComplex);
}
