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

#include "QuadFmDemod.h"

#include <cuComplex.h>
#include <gsdr/gsdr.h>
#include <util/util.h>

#include "CudaErrors.h"
#include "Factories.h"

using namespace std;

using InputType = cuComplex;
using OutputType = float;

Result<Filter> QuadFmDemod::create(
    float gain,
    ICudaCommandQueue* commandQueue,
    IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  Ref<IMemSet> memSet;
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;

    auto copierFactory = factories->getCudaBufferCopierFactory();
    Ref<IBufferCopier> copier;
    UNWRAP_OR_FWD_RESULT(copier, copierFactory->createBufferCopier(commandQueue, cudaMemcpyDeviceToDevice));
    vector<ImmutableRef<IBufferCopier>> portOutputCopiers;
    UNWRAP_MOVE_OR_FWD_RESULT(portOutputCopiers, createOutputBufferCopierVector(copier.get()));

  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(commandQueue, 32, false));
  UNWRAP_OR_FWD_RESULT(memSet, factories->getCudaMemSetFactory()->create(commandQueue));

  return makeRefResultNonNull<Filter>(new (nothrow) QuadFmDemod(
      gain,
      commandQueue,
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory.get(),
      memSet.get(),
      std::move(portOutputCopiers)));
}

QuadFmDemod::QuadFmDemod(
    float gain,
    ICudaCommandQueue* commandQueue,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet,
    std::vector<ImmutableRef<IBufferCopier>>&& portOutputCopiers) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, std::move(portOutputCopiers), memSet),
      mCommandQueue(commandQueue),
      mGain(gain) {}

size_t QuadFmDemod::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return 32 * sizeof(OutputType);
}

size_t QuadFmDemod::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  Ref<IBuffer> inputBuffer;
  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(port), 0);
  const size_t numInputElements = inputBuffer->range()->used() / sizeof(InputType);
  const size_t numOutputElements = numInputElements == 0 ? 0 : numInputElements - 1;

  return numOutputElements * sizeof(OutputType);
}

Status QuadFmDemod::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");

  Ref<IBuffer> inputBuffer;
  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));
  IBuffer* outputBuffer = portOutputBuffers[0];

  const size_t availableNumInputElements = inputBuffer->range()->used() / sizeof(InputType);
  const size_t maxNumOutputElementsInBuffer = outputBuffer->range()->remaining() / sizeof(OutputType);
  const size_t maxNumOutputElementsAvailable = availableNumInputElements == 0 ? 0 : availableNumInputElements - 1;
  const size_t numOutputElements = min(maxNumOutputElementsAvailable, maxNumOutputElementsInBuffer);

  SAFE_CUDA_OR_RET_STATUS(gsdrQuadFmDemod(
      inputBuffer->readPtr<InputType>(),
      outputBuffer->writePtr<OutputType>(),
      mGain,
      numOutputElements,
      mCommandQueue->cudaDevice(),
      mCommandQueue->cudaStream()));

  const size_t writtenNumBytes = numOutputElements * sizeof(OutputType);
  const size_t discardNumInputBytes = numOutputElements * sizeof(InputType);

  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(writtenNumBytes));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, discardNumInputBytes));

  return Status_Success;
}

size_t QuadFmDemod::preferredInputBufferSize(size_t port) noexcept { return 1 << 20; }
