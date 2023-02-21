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

#include "QuadAmDemod.h"

#include <cuComplex.h>
#include <gsdr/gsdr.h>

#include "CudaErrors.h"
#include "Factories.h"

using namespace std;

using InputType = cuComplex;
using OutputType = float;

Result<Filter> QuadAmDemod::create(int32_t cudaDevice, cudaStream_t cudaStream, IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  ConstRef<IMemSet> memSet = factories->getSysMemSet();
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;

  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(cudaDevice, cudaStream, 32, false));

  return makeRefResultNonNull<Filter>(new (nothrow) QuadAmDemod(
      cudaDevice,
      cudaStream,
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory.get(),
      memSet.get()));
}

QuadAmDemod::QuadAmDemod(
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, memSet),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t QuadAmDemod::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return 32 * sizeof(OutputType);
}

size_t QuadAmDemod::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(port), 0);
  const size_t numInputElements = inputBuffer->range()->used() / sizeof(InputType);
  const size_t numOutputElements = numInputElements;

  return numOutputElements * sizeof(OutputType);
}

Status QuadAmDemod::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");

  Ref<IBuffer> inputBuffer;
  IBuffer* outputBuffer = portOutputBuffers[0];

  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));

  const size_t availableNumInputElements = inputBuffer->range()->used() / sizeof(InputType);
  const size_t maxNumOutputElementsInBuffer = outputBuffer->range()->remaining() / sizeof(OutputType);
  const size_t maxNumOutputElementsAvailable = availableNumInputElements;
  const size_t numOutputElements = min(maxNumOutputElementsAvailable, maxNumOutputElementsInBuffer);

  SAFE_CUDA_OR_RET_STATUS(gsdrQuadAmDemod(
      inputBuffer->readPtr<InputType>(),
      outputBuffer->writePtr<OutputType>(),
      numOutputElements,
      mCudaDevice,
      mCudaStream));

  const size_t writtenNumBytes = numOutputElements * sizeof(OutputType);
  const size_t discardNumInputBytes = numOutputElements * sizeof(InputType);

  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(writtenNumBytes));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, discardNumInputBytes));

  return Status_Success;
}
