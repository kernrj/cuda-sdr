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

#include "AddConst.h"

#include <gsdr/gsdr.h>

#include "CudaErrors.h"

using namespace std;

const size_t AddConst::mAlignment = 32;

Result<Filter> AddConst::create(
    float addValueToMagnitude,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IFactories* factories) noexcept {
  Ref<IRelocatableResizableBufferFactory> relocatableCudaBufferFactory;
  ConstRef<IBufferSliceFactory> bufferSliceFactory = factories->getBufferSliceFactory();
  ConstRef<IMemSet> memSet = factories->getSysMemSet();
  Ref<IRelocatableResizableBufferFactory> relocatableResizableBufferFactory;

  UNWRAP_OR_FWD_RESULT(
      relocatableCudaBufferFactory,
      factories->createRelocatableCudaBufferFactory(cudaDevice, cudaStream, mAlignment, false));

  return makeRefResultNonNull<Filter>(new (nothrow) AddConst(
      addValueToMagnitude,
      cudaDevice,
      cudaStream,
      relocatableCudaBufferFactory.get(),
      bufferSliceFactory.get(),
      memSet.get()));
}

AddConst::AddConst(
    float addConst,
    int32_t cudaDevice,
    cudaStream_t cudaStream,
    IRelocatableResizableBufferFactory* relocatableBufferFactory,
    IBufferSliceFactory* bufferSliceFactory,
    IMemSet* memSet) noexcept
    : BaseFilter(relocatableBufferFactory, bufferSliceFactory, 1, memSet),
      mAddConst(addConst),
      mCudaDevice(cudaDevice),
      mCudaStream(cudaStream) {}

size_t AddConst::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return getAvailableNumInputElements() * sizeof(float);
}

size_t AddConst::getAvailableNumInputElements() const noexcept {
  Ref<const IBuffer> inputBuffer;
  UNWRAP_OR_RETURN(inputBuffer, getPortInputBuffer(0), 0);

  return inputBuffer->range()->used() / sizeof(float);
}

size_t AddConst::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return mAlignment * sizeof(float);
}

Status AddConst::readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept {
  if (numPorts == 0) {
    gslog(GSLOG_ERROR, "One output port is required");
    return Status_InvalidArgument;
  }

  Ref<IBuffer> inputBuffer;
  UNWRAP_OR_FWD_STATUS(inputBuffer, getPortInputBuffer(0));
  const auto outputBuffer = portOutputBuffers[0];

  const size_t numInputElements = getAvailableNumInputElements();
  const size_t maxNumOutputElements = outputBuffer->range()->remaining() / sizeof(float);
  const size_t processNumElements = min(numInputElements, maxNumOutputElements);

  SAFE_CUDA_OR_RET_STATUS(gsdrAddConstFF(
      inputBuffer->readPtr<float>(),
      mAddConst,
      outputBuffer->writePtr<float>(),
      processNumElements,
      mCudaDevice,
      mCudaStream));

  const size_t writtenNumBytes = processNumElements * sizeof(float);
  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(writtenNumBytes));
  FWD_IF_ERR(consumeInputBytesAndMoveUsedToStart(0, writtenNumBytes));

  return Status_Success;
}
