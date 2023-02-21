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

#include "filters/BaseSink.h"

#include <stdexcept>

#include "GSErrors.h"

using namespace std;

BaseSink::BaseSink(
    IRelocatableResizableBufferFactory* relocatableResizableBufferFactory,
    IBufferSliceFactory* slicedBufferFactory,
    size_t inputPortCount,
    IMemSet* memSet)
    : mInputPortCount(inputPortCount),
      mSlicedBufferFactory(slicedBufferFactory),
      mMemSet(memSet),
      mRelocatableResizableBufferFactory(relocatableResizableBufferFactory) {}

vector<BaseSink::InputPort> BaseSink::createInputPorts() {
  vector<InputPort> inputPorts;

  const size_t defaultBufferSize = 8192;
  for (size_t i = 0; i < mInputPortCount; i++) {
    inputPorts.emplace_back(InputPort {
        .inputBuffer = unwrap(mRelocatableResizableBufferFactory->createRelocatableBuffer(defaultBufferSize)),
        .bufferCheckedOut = false,
    });
  }

  return inputPorts;
}

Result<IBuffer> BaseSink::requestBuffer(size_t port, size_t numBytes) noexcept {
  GS_REQUIRE_OR_RET_RESULT_FMT(
      port < mInputPorts.size(),
      "Cannot request buffer. Input port [%zu] is out of range.",
      port);

  InputPort& inputPort = mInputPorts[port];

  GS_REQUIRE_OR_RET_RESULT(!inputPort.bufferCheckedOut, "Cannot request buffer - it is already checked out");

  auto& buffer = inputPort.inputBuffer;

  if (buffer->range()->remaining() < numBytes) {
    buffer->resize(buffer->range()->endOffset() + numBytes);
  }

  inputPort.bufferCheckedOut = true;

  IBuffer* requestedBuffer;
  UNWRAP_OR_FWD_RESULT(requestedBuffer, mSlicedBufferFactory->sliceRemaining(buffer));

#ifdef DEBUG
  if (mMemSet != nullptr) {
    mMemSet->memSet(requestedBuffer->writePtr(), 0, requestedBuffer->range()->remaining());
  }
#endif  // DEBUG

  return makeRefResultNonNull(requestedBuffer);
}

Status BaseSink::commitBuffer(size_t port, size_t numBytes) noexcept {
  GS_REQUIRE_OR_RET_STATUS_FMT(
      port < mInputPorts.size(),
      "Cannot commit buffer. Input port [%zu] is out of range",
      port);

  InputPort& inputPort = mInputPorts[port];

  GS_REQUIRE_OR_RET_STATUS(inputPort.bufferCheckedOut, "Cannot commit buffer - it was not checked out");
  GS_REQUIRE_OR_RET_STATUS(
      numBytes <= inputPort.inputBuffer->range()->remaining(),
      "Cannot commit buffer - the committed number of bytes exceeds its capacity");
  FWD_IF_ERR(inputPort.inputBuffer->range()->increaseEndOffset(numBytes));

  inputPort.bufferCheckedOut = false;

  return Status_Success;
}

Result<IBuffer> BaseSink::getPortInputBuffer(size_t port) noexcept {
  FWD_IN_RESULT_IF_ERR(ensureInputPortsInit());
  GS_REQUIRE_OR_RET_RESULT_FMT(
      port < mInputPorts.size(),
      "Cannot get input buffer - Input port [%zu] is out of range",
      port);

  auto& inputPort = mInputPorts[port];

  GS_REQUIRE_OR_RET_RESULT(!inputPort.bufferCheckedOut, "Cannot get input buffer - buffer is checked out");

  return makeRefResultNonNull<IBuffer>(mInputPorts[port].inputBuffer);
}

Result<const IBuffer> BaseSink::getPortInputBuffer(size_t port) const noexcept {
  GS_REQUIRE_OR_RET_RESULT_FMT(
      port < mInputPorts.size(),
      "Cannot get input buffer - Input port [%zu] is out of range",
      port);

  auto& inputPort = mInputPorts[port];

  GS_REQUIRE_OR_RET_RESULT(!inputPort.bufferCheckedOut, "Cannot get input buffer - buffer is checked out");

  return makeRefResultNonNull<const IBuffer>(mInputPorts[port].inputBuffer);
}

Status BaseSink::consumeInputBytesAndMoveUsedToStart(size_t port, size_t numBytes) noexcept {
  FWD_IF_ERR(ensureInputPortsInit());
  GS_REQUIRE_OR_RET_STATUS_FMT(
      port < mInputPorts.size(),
      "Cannot get input buffer - Input port [%zu] is out of range",
      port);

  auto& inputBuffer = mInputPorts[port].inputBuffer;
  const size_t offset = inputBuffer == nullptr ? 0 : inputBuffer->range()->offset();

  if (numBytes == 0 && offset == 0) {
    return Status_Success;
  }

  GS_REQUIRE_OR_RET(inputBuffer != nullptr, "Cannot consume data - no buffer has been created", Status_InvalidState);

  FWD_IF_ERR(inputBuffer->range()->increaseOffset(numBytes));
  FWD_IF_ERR(inputBuffer->relocateUsedToStart());

  return Status_Success;
}

Status BaseSink::ensureInputPortsInit() noexcept {
  if (mInputPorts.size() < mInputPortCount) {
    DO_OR_RET_STATUS(mInputPorts = createInputPorts());
  }

  return Status_Success;
}
