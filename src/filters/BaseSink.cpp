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
    const std::shared_ptr<IRelocatableResizableBufferFactory>& relocatableResizableBufferFactory,
    const std::shared_ptr<IBufferSliceFactory>& slicedBufferFactory,
    size_t inputPortCount,
    const std::shared_ptr<IMemSet>& memSet)
    : mRelocatableResizableBufferFactory(relocatableResizableBufferFactory),
      mSlicedBufferFactory(slicedBufferFactory),
      mInputPorts(createInputPorts(inputPortCount)),
      mMemSet(memSet) {}

vector<BaseSink::InputPort> BaseSink::createInputPorts(size_t inputPortCount) {
  vector<InputPort> inputPorts(inputPortCount);

  for (size_t i = 0; i < inputPortCount; i++) {
    inputPorts[i] = InputPort {
        .bufferCheckedOut = false,
    };
  }

  return inputPorts;
}

shared_ptr<IBuffer> BaseSink::requestBuffer(size_t port, size_t numBytes) {
  if (port >= mInputPorts.size()) {
    throw runtime_error("Cannot request buffer - Input port [" + to_string(port) + "] is out of range.");
  }

  InputPort& inputPort = mInputPorts[port];

  if (inputPort.bufferCheckedOut) {
    throw runtime_error("Cannot request buffer - it is already checked out");
  }

  if (inputPort.inputBuffer == nullptr) {
    inputPort.inputBuffer = mRelocatableResizableBufferFactory->createRelocatableBuffer(numBytes);
  }

  auto& buffer = inputPort.inputBuffer;

  if (buffer->range()->remaining() < numBytes) {
    buffer->resize(buffer->range()->endOffset() + numBytes, nullptr);
  }

  inputPort.bufferCheckedOut = true;

  auto requestedBuffer = mSlicedBufferFactory->sliceRemaining(buffer);

#ifdef DEBUG
  if (mMemSet != nullptr) {
    mMemSet->memSet(requestedBuffer->writePtr(), 0, requestedBuffer->range()->remaining());
  }
#endif  // DEBUG

  return requestedBuffer;
}

void BaseSink::commitBuffer(size_t port, size_t numBytes) {
  if (port >= mInputPorts.size()) {
    throw runtime_error("Cannot commit buffer - Input port [" + to_string(port) + "] is out of range.");
  }

  InputPort& inputPort = mInputPorts[port];

  if (!inputPort.bufferCheckedOut) {
    throw runtime_error("Cannot commit buffer - it was not checked out");
  } else if (numBytes > inputPort.inputBuffer->range()->remaining()) {
    throw runtime_error("Cannot commit buffer - the number of bytes is greater than its capacity.");
  }

  inputPort.inputBuffer->range()->increaseEndOffset(numBytes);
  inputPort.bufferCheckedOut = false;
}

shared_ptr<IBuffer> BaseSink::getPortInputBuffer(size_t port) {
  if (port >= mInputPorts.size()) {
    throw runtime_error("Cannot get input buffer - Input port [" + to_string(port) + "] is out of range.");
  }

  auto& inputPort = mInputPorts[port];

  if (inputPort.bufferCheckedOut) {
    throw runtime_error("Cannot get input buffer - buffer is checked out.");
  }

  return mInputPorts[port].inputBuffer;
}

shared_ptr<const IBuffer> BaseSink::getPortInputBuffer(size_t port) const {
  if (port >= mInputPorts.size()) {
    throw runtime_error("Cannot get input buffer - Input port [" + to_string(port) + "] is out of range.");
  }

  auto& inputPort = mInputPorts[port];

  if (inputPort.bufferCheckedOut) {
    throw runtime_error("Cannot get input buffer - buffer is checked out.");
  }

  return mInputPorts[port].inputBuffer;
}

void BaseSink::consumeInputBytesAndMoveUsedToStart(size_t port, size_t numBytes) {
  if (port >= mInputPorts.size()) {
    throw runtime_error("Cannot get input buffer - Input port [" + to_string(port) + "] is out of range.");
  }

  auto& inputBuffer = mInputPorts[port].inputBuffer;
  const size_t offset = inputBuffer == nullptr ? 0 : inputBuffer->range()->offset();

  if (numBytes == 0 && offset == 0) {
    return;
  } else if (inputBuffer == nullptr) {
    THROW("Cannot consume data - no buffer has been created");
  }

  inputBuffer->range()->increaseOffset(numBytes);
  inputBuffer->relocateUsedToStart();
}
