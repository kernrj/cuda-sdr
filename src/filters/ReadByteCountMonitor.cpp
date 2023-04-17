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

#include "ReadByteCountMonitor.h"

ReadByteCountMonitor::ReadByteCountMonitor(Filter* filter) noexcept
    : mFilter(filter) {}

size_t ReadByteCountMonitor::getByteCountRead(size_t port) noexcept {
  if (port >= mTotalByteCountRead.size()) {
    return 0;
  } else {
    return mTotalByteCountRead[port];
  }
}

Result<IBuffer> ReadByteCountMonitor::requestBuffer(size_t port, size_t byteCount) noexcept {
  return mFilter->requestBuffer(port, byteCount);
}

Status ReadByteCountMonitor::commitBuffer(size_t port, size_t byteCount) noexcept {
  return mFilter->commitBuffer(port, byteCount);
}

size_t ReadByteCountMonitor::getOutputDataSize(size_t port) noexcept { return mFilter->getOutputDataSize(port); }

size_t ReadByteCountMonitor::getOutputSizeAlignment(size_t port) noexcept {
  return mFilter->getOutputSizeAlignment(port);
}

Status ReadByteCountMonitor::readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept {
  if (mTotalByteCountRead.size() < numPorts) {
    mTotalByteCountRead.resize(numPorts);
    mUsedDataBeforeReadOnLastIteration.resize(numPorts);
  }

  for (size_t port = 0; port < numPorts; port++) {
    mUsedDataBeforeReadOnLastIteration[port] = portOutputBuffers[port]->range()->used();
  }

  FWD_IF_ERR(mFilter->readOutput(portOutputBuffers, numPorts));

  for (size_t port = 0; port < numPorts; port++) {
    const size_t byteCountJustRead =
        portOutputBuffers[port]->range()->used() - mUsedDataBeforeReadOnLastIteration[port];
    mTotalByteCountRead[port] += byteCountJustRead;
  }

  return Status_Success;
}

size_t ReadByteCountMonitor::preferredInputBufferSize(size_t port) noexcept {
  return mFilter->preferredInputBufferSize(port);
}

IBufferCopier* ReadByteCountMonitor::getOutputCopier(size_t port) noexcept { return mFilter->getOutputCopier(port); }
