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

#include "PortRemappingSink.h"

PortRemappingSink::PortRemappingSink(Sink* sink) noexcept
    : mMapToSink(sink) {}

void PortRemappingSink::addPortMapping(size_t outerPort, size_t innerPort) noexcept { mPortMap[outerPort] = innerPort; }

Result<IBuffer> PortRemappingSink::requestBuffer(size_t port, size_t byteCount) noexcept {
  GS_REQUIRE_OR_RET_RESULT_FMT(mPortMap.find(port) != mPortMap.end(), "Input port [%zu] is not mapped", port);

  size_t innerPort = mPortMap[port];
  return mMapToSink->requestBuffer(innerPort, byteCount);
}

Status PortRemappingSink::commitBuffer(size_t port, size_t byteCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS_FMT(mPortMap.find(port) != mPortMap.end(), "Input port [%zu] is not mapped", port);

  size_t innerPort = mPortMap[port];
  FWD_IF_ERR(mMapToSink->commitBuffer(innerPort, byteCount));

  return Status_Success;
}

size_t PortRemappingSink::preferredInputBufferSize(size_t port) noexcept {
  GS_REQUIRE_OR_ABORT(mPortMap.find(port) != mPortMap.end(), "Port not found");
  return mMapToSink->preferredInputBufferSize(mPortMap[port]);
}
