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

void PortRemappingSink::addPortMapping(size_t outerPort, Sink* innerSink, size_t innerSinkPort) noexcept {
  mPortMap.emplace(outerPort, SinkAndPort {innerSink, innerSinkPort});
}

Result<IBuffer> PortRemappingSink::requestBuffer(size_t port, size_t byteCount) noexcept {
  auto it = mPortMap.find(port);

  GS_REQUIRE_OR_RET_RESULT_FMT(it != mPortMap.end(), "Input port [%zu] is not mapped", port);

  auto sink = it->second.sink;
  auto innerPort = it->second.port;

  return sink->requestBuffer(innerPort, byteCount);
}

Status PortRemappingSink::commitBuffer(size_t port, size_t byteCount) noexcept {
  auto it = mPortMap.find(port);

  GS_REQUIRE_OR_RET_STATUS_FMT(it != mPortMap.end(), "Input port [%zu] is not mapped", port);

  auto sink = it->second.sink;
  auto innerPort = it->second.port;

  FWD_IF_ERR(sink->commitBuffer(innerPort, byteCount));

  return Status_Success;
}

size_t PortRemappingSink::preferredInputBufferSize(size_t port) noexcept {
  auto it = mPortMap.find(port);

  GS_REQUIRE_OR_ABORT(it != mPortMap.end(), "Port not found");

  auto sink = it->second.sink;
  auto innerPort = it->second.port;

  return sink->preferredInputBufferSize(innerPort);
}
