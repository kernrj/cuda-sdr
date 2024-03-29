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

#ifndef GPUSDRPIPELINE_PORTREMAPPINGSINK_H
#define GPUSDRPIPELINE_PORTREMAPPINGSINK_H

#include <unordered_map>

#include "Result.h"
#include "filters/IPortRemappingSink.h"

class PortRemappingSink final : public IPortRemappingSink {
 public:
  void addPortMapping(size_t outerPort, Sink* innerSink, size_t innerSinkPort) noexcept final;

  [[nodiscard]] Result<IBuffer> requestBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] Status commitBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] size_t preferredInputBufferSize(size_t port) noexcept final;

 private:
  struct SinkAndPort {
    ConstRef<Sink> sink;
    const size_t port;
  };

 private:
  std::unordered_map<size_t, SinkAndPort> mPortMap;

  REF_COUNTED(PortRemappingSink);
};

#endif  // GPUSDRPIPELINE_PORTREMAPPINGSINK_H
