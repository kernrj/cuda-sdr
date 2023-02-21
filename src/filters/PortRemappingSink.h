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
  explicit PortRemappingSink(Sink* sink) noexcept;

  void addPortMapping(size_t outerPort, size_t innerPort) noexcept final;

  Result<IBuffer> requestBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] Status commitBuffer(size_t port, size_t byteCount) noexcept final;

 private:
  ConstRef<Sink> mMapToSink;
  std::unordered_map<size_t, size_t> mPortMap;

  REF_COUNTED(PortRemappingSink);
};

#endif  // GPUSDRPIPELINE_PORTREMAPPINGSINK_H
