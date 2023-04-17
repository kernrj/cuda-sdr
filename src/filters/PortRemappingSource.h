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

#ifndef GPUSDRPIPELINE_PORTREMAPPINGSOURCE_H
#define GPUSDRPIPELINE_PORTREMAPPINGSOURCE_H

#include <unordered_map>
#include <vector>

#include "Factories.h"
#include "filters/IPortRemappingSource.h"

class PortRemappingSource final : public IPortRemappingSource {
 public:
  static Result<IPortRemappingSource> create(IFactories* factories) noexcept;

  void addPortMapping(size_t outerPort, Source* innerSource, size_t innerSourcePort) noexcept final;

  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  [[nodiscard]] size_t getAlignedOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;
  IBufferCopier* getOutputCopier(size_t port) noexcept final;

 private:
  struct SourceAndPort {
    ConstRef<Source> source;
    const size_t port;
  };

  struct SourceMapping {
    size_t exposedPort;
    size_t innerPort;
  };

 private:
  std::unordered_map<size_t, SourceAndPort> mPortMapByExternalPort;
  std::unordered_map<Source*, std::vector<SourceMapping>> mPortMapBySource;

 private:
  static std::string mappingsForSourceToString(const std::vector<SourceMapping>& mappings);
  size_t getMinOutputBufferCount() const;

  REF_COUNTED(PortRemappingSource);
};

#endif  // GPUSDRPIPELINE_PORTREMAPPINGSOURCE_H
