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

#include "PortRemappingSource.h"

using namespace std;

Result<IPortRemappingSource> PortRemappingSource::create(IFactories* factories) noexcept {

  return makeRefResultNonNull<IPortRemappingSource>(new (nothrow) PortRemappingSource());
}

void PortRemappingSource::addPortMapping(size_t outerPort, Source* innerSource, size_t innerSourcePort) noexcept {
  mPortMapByExternalPort.emplace(outerPort, SourceAndPort {innerSource, innerSourcePort});

  auto it = mPortMapBySource.find(innerSource);
  if (it == mPortMapBySource.end()) {
    auto insertIt = mPortMapBySource.emplace(innerSource, vector<SourceMapping>());
    it = insertIt.first;
  }

  vector<SourceMapping>& mappingsForSource = it->second;
  mappingsForSource.push_back(SourceMapping {
      .exposedPort = outerPort,
      .innerPort = innerSourcePort,
  });
}

size_t PortRemappingSource::getOutputDataSize(size_t port) noexcept {
  auto it = mPortMapByExternalPort.find(port);

  if (it == mPortMapByExternalPort.end()) {
    gslogw("Cannot get output data size. Output port [%zu] is not mapped.", port);
    return 0;
  }

  auto source = it->second.source;
  auto innerPort = it->second.port;

  return source->getOutputDataSize(innerPort);
}

size_t PortRemappingSource::getOutputSizeAlignment(size_t port) noexcept {
  auto it = mPortMapByExternalPort.find(port);

  if (it == mPortMapByExternalPort.end()) {
    gslogw("Cannot get output size alignment. Output port [%zu] is not mapped.", port);
    return 1;
  }

  auto source = it->second.source;
  auto innerPort = it->second.port;

  return source->getOutputSizeAlignment(innerPort);
}

size_t PortRemappingSource::getAlignedOutputDataSize(size_t port) noexcept {
  auto it = mPortMapByExternalPort.find(port);

  if (it == mPortMapByExternalPort.end()) {
    gslogw("Cannot get aligned output data size. Output port [%zu] is not mapped.", port);
    return 1;
  }

  auto source = it->second.source;
  auto innerPort = it->second.port;

  return source->getAlignedOutputDataSize(innerPort);
}

Status PortRemappingSource::readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept {
  for (const auto& sourceMappingsPair : mPortMapBySource) {
    ConstRef<Source> source = sourceMappingsPair.first;
    const vector<SourceMapping>& portMappingsForSource = sourceMappingsPair.second;

    vector<IBuffer*> portOutputBuffersForSource;

    // Assemble the list of buffers
    for (const auto& mapping : portMappingsForSource) {
      while (mapping.innerPort >= portOutputBuffersForSource.size()) {
        portOutputBuffersForSource.push_back(nullptr);
      }

      if (mapping.exposedPort >= numPorts) {
        gsloge("Too few buffers were supplied. Minimum is [%zu].", getMinOutputBufferCount());
      }

      portOutputBuffersForSource[mapping.innerPort] = portOutputBuffers[mapping.exposedPort];
    }

    // Make sure there all buffers are set
    for (size_t i = 0; i < portOutputBuffersForSource.size(); i++) {
      const IBuffer* buffer = portOutputBuffersForSource[i];
      if (buffer == nullptr) {
        gsloge(
            "Inner port [%zu] is not connected for source with internal -> external mappings [%s]",
            i,
            mappingsForSourceToString(portMappingsForSource).c_str());

        return Status_InvalidArgument;
      }
    }

    // Get output from source
    Status status = source->readOutput(portOutputBuffersForSource.data(), portOutputBuffersForSource.size());
    if (status != Status_Success) {
      return status;
    }
  }

  return Status_Success;
}

std::string PortRemappingSource::mappingsForSourceToString(const std::vector<SourceMapping>& mappings) {
  ostringstream out;

  bool firstOutput = true;
  for (const auto& mapping : mappings) {
    if (!firstOutput) {
      out << ", ";
    }
    firstOutput = false;

    out << mapping.innerPort << " -> " << mapping.exposedPort;
  }

  return out.str();
}

size_t PortRemappingSource::getMinOutputBufferCount() const {
  size_t minRequiredBufferCount = 0;
  for (const auto& portInfo : mPortMapByExternalPort) {
    if (portInfo.first >= minRequiredBufferCount) {
      minRequiredBufferCount = portInfo.first + 1;
    }
  }

  return minRequiredBufferCount;
}

IBufferCopier* PortRemappingSource::getOutputCopier(size_t port) noexcept {
  auto it = mPortMapByExternalPort.find(port);

  if (it == mPortMapByExternalPort.end()) {
    gslogw("Cannot get output copier. Output port [%zu] is not mapped.", port);
    return nullptr;
  }

  auto source = it->second.source;
  auto innerPort = it->second.port;

  return source->getOutputCopier(innerPort);
}
