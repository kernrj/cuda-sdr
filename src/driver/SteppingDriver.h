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

#ifndef GPUSDRPIPELINE_STEPPINGDRIVER_H
#define GPUSDRPIPELINE_STEPPINGDRIVER_H

#include <list>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "gpusdrpipeline/buffers/IBufferCopier.h"
#include "gpusdrpipeline/driver/ISteppingDriver.h"
#include "gpusdrpipeline/filters/Filter.h"

class SteppingDriver final : public ISteppingDriver {
 public:
  SteppingDriver() noexcept = default;
  SteppingDriver(const SteppingDriver&) = delete;
  SteppingDriver(SteppingDriver&&) = delete;

  /**
   *
   * @param source
   * @param sourcePort
   * @param sink
   * @param sinkPort
   * @param sourceOutputMemCopier Used when the source connect
   */
  [[nodiscard]] Status connect(Source* source, size_t sourcePort, Sink* sink, size_t sinkPort) noexcept final;
  [[nodiscard]] Status setupNode(Node* node, const char* functionInGraph) noexcept final;
  [[nodiscard]] Status setupSourcePort(Source* source, size_t sourcePort, const IBufferCopier* sourceOutputMemCopier) noexcept final;

  void iterateOverConnections(void* context, void (*connectionIterator)(
                                                 IDriver* driver,
                                                 void* context,
                                                 Source* source,
                                                 size_t sourcePort,
                                                 Sink* sink,
                                                 size_t sinkPort) noexcept) noexcept final;
  void iterateOverNodes(void* context, void (*nodeIterator)(IDriver* driver, void* context, Node* node) noexcept) noexcept final;
  void iterateOverNodeAttributes(Node* node, void* context, void (*nodeAttrIterator)(
                                                                IDriver* driver,
                                                                Node* node,
                                                                void* context,
                                                                const char* attrName,
                                                                const char* attrVal) noexcept) noexcept final;
  size_t getNodeName(Node* node, char* name, size_t nameBufLen, bool* foundOut) noexcept final;

  [[nodiscard]] Status doFilter() noexcept final;

 private:
  struct SourceKey {
    SourceKey(const ImmutableRef<Source>& source, size_t port);
    SourceKey();
    bool operator==(const SourceKey& other) const;

    Ref<Source> source;
    size_t port;
  };

  struct SinkKey {
    SinkKey(const ImmutableRef<Sink>& sink, size_t port);
    SinkKey(Sink* sink, size_t port);
    SinkKey();
    bool operator==(const SinkKey& other) const;

    Sink* sink;
    size_t port;
  };

  struct SourceKeyHasher {
    size_t operator()(const SourceKey& sourceInfo) const;
  };

  struct SinkKeyHasher {
    size_t operator()(const SinkKey& sourceInfo) const;
  };

  struct NodeInfo;
  struct SourcePortInfo;
  struct SourceConnectionInfo;
  struct SinkConnectionInfo;

  using ConnectionMapType = std::unordered_map<Source*, std::shared_ptr<SourceConnectionInfo>>;
  using ReverseConnectionMapType = std::unordered_map<Sink*, std::shared_ptr<SinkConnectionInfo>>;

  ConnectionMapType mConnections;
  ReverseConnectionMapType mReverseConnections;

  std::unordered_set<Sink*> mGraphTails;
  std::unordered_map<Node*, std::shared_ptr<NodeInfo>> mNodeInfos;
  std::unordered_map<SourceKey, std::shared_ptr<SourcePortInfo>, SourceKeyHasher> mSourcePortInfo;

 private:
  std::shared_ptr<SourcePortInfo> getOrCreateSourcePortInfo(const ImmutableRef<Source>& source, size_t port);
  std::shared_ptr<SourceConnectionInfo> getOrCreateSourceConnectionInfo(const ImmutableRef<Source>& source);
  std::shared_ptr<SinkConnectionInfo> getOrCreateSinkConnectionInfo(const ImmutableRef<Sink>& sink);
  [[nodiscard]] std::shared_ptr<SinkConnectionInfo> getSinkConnectionInfo(Sink* sink);
  std::shared_ptr<NodeInfo> getOrCreateNodeInfo(const ImmutableRef<Node>& node);
  [[nodiscard]] std::shared_ptr<NodeInfo> getNodeInfo(Node* node);
  std::string getLocalNodeName(Node* node);

  [[nodiscard]] Status ensureConnectable(const SinkKey& sinkInfo) noexcept;
  [[nodiscard]] Status doSinkInput(Sink* sink);
  [[nodiscard]] Status doSourceOutput(const ImmutableRef<Source>& source);
  [[nodiscard]] bool sourceHasDataForAllPorts(const ImmutableRef<Source>& source);

  REF_COUNTED(SteppingDriver);
};

#endif  // GPUSDRPIPELINE_STEPPINGDRIVER_H
