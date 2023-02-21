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

#include "SteppingDriver.h"

#include <cstring>
#include <functional>
#include <list>
#include <utility>

#include "GSErrors.h"
#include "GSLog.h"
#include "Result.h"

using namespace std;

static string NAME_NOT_SET = "NOT SET";

struct SteppingDriver::SourcePortInfo {
  Ref<const IBufferCopier> memCopier;
  list<SinkKey> connectedSinks;
};

struct SteppingDriver::SourceConnectionInfo {
  explicit SourceConnectionInfo(const ImmutableRef<Source>& source)
      : source(source) {}

  ConstRef<Source> source;
  vector<SourcePortInfo> connections;
};

struct SteppingDriver::SinkConnectionInfo {
  explicit SinkConnectionInfo(const ImmutableRef<Sink>& sink)
      : sinkName(NAME_NOT_SET),
        sink(sink) {}

  string sinkName;
  const ConstRef<Sink> sink;
  vector<SourceKey> connectedSources;  // Source Port -> connected source/port
};

struct SteppingDriver::NodeInfo {
  explicit NodeInfo(const ImmutableRef<Node>& node)
      : node(node),
        name(NAME_NOT_SET) {}
  NodeInfo()
      : name(NAME_NOT_SET) {}

  ConstRef<Node> node;
  string name;
};

size_t SteppingDriver::SourceKeyHasher::operator()(const SteppingDriver::SourceKey& sourceInfo) const {
  return hash<Source*> {}(sourceInfo.source.get()) ^ hash<size_t> {}(sourceInfo.port);
}

size_t SteppingDriver::SinkKeyHasher::operator()(const SteppingDriver::SinkKey& sinkInfo) const {
  return hash<Sink*> {}(sinkInfo.sink) ^ hash<size_t> {}(sinkInfo.port);
}

SteppingDriver::SourceKey::SourceKey(const ImmutableRef<Source>& source, size_t port)
    : source(source),
      port(port) {}

SteppingDriver::SourceKey::SourceKey()
    : port(0) {}

namespace std {}  // namespace std

bool SteppingDriver::SourceKey::operator==(const SourceKey& other) const {
  return other.source == source && other.port == port;
}

SteppingDriver::SinkKey::SinkKey(const ImmutableRef<Sink>& sink, size_t port)
    : sink(sink.get()),
      port(port) {}

SteppingDriver::SinkKey::SinkKey(Sink* sink, size_t port)
    : sink(sink),
      port(port) {}

SteppingDriver::SinkKey::SinkKey()
    : sink(nullptr),
      port(0) {}

bool SteppingDriver::SinkKey::operator==(const SinkKey& other) const {
  return other.sink == sink && other.port == port;
}

Status SteppingDriver::connect(Source* source, size_t sourcePort, Sink* sink, size_t sinkPort) noexcept {
  const SourceKey sourceKey(source, sourcePort);
  const SinkKey sinkKey(sink, sinkPort);

  Status connectableStatus = ensureConnectable(sinkKey);
  if (connectableStatus != Status_Success) {
    return connectableStatus;
  }

  auto sourceConnectionInfo = getOrCreateSourceConnectionInfo(source);
  if (sourceConnectionInfo->connections.size() <= sourcePort) {
    sourceConnectionInfo->connections.resize(sourcePort + 1);
  }
  sourceConnectionInfo->connections[sourcePort].connectedSinks.push_back(sinkKey);

  auto sinkConnectionInfo = getOrCreateSinkConnectionInfo(sink);
  if (sinkConnectionInfo->connectedSources.size() <= sinkPort) {
    sinkConnectionInfo->connectedSources.resize(sinkPort + 1);
  }
  sinkConnectionInfo->connectedSources.emplace_back(source, sourcePort);

  auto it = sinkConnectionInfo->connectedSources.begin() + sinkPort;
  sinkConnectionInfo->connectedSources.emplace(it, source, sourcePort);

  if (source->asSink() != nullptr) {
    mGraphTails.erase(source->asSink());
  }

  if (sink->asSource() == nullptr || mConnections.find(sink->asSource()) == mConnections.end()) {
    mGraphTails.emplace(sink);
  }

  return Status_Success;
}

void SteppingDriver::setupNode(Node* node, const char* functionInGraph) noexcept {
  getOrCreateNodeInfo(node)->name = functionInGraph;
}

void SteppingDriver::setupSourcePort(
    Source* source,
    size_t sourcePort,
    const IBufferCopier* sourceOutputMemCopier) noexcept {
  const auto connectionInfo = getOrCreateSourcePortInfo(source, sourcePort)->memCopier = sourceOutputMemCopier;
}

void SteppingDriver::iterateOverConnections(
    void* context,
    void (*connectionIterator)(
        IDriver* driver,
        void* context,
        Source* source,
        size_t sourcePort,
        Sink* sink,
        size_t sinkPort) noexcept) noexcept {
  for (auto& sourcesWithConnections : mConnections) {
    Source* source = sourcesWithConnections.first;
    auto connections = sourcesWithConnections.second->connections;

    for (size_t sourcePort = 0; sourcePort < connections.size(); sourcePort++) {
      auto& sinks = connections[sourcePort].connectedSinks;

      for (auto& sinkInfo : sinks) {
        connectionIterator(this, context, source, sourcePort, sinkInfo.sink, sinkInfo.port);
      }
    }
  }
}

void SteppingDriver::iterateOverNodes(
    void* context,
    void (*nodeIterator)(IDriver* driver, void* context, Node* node) noexcept) noexcept {
  for (auto& nodeInfo : mNodeInfos) {
    nodeIterator(this, context, nodeInfo.second->node.get());
  }
}

void SteppingDriver::iterateOverNodeAttributes(
    Node* node,
    void* context,
    void (*nodeAttrIterator)(
        IDriver* driver,
        Node* node,
        void* context,
        const char* attrName,
        const char* attrVal) noexcept) noexcept {}

/*
 * Start with graph tails.
 * For each sink:
 *   for each source, if also a sink, recurse
 *   for each source, read output
 *   commit input buffers to sink
 *
 * Only one source can be connected to a sink port.
 */
Status SteppingDriver::doFilter() noexcept {
  for (auto& sink : mGraphTails) {
    DO_OR_RET_STATUS({
      Status status = doSinkInput(sink);
      if (status != Status_Success) {
        return status;
      }
    });
  }

  return Status_Success;
}

Status SteppingDriver::doSinkInput(Sink* sink) {
  auto sinkConnectionInfo = getSinkConnectionInfo(sink);
  if (sinkConnectionInfo == nullptr) {
    // cerr << "Requested Sink [" << getNodeName(sink) << "] input, but it has no connection info" << endl;
    return Status_Success;
  }

  const string sinkName = getLocalNodeName(sink);
  for (auto& connection : sinkConnectionInfo->connectedSources) {
    const string sourceName = getLocalNodeName(connection.source.get());

    if (connection.source == nullptr) {
      gslog(GSLOG_TRACE, "Sink [%s] does not have a Source connected it.", sinkName.c_str());
      continue;
    }

    gslog(GSLOG_TRACE, "Found [%s] -> [%s]", sourceName.c_str(), sinkName.c_str());

    const bool sourceIsSink = connection.source->asSink();

    while (sourceIsSink && !sourceHasDataForAllPorts(connection.source)) {
      gslog(GSLOG_TRACE, "[%s] needs data - fetching", sourceName.c_str());

      doSinkInput(connection.source->asSink());

      if (!sourceHasDataForAllPorts(connection.source)) {
        gslog(
            GSLOG_TRACE,
            "[%s] did not produce output on all ports after requesting input for it",
            sourceName.c_str());

        return Status_Success;
      }
    }

    if (!sourceIsSink) {
      gslog(GSLOG_TRACE, "[%s] is not a sink, not back-propagating", sourceName.c_str());
    }

    if (sourceHasDataForAllPorts(connection.source)) {
      gslog(
          GSLOG_TRACE,
          "[%s] has data for all output ports [%zu]",
          sourceName.c_str(),
          connection.source->getOutputDataSize(0));

      doSourceOutput(connection.source);
    }
  }

  return Status_Success;
}

Status SteppingDriver::doSourceOutput(const ImmutableRef<Source>& source) {
  const auto connectionInfo = getOrCreateSourceConnectionInfo(source);
  const string sourceName = getLocalNodeName(connectionInfo->source.get());

  gslog(GSLOG_TRACE, "doSourceOutput for [%s]", sourceName.c_str());

  vector<IBuffer*> sourceOutputBuffers(connectionInfo->connections.size());

  /*
   * When an output port has more than one Sink, the first buffer is used for source output, then that buffer is copied
   * to the second+ buffers stored in extraBuffersCopiedInto[sourcePort]
   */
  vector<list<IBuffer*>> extraBuffersCopiedInto(connectionInfo->connections.size());
  unordered_map<SinkKey, size_t, SinkKeyHasher> sinkPortSizesToCommit;

  for (size_t sourcePort = 0; sourcePort < connectionInfo->connections.size(); sourcePort++) {
    const auto& sinksConnectedToUpstreamSource = connectionInfo->connections[sourcePort].connectedSinks;

    for (auto& sinkKey : sinksConnectedToUpstreamSource) {
      const auto sink = sinkKey.sink;
      const size_t sinkPort = sinkKey.port;

      if (sink == nullptr) {
        GS_FAIL(
            "Source [" << getLocalNodeName(source.get()) << "] must have a sink connected to port [" << sourcePort
                       << "]");
      }

      const size_t alignment = source->getOutputSizeAlignment(sinkPort);
      const size_t sinkPreferredInputSize = sink->preferredInputBufferSize(sinkPort);
      const size_t sourceAvailableOutputSize = source->getOutputDataSize(sourcePort);
      const size_t bufferSize =
          (min(sinkPreferredInputSize, sourceAvailableOutputSize) + alignment - 1) / alignment * alignment;

      Ref<IBuffer> sinkBuffer;
      UNWRAP_OR_FWD_STATUS(sinkBuffer, sink->requestBuffer(sinkPort, bufferSize));

#ifdef DEBUG
      const string sinkName = getLocalNodeName(sink);

      gslog(
          GSLOG_TRACE,
          "Requested buffer of size [%zu] from [%s] and got [%zu]",
          bufferSize,
          sinkName.c_str(),
          sinkBuffer->range()->remaining());

      if (sinkBuffer->range()->used() > 0) {
        gslog(
            GSLOG_TRACE,
            "Unexpected data in sink [%s] input buffer, size [%zu]",
            sinkName.c_str(),
            sinkBuffer->range()->remaining());
      }
#endif  // DEBUG

      if (sourceOutputBuffers[sourcePort] == nullptr) {
        sourceOutputBuffers[sourcePort] = sinkBuffer.get();
      } else {
        extraBuffersCopiedInto[sourcePort].push_back(sinkBuffer.get());
      }
    }
  }

  FWD_IF_ERR(source->readOutput(sourceOutputBuffers.data(), sourceOutputBuffers.size()));

  for (size_t sourcePort = 0; sourcePort < connectionInfo->connections.size(); sourcePort++) {
    auto& connection = connectionInfo->connections[sourcePort];
    const auto& sinksConnectedToUpstreamSource = connection.connectedSinks;

#ifdef DEBUG
    for (auto& sinkKey : sinksConnectedToUpstreamSource) {
      const auto sinkName = getLocalNodeName(sinkKey.sink);

      gslog(
          GSLOG_TRACE,
          "[%s|%zu] -> [%s|%zu] num bytes [%zu]",
          sourceName.c_str(),
          sourcePort,
          sinkName.c_str(),
          sinkKey.port,
          sourceOutputBuffers[sourcePort]->range()->used());
    }
  }
#endif  // DEBUG

  for (size_t sourcePort = 0; sourcePort < sourceOutputBuffers.size(); sourcePort++) {
    auto populatedBuffer = sourceOutputBuffers[sourcePort];
    const size_t dataSize = populatedBuffer->range()->used();

    auto copyToList = extraBuffersCopiedInto[sourcePort];
    for (auto& targetBuffer : copyToList) {
      const auto& memCopier = connectionInfo->connections[sourcePort].memCopier;

      if (memCopier == nullptr) {
        gslog(
            GSLOG_ERROR,
            "Source [%s] has multiple outgoing connections form port [%zu], but a copier hasn't been setup. Call "
            "setupSourcePort() to configure.",
            getLocalNodeName(source).c_str(),
            sourcePort);
      }
      FWD_IF_ERR(memCopier->copy(targetBuffer->writePtr(), populatedBuffer->readPtr(), dataSize));
      FWD_IF_ERR(targetBuffer->range()->increaseEndOffset(dataSize));
    }

    for (const auto& sinkKey : connectionInfo->connections[sourcePort].connectedSinks) {
      FWD_IF_ERR(sinkKey.sink->commitBuffer(sinkKey.port, dataSize));
    }
  }

  return Status_Success;
}

shared_ptr<SteppingDriver::SourceConnectionInfo> SteppingDriver::getOrCreateSourceConnectionInfo(
    const ImmutableRef<Source>& source) {
  auto it = mConnections.find(source.get());
  if (it == mConnections.end()) {
    auto insertIt = mConnections.emplace(make_pair(source.get(), make_shared<SourceConnectionInfo>(source)));
    it = insertIt.first;
  }

  return it->second;
}

std::shared_ptr<SteppingDriver::SourcePortInfo> SteppingDriver::getOrCreateSourcePortInfo(
    const ImmutableRef<Source>& source,
    size_t port) {
  SourceKey sourceKey(source, port);
  auto it = mSourcePortInfo.find(sourceKey);
  if (it == mSourcePortInfo.end()) {
    auto insertIt = mSourcePortInfo.emplace(pair(sourceKey, make_shared<SourcePortInfo>()));
    it = insertIt.first;
  }

  return it->second;
}

shared_ptr<SteppingDriver::SinkConnectionInfo> SteppingDriver::getOrCreateSinkConnectionInfo(
    const ImmutableRef<Sink>& sink) {
  auto it = mReverseConnections.find(sink.get());
  if (it == mReverseConnections.end()) {
    auto insertIt = mReverseConnections.emplace(sink.get(), make_shared<SinkConnectionInfo>(sink));
    it = insertIt.first;
  }

  return it->second;
}

shared_ptr<SteppingDriver::SinkConnectionInfo> SteppingDriver::getSinkConnectionInfo(Sink* sink) {
  auto it = mReverseConnections.find(sink);
  if (it == mReverseConnections.end()) {
    return nullptr;
  }

  return it->second;
}

shared_ptr<SteppingDriver::NodeInfo> SteppingDriver::getOrCreateNodeInfo(const ImmutableRef<Node>& node) {
  auto it = mNodeInfos.find(node.get());
  if (it == mNodeInfos.end()) {
    auto insertIt = mNodeInfos.emplace(node.get(), make_shared<NodeInfo>(node));
    it = insertIt.first;
  }

  return it->second;
}

shared_ptr<SteppingDriver::NodeInfo> SteppingDriver::getNodeInfo(Node* node) {
  auto it = mNodeInfos.find(node);
  if (it == mNodeInfos.end()) {
    return nullptr;
  }

  return it->second;
}

Status SteppingDriver::ensureConnectable(const SinkKey& sinkInfo) noexcept {
  auto it = mReverseConnections.find(sinkInfo.sink);

  if (it == mReverseConnections.end()) {
    return Status_Success;
  }

  const auto& sinkConnectionInfo = it->second;
  const auto& connectedSources = sinkConnectionInfo->connectedSources;

  if (sinkInfo.port < connectedSources.size() && connectedSources[sinkInfo.port].source != nullptr) {
    const auto& source = connectedSources[sinkInfo.port].source;
    const size_t sourcePort = connectedSources[sinkInfo.port].port;
    gslog(
        GSLOG_ERROR,
        "Sink [%s] port [%zu] is already connected to Source [%s] port [%zu]",
        getLocalNodeName(sinkInfo.sink).c_str(),
        sinkInfo.port,
        getLocalNodeName(source.get()).c_str(),
        sourcePort);

    return Status_InvalidState;
  }

  return Status_Success;
}

std::string SteppingDriver::getLocalNodeName(Node* node) {
  const auto nodeInfo = getNodeInfo(node);
  if (nodeInfo == nullptr) {
    return NAME_NOT_SET;
  }

  return nodeInfo->name;
}

bool SteppingDriver::sourceHasDataForAllPorts(const ImmutableRef<Source>& source) {
  const auto connectionInfo = getOrCreateSourceConnectionInfo(source);

  for (size_t sourcePort = 0; sourcePort < connectionInfo->connections.size(); sourcePort++) {
    if (source->getOutputDataSize(sourcePort) == 0) {
      return false;
    }
  }

  return true;
}

size_t SteppingDriver::getNodeName(Node* node, char* name, size_t nameBufLen, bool* foundOut) noexcept {
  auto nodeIt = mNodeInfos.find(node);

  if (nodeIt != mNodeInfos.end()) {
    auto nameStr = getLocalNodeName(node);
    nameStr.copy(name, nameBufLen, 0);
    if (nameBufLen > nameStr.length()) {
      name[nameStr.length()] = 0;
    }

    *foundOut = true;
    return nameStr.length();
  }

  for (auto& nodeInfo : mNodeInfos) {
    auto driver = nodeInfo.first->asDriver();
    if (driver != nullptr) {
      size_t nameLength = driver->getNodeName(node, name, nameBufLen, foundOut);

      if (*foundOut) {
        return nameLength;
      }
    }
  }

  *foundOut = false;
  if (nameBufLen > 0) {
    name[0] = 0;
  }

  return 0;
}
