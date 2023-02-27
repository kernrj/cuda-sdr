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

#ifndef GPUSDRPIPELINE_DRIVERTOOL_H
#define GPUSDRPIPELINE_DRIVERTOOL_H

#include <gpusdrpipeline/driver/IDriver.h>

#include <list>
#include <unordered_map>

class DriverTool {
 public:
  struct SinkConnection {
    Sink* const sink;
    const size_t port;
  };

  using SourcePortType = size_t;
  using ConnectedSinks = std::list<SinkConnection>;
  using SourcePorts = std::unordered_map<SourcePortType, ConnectedSinks>;

 public:
  static std::vector<Node*> getNodes(IDriver* driver) {
    GetNodesContext context;
    driver->iterateOverNodes(&context, iterateOverNodesCallback);

    return std::move(context.nodes);
  }

  static std::vector<std::pair<std::string, std::string>> getNodeAttributes(IDriver* driver, Node* node) {
    GetNodeAttributesContext context;
    driver->iterateOverNodeAttributes(node, &context, iterateOverNodeAttributesCallback);
    return std::move(context.attributes);
  }

  static std::unordered_map<Source*, SourcePorts> getConnections(IDriver* driver) {
    GetConnectionsContext context;
    driver->iterateOverConnections(&context, iterateOverConnectionsCallback);
    return std::move(context.connections);
  }

  static std::string getNodeName(IDriver* driver, Node* node, bool* foundNodeOut) {
    const size_t nodeNameLength = driver->getNodeName(node, nullptr, 0, foundNodeOut);
    std::string nodeName(nodeNameLength, 0);
    driver->getNodeName(node, nodeName.data(), nodeNameLength, foundNodeOut);

    return nodeName;
  }

 private:
  struct GetNodesContext {
    std::vector<Node*> nodes;
  };

  struct GetNodeAttributesContext {
    std::vector<std::pair<std::string, std::string>> attributes;
  };

  struct GetConnectionsContext {
    std::unordered_map<Source*, SourcePorts> connections;
  };

 private:
  static void iterateOverNodesCallback([[maybe_unused]] IDriver* driver, void* context, Node* node) noexcept {
    auto getNodesContext = reinterpret_cast<GetNodesContext*>(context);
    getNodesContext->nodes.push_back(node);
  }

  static void iterateOverNodeAttributesCallback(
      [[maybe_unused]] IDriver* driver,
      [[maybe_unused]] Node* node,
      void* context,
      const char* attrName,
      const char* attrVal) noexcept {
    auto attrContext = reinterpret_cast<GetNodeAttributesContext*>(context);
    attrContext->attributes.emplace_back(attrName, attrVal);
  }

  static void iterateOverConnectionsCallback(
      [[maybe_unused]] IDriver* driver,
      void* context,
      Source* source,
      size_t sourcePort,
      Sink* sink,
      size_t sinkPort) noexcept {
    auto connectionContext = reinterpret_cast<GetConnectionsContext*>(context);

    auto& connections = connectionContext->connections;
    auto sourceIt = connections.find(source);
    if (sourceIt == connections.end()) {
      auto insertIt = connections.emplace(source, SourcePorts());
      sourceIt = insertIt.first;
    }

    auto& sourcePorts = sourceIt->second;
    auto sourcePortIt = sourcePorts.find(sourcePort);
    if (sourcePortIt == sourcePorts.end()) {
      auto insertIt = sourcePorts.emplace(sourcePort, ConnectedSinks());
      sourcePortIt = insertIt.first;
    }

    auto& connectedSinks = sourcePortIt->second;
    connectedSinks.push_back(SinkConnection {
        .sink = sink,
        .port = sinkPort,
    });
  }
};

#endif  // GPUSDRPIPELINE_DRIVERTOOL_H
