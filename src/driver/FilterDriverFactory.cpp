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
#include "FilterDriverFactory.h"

#include <ParseJson.h>

#include <nlohmann/json.hpp>

using namespace std;

FilterDriverFactory::FilterDriverFactory(IFactories* factories) noexcept
    : mFactories(factories) {}

Result<Node> FilterDriverFactory::create(const char* jsonParameters) noexcept {
  try {
    const auto params = nlohmann::json::parse(jsonParameters);
    auto inputPortsData = tryGetJsonArray(params, "inputPorts");
    auto outputPortsData = tryGetJsonArray(params, "outputPorts");
    auto nodeData = tryGetJsonObj(params, "nodes");
    auto connectionsData = tryGetJsonArray(params, "connections");

    // Build nodes first
    // Build input and output port mappings second (depends on node defs)
    // Build Connections third (depends on nodes)

    unordered_map<string, ImmutableRef<Node>> nodes;
    for (const auto& kvPair : nodeData) {
      const auto& nodeId = kvPair.first;
      const auto& nodeDef = kvPair.second;

      if (!nodeDef.contains("type")) {
        gsloge("Node definition for [%s] does not contain a type.", nodeId.c_str());
        return ERR_RESULT(Status_InvalidArgument);
      }

      string nodeType = nodeDef["type"];
      Ref<Node> node;
      UNWRAP_OR_FWD_RESULT(node, createNode(nodeType.c_str(), jsonParameters));

      auto insertIt = nodes.emplace(nodeId, node);

      if (!insertIt.second) {
        gsloge("Duplicate definition for node [%s].", nodeId.c_str());
        return ERR_RESULT(Status_InvalidArgument);
      }
    }

    const auto component = unwrapRaw(mFactories->getFilterDriverFactory()->createFilterDriver());

    // Setup input port mappings
    if (!inputPortsData.empty()) {
      Ref<IPortRemappingSink> inputPortMapper;
      UNWRAP_OR_FWD_RESULT(inputPortMapper, mFactories->getPortRemappingSinkFactory()->create());

      for (const auto& inputMapping : inputPortsData) {
        const auto exposedPort = getJson<uint64_t>(inputMapping, "exposedPort");
        const auto sinkName = getJson<string>(inputMapping, "mapped/node"_json_pointer);
        const auto innerPort = getJson<uint64_t>(inputMapping, "mapped/port"_json_pointer);

        auto nodeIt = nodes.find(sinkName);
        if (nodeIt == nodes.end()) {
          gsloge("Cannot add an input port mapping with node [%s] because it was not defined.", sinkName.c_str());
          return ERR_RESULT(Status_NotFound);
        }

        ConstRef<Sink> sink = nodeIt->second->asSink();
        if (sink == nullptr) {
          gsloge("Cannot add an input port mapping with node [%s] because it is not a sink.", sinkName.c_str());
          return ERR_RESULT(Status_InvalidArgument);
        }

        inputPortMapper->addPortMapping(exposedPort, sink, innerPort);
      }

      component->setDriverInput(inputPortMapper.get());
    }

    // Setup output port mappings
    if (!outputPortsData.empty()) {
      Ref<IPortRemappingSource> outputPortMapper;
      UNWRAP_OR_FWD_RESULT(outputPortMapper, mFactories->getPortRemappingSourceFactory()->create());

      for (const auto& outputMapping : outputPortsData) {
        const auto exposedPort = getJson<uint64_t>(outputMapping, "exposedPort");
        const auto nodeId = getJson<string>(outputMapping, "mapped/node"_json_pointer);
        const auto innerPort = getJson<uint64_t>(outputMapping, "mapped/port"_json_pointer);

        auto nodeIt = nodes.find(nodeId);
        if (nodeIt == nodes.end()) {
          gsloge("Cannot add an output port mapping with node [%s] because it was not defined.", nodeId.c_str());
          return ERR_RESULT(Status_NotFound);
        }

        ConstRef<Source> source = nodeIt->second->asSource();
        if (source == nullptr) {
          gsloge("Cannot add an output port mapping with node [%s] because it is not a source.", nodeId.c_str());
          return ERR_RESULT(Status_InvalidArgument);
        }

        outputPortMapper->addPortMapping(exposedPort, source, innerPort);
      }

      component->setDriverOutput(outputPortMapper.get());
    }

    // Connect nodes
    for (const auto& connection : connectionsData) {
      const string sourceId = connection["source"];
      const size_t sourcePort = connection["sourcePort"];
      const string sinkId = connection["sink"];
      const size_t sinkPort = connection["sinkPort"];

      auto sourceIt = nodes.find(sourceId);
      auto sinkIt = nodes.find(sinkId);

      if (sourceIt == nodes.end()) {
        gsloge(
            "Cannot connect source [%s] port [%zu] to sink [%s] port [%zu]. Source is not defined.",
            sourceId.c_str(),
            sourcePort,
            sinkId.c_str(),
            sinkPort);

        return ERR_RESULT(Status_InvalidArgument);
      } else if (sinkIt == nodes.end()) {
        gsloge(
            "Cannot connect source [%s] port [%zu] to sink [%s] port [%zu]. Sink is not defined.",
            sourceId.c_str(),
            sourcePort,
            sinkId.c_str(),
            sinkPort);

        return ERR_RESULT(Status_InvalidArgument);
      }

      ConstRef<Source> source = sourceIt->second->asSource();
      ConstRef<Sink> sink = sinkIt->second->asSink();

      if (source == nullptr) {
        gsloge(
            "Cannot connect source [%s] port [%zu] to sink [%s] port [%zu]. [%s] is not a source.",
            sourceId.c_str(),
            sourcePort,
            sinkId.c_str(),
            sinkPort,
            sourceId.c_str());

        return ERR_RESULT(Status_InvalidArgument);
      } else if (sink == nullptr) {
        gsloge(
            "Cannot connect source [%s] port [%zu] to sink [%s] port [%zu]. [%s] is not a sink.",
            sourceId.c_str(),
            sourcePort,
            sinkId.c_str(),
            sinkPort,
            sinkId.c_str());

        return ERR_RESULT(Status_InvalidArgument);
      }

      FWD_IN_RESULT_IF_ERR(component->connect(source, sourcePort, sink, sinkPort));
    }

    return makeRefResultNonNull<Node>(component);
  }
  IF_CATCH_RETURN_RESULT;

  /*
   *
   * {
   *
   *   type: "component",
   *   inputPorts: [
   *     {
   *       exposedPort: 0,
   *       mapped: {
   *         node: "idOfSomeNode",
   *         port: 2
   *       }
   *     },
   *     {
   *       exposedPort: 1,
   *       mapped {
   *       node: "anotherNode",
   *       port: 0
   *     }
   *   ],
   *   outputPorts: [
   *     {
   *       exposedPort: 0,
   *       mapped: {
   *         node: "sourceNodeId",
   *         port: 1
   *       }
   *     }
   *   ],
   *   nodes: ...
   *   connections: ...
   * }
   *
   *
   * {
   *   type: "component",
   *     inputPorts: [
   *
   *     ],
   *     outputPorts: [
   *
   *     ],
   *     connections: [
   *       {
   *         source: "hackrf",
   *         description: "Get samples from HackRF",
   *         sourcePort: 0,
   *         target: "multiply",
   *         targetPort: 0
   *       },
   *       {
   *         source: "cosine",
   *         sourcePort: 0,
   *         target: "multiply",
   *         targetPort: 1
   *       },
   *       ...
   *       {
   *         source: "audioLowPass",
   *         sourcePort: 0,
   *         target: "audioWriter",
   *         targetPort: 0
   *       }
   *     ],
   *     nodes: {
   *       hackrf: {
   *         type: "HackRfSource",
   *         centerFrequency: 98.5e6,
   *         ...
   *       },
   *       cosine: {
   *         type: "Cosine",
   *         sampleRate: 12345
   *         frequency: 1234
   *       },
   *       multiply: {
   *         type: "MultiplyCC",
   *       },
   *       rfLowPass: {
   *
   *       },
   *       quadDemod: {
   *
   *       },
   *       audioLowPass: {
   *
   *       },
   *       audioWriter: {
   *
   *       }
   *     }
   *   }
   * }
   */
}

Result<IFilterDriver> FilterDriverFactory::createFilterDriver() noexcept {
  ISteppingDriverFactory* steppingDriverFactory = mFactories->getSteppingDriverFactory();
  ISteppingDriver* steppingDriver;
  UNWRAP_OR_FWD_RESULT(steppingDriver, steppingDriverFactory->createSteppingDriver());

  return makeRefResultNonNull<IFilterDriver>(new (std::nothrow) FilterDriver(steppingDriver));
}
