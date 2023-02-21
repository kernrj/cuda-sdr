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

#include "DriverToDot.h"

#include <bits/stdc++.h>

#include "DriverTool.h"

using namespace std;

const string DriverToDot::kPostfixForSubgraphConnections = "_hiddenForSubgraphConnections";
const string DriverToDot::kNodeNameNotFound = "NODE NOT IN GRAPH";

static string createIndent(size_t indentSpaces) { return string(indentSpaces, ' '); }

void DriverToDot::outputConnections(
    IDriver* driver,
    void* context,
    Source* source,
    size_t sourcePort,
    Sink* sink,
    size_t sinkPort) noexcept {
  auto connectionCtx = reinterpret_cast<ConnectionCtx*>(context);

  const auto indentStr = createIndent(connectionCtx->rootIndent);
  vector<string> attributes;

  const string sourceId = connectionCtx->self->getOrCreateIdForConnections(source);
  const string sinkId = connectionCtx->self->getOrCreateIdForConnections(sink);

  if (source->asDriver() != nullptr) {
    const string tailId = connectionCtx->self->getOrCreateClusterId(source->asDriver());
    attributes.push_back(SSTREAM("ltail=\"" << tailId << "\""));
  }

  if (sink->asDriver() != nullptr) {
    const string headId = connectionCtx->self->getOrCreateClusterId(sink->asDriver());
    attributes.push_back(SSTREAM("lhead=\"" << headId << "\""));
  }

  connectionCtx->self->mOutStream << indentStr << '"' << sourceId << "\" -> \"" << sinkId << '"';

  if (!attributes.empty()) {
    connectionCtx->self->mOutStream << " [";

    for (size_t i = 0; i < attributes.size(); i++) {
      if (i > 0) {
        connectionCtx->self->mOutStream << ", ";
      }

      connectionCtx->self->mOutStream << attributes[i];
    }

    connectionCtx->self->mOutStream << ']';
  }

  connectionCtx->self->mOutStream << endl;
}

void DriverToDot::outputNodeAttributes(
    IDriver* driver,
    Node* node,
    void* context,
    const char* attrName,
    const char* attrVal) noexcept {
  auto attrCtx = reinterpret_cast<NodeAttrContext*>(context);

  attrCtx->attributes.emplace_back(attrName, attrVal);
}

void DriverToDot::outputSingleNodeDefinitions(IDriver* driver, void* context, Node* node) noexcept {
  auto connectionCtx = reinterpret_cast<ConnectionCtx*>(context);
  NodeAttrContext attrCtx {
      .indent = connectionCtx->rootIndent,
  };

  if (node->asDriver() != nullptr) {
    return;  // Handled in outputSubgraphs()
  }

  driver->iterateOverNodeAttributes(node, &attrCtx, outputNodeAttributes);

  auto indentStr = createIndent(connectionCtx->rootIndent);

  const string nodeId = connectionCtx->self->getOrCreateIdForConnections(node);
  const string nodeName = connectionCtx->self->getNodeName(node);
  connectionCtx->self->mOutStream << indentStr << '"' << nodeId << "\" [label=\"" << nodeName;

  for (auto& attr : attrCtx.attributes) {
    connectionCtx->self->mOutStream << "\\n" << attr.first << '=' << attr.second;
  }

  connectionCtx->self->mOutStream << "\"]" << endl;
}

void DriverToDot::outputSubgraphs(IDriver* parentDriver, void* context, Node* node) noexcept {
  auto ctx = reinterpret_cast<ConnectionCtx*>(context);

  if (node->asDriver() != nullptr) {
    NodeAttrContext attrCtx {
        .indent = ctx->rootIndent,
    };

    parentDriver->iterateOverNodeAttributes(node, &attrCtx, outputNodeAttributes);

    const string driverNodeName = ctx->self->getNodeName(node->asDriver());
    string graphLabel;
    if (!attrCtx.attributes.empty()) {
      ostringstream out;
      out << driverNodeName;

      for (auto& attr : attrCtx.attributes) {
        out << "\\n" << attr.first << '=' << attr.second;
      }

      graphLabel = out.str();
    } else {
      graphLabel = driverNodeName;
    }

    outputGraph(ctx->self, node->asDriver(), true, ctx->rootIndent, graphLabel, ctx->leftRightDir);
  }
}

Result<size_t> DriverToDot::convertToDot(
    IDriver* driver,
    const char* name,
    char* diagramBuffer,
    size_t diagramSize) noexcept {
  NON_NULL_OR_RET(driver);
  NON_NULL_OR_RET(name);

  if (diagramSize > 0 && diagramBuffer == nullptr) {
    gslog(GSLOG_ERROR, "diagramBuffer must be set when diagramSize [%zu] > 0", diagramSize);
    return ERR_RESULT(Status_InvalidArgument);
  }

  mOutStream.clear();

  mRootDriver = driver;
  mRootDriverName = name;
  outputGraph(this, driver, false, 0, name, true);
  mRootDriver = nullptr;
  mRootDriverName = "";

  string dot = mOutStream.str();

  mOutStream.clear();

  const size_t copyNumBytes = min(diagramSize, dot.size());
  memcpy(diagramBuffer, dot.c_str(), copyNumBytes);

  return makeValResult(dot.size());
}

Status DriverToDot::outputGraph(
    DriverToDot* self,
    IDriver* driver,
    bool isSubgraph,
    size_t rootIndent,
    const string& label,
    bool leftRightDir) {
  const size_t indentDelta = 2;
  const size_t indent = rootIndent + indentDelta;
  const size_t maxIndent = 511;
  const size_t maxRootIndent = maxIndent - indentDelta;

  GS_REQUIRE_OR_RET_STATUS(rootIndent < maxRootIndent, "Indent is too large");

  if (isSubgraph) {
    GS_REQUIRE_OR_RET_STATUS(!label.empty(), "Subgraphs must have labels");
  }

  auto rootIndentStr = createIndent(rootIndent);
  auto indentStr = createIndent(rootIndent + indentDelta);

  const char* graphType = isSubgraph ? "subgraph" : "digraph";
  const char* namePrefix = isSubgraph ? "cluster" : "";
  self->mOutStream << rootIndentStr << graphType << " \"" << self->getOrCreateClusterId(driver) << "\" {" << endl;

  if (!label.empty()) {
    self->mOutStream << indentStr << "label=\"" << label << '"' << endl;
  }

  if (!isSubgraph) {
    self->mOutStream << indentStr << "compound=true" << endl;
  } else {
    self->mOutStream << indentStr << '"' << self->getOrCreateHiddenNodeId(driver)
                     << R"(" [style="invisible", shape=none, width=0, height=0, fixedsize=true])" << endl;
  }

  const string dir = leftRightDir ? "LR" : "TB";
  self->mOutStream << indentStr << "rankdir=\"" << dir << '"' << endl;

  ConnectionCtx ctx = {
      .rootIndent = rootIndent + indentDelta,
      .self = self,
      .leftRightDir = !leftRightDir,
  };

  /*
  auto nodes = DriverTool::getNodes(driver);
  for (auto& node : nodes) {
    if (node->asDriver() != nullptr) {
      outputSubgraph(node->asDriver());
    } else {
      outputSingleNodeDefinition(node);
    }
  }
   */

  driver->iterateOverNodes(&ctx, outputSingleNodeDefinitions);  // output non-subgraph node definitions
  driver->iterateOverNodes(&ctx, outputSubgraphs);              // output subgraph definitions

  if (driver == self->mRootDriver) {
    vector<DriverInfo> drivers;
    drivers.push_back(DriverInfo {
        .depth = 0,
        .driver = self->mRootDriver,
    });
    self->getDriversRecursively(self->mRootDriver, 0, &drivers);

    for (auto subDriver : drivers) {
      auto subDriverCtx = ConnectionCtx {
          .rootIndent = indent,
          .self = self,
      };

      subDriver.driver->iterateOverConnections(&ctx, outputConnections);
    }
  }

  self->mOutStream << rootIndentStr << '}' << endl;

  return Status_Success;
}

std::string DriverToDot::getOrCreateBaseNodeId(Node* node) {
  const string name = getNodeName(node);

  auto existingNameIt = mExistingNodeNames.find(name);
  if (existingNameIt == mExistingNodeNames.end()) {
    auto insertIt = mExistingNodeNames.emplace(name, vector<Node*>());
    existingNameIt = insertIt.first;
  }

  vector<Node*>& listOfNodesWithTheSameName = existingNameIt->second;
  auto thisNodeIt = find(listOfNodesWithTheSameName.begin(), listOfNodesWithTheSameName.end(), node);
  if (thisNodeIt == listOfNodesWithTheSameName.end()) {
    thisNodeIt = listOfNodesWithTheSameName.insert(listOfNodesWithTheSameName.end(), node);
  }

  const size_t index = thisNodeIt - listOfNodesWithTheSameName.begin();

  return SSTREAM(name << index);
}

std::string DriverToDot::getOrCreateClusterId(IDriver* driver) {
  string clusterPrefix;
  if (driver != mRootDriver) {
    clusterPrefix = "cluster";
  }

  return SSTREAM(clusterPrefix << getOrCreateBaseNodeId(driver));
}

std::string DriverToDot::getNodeName(Node* node) {
  if (node->asDriver() == mRootDriver) {
    return mRootDriverName;
  }

  bool nodeFound = false;
  string name = DriverTool::getNodeName(mRootDriver, node, &nodeFound);

  if (!nodeFound) {
    return kNodeNameNotFound;
  }

  return name;
}

std::string DriverToDot::getOrCreateHiddenNodeId(IDriver* driver) {
  return SSTREAM(getOrCreateBaseNodeId(driver) << kPostfixForSubgraphConnections);
}

std::string DriverToDot::getOrCreateIdForConnections(Node* node) {
  if (node->asDriver() != nullptr) {
    return getOrCreateHiddenNodeId(node->asDriver());
  }

  return getOrCreateBaseNodeId(node);
}

void DriverToDot::getDriversRecursivelyCallback(IDriver* driver, void* context, Node* node) noexcept {
  if (node->asDriver() == nullptr) {
    return;
  }

  auto getDriversCtx = reinterpret_cast<GetDriversCtx*>(context);
  const size_t thisDriversDepth = getDriversCtx->parentDriverDepth + 1;
  getDriversCtx->drivers->push_back(DriverInfo {
      .depth = thisDriversDepth,
      .driver = node->asDriver(),
  });
  getDriversCtx->self->getDriversRecursively(node->asDriver(), thisDriversDepth, getDriversCtx->drivers);
}

void DriverToDot::getDriversRecursively(IDriver* driver, size_t parentDriverDepth, vector<DriverInfo>* subDrivers) {
  auto ctx = GetDriversCtx {
      .parentDriverDepth = parentDriverDepth,
      .self = this,
      .drivers = subDrivers,
  };

  driver->iterateOverNodes(&ctx, getDriversRecursivelyCallback);
}
