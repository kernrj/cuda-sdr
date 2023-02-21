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

#ifndef GPUSDRPIPELINE_DRIVERTODOT_H
#define GPUSDRPIPELINE_DRIVERTODOT_H

#include <unordered_map>
#include <vector>

#include "driver/IDriver.h"
#include "driver/IDriverToDiagram.h"

class DriverToDot final : public IDriverToDiagram {
 public:
  Result<size_t> convertToDot(IDriver* driver, const char* name, char* diagram, size_t diagramSize) noexcept final;

 private:
  static const std::string kNodeNameNotFound;
  static const std::string kPostfixForSubgraphConnections;

  std::ostringstream mOutStream;
  IDriver* mRootDriver;
  std::string mRootDriverName;
  std::unordered_map<std::string, std::vector<Node*>> mExistingNodeNames;

 private:
  struct NodeAttrContext {
    std::vector<std::pair<std::string, std::string>> attributes;
    const size_t indent;
  };

  struct ConnectionCtx {
    const size_t rootIndent;
    DriverToDot* const self;
    const bool leftRightDir;
  };

  struct DriverInfo {
    const size_t depth;
    IDriver* const driver;
  };

  struct GetDriversCtx {
    size_t parentDriverDepth;
    DriverToDot* const self;
    std::vector<DriverInfo>* const drivers;
  };

 private:
  static void outputNodeAttributes(
      IDriver* driver,
      Node* node,
      void* context,
      const char* attrName,
      const char* attrVal) noexcept;

  static void outputSubgraphs(IDriver* driver, void* context, Node* node) noexcept;

  static void outputConnections(
      IDriver* driver,
      void* context,
      Source* source,
      size_t sourcePort,
      Sink* sink,
      size_t sinkPort) noexcept;

  static void outputSingleNodeDefinitions(IDriver* driver, void* context, Node* node) noexcept;

  static Status outputGraph(
      DriverToDot* self,
      IDriver* driver,
      bool isSubgraph,
      size_t rootIndent,
      const std::string& label,
      bool leftRightDir);

  static void getDriversRecursivelyCallback(IDriver* driver, void* context, Node* node) noexcept;

  std::string getOrCreateBaseNodeId(Node* node);  // omits 'cluster' if an IDriver
  std::string getOrCreateClusterId(IDriver* driver);
  std::string getOrCreateHiddenNodeId(IDriver* driver);
  std::string getNodeName(Node* node);
  std::string getOrCreateIdForConnections(Node* node);
  void getDriversRecursively(IDriver* driver, size_t parentDriverDepth, std::vector<DriverInfo>* drivers);

  REF_COUNTED(DriverToDot);
};

#endif  // GPUSDRPIPELINE_DRIVERTODOT_H
