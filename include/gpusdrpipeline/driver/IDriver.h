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

#ifndef GPUSDRPIPELINE_IDRIVER_H
#define GPUSDRPIPELINE_IDRIVER_H

#include <gpusdrpipeline/buffers/IBufferCopier.h>
#include <gpusdrpipeline/filters/Filter.h>

class IDriver : public Node {
 public:
  IDriver* asDriver() noexcept override { return this; }

  /**
   *
   * @param source
   * @param sourcePort
   * @param sink
   * @param sinkPort
   * @param sourceOutputMemCopier Used when the source connect
   */
  [[nodiscard]] virtual Status connect(Source* source, size_t sourcePort, Sink* sink, size_t sinkPort) noexcept = 0;

  virtual void setupNode(Node* node, const char* functionInGraph) noexcept = 0;

  virtual void iterateOverConnections(
      void* context,
      void (*connectionIterator)(
          IDriver* driver,
          void* context,
          Source* source,
          size_t sourcePort,
          Sink* sink,
          size_t sinkPort) noexcept) noexcept = 0;
  virtual void iterateOverNodes(
      void* context,
      void (*nodeIterator)(IDriver* driver, void* context, Node* node) noexcept) noexcept = 0;
  virtual void iterateOverNodeAttributes(
      Node* node,
      void* context,
      void (*nodeAttrIterator)(
          IDriver* driver,
          Node* node,
          void* context,
          const char* attrName,
          const char* attrVal) noexcept) noexcept = 0;

  virtual size_t getNodeName(Node* node, char* name, size_t nameBufLen, bool* foundOut) noexcept = 0;

  virtual void setupSourcePort(
      Source* source,
      size_t sourcePort,
      const IBufferCopier* sourceOutputMemCopier) noexcept = 0;

  ABSTRACT_IREF(IDriver);
};

#endif  // GPUSDRPIPELINE_IDRIVER_H
