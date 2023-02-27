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

#ifndef GPUSDRPIPELINE_FILTERDRIVER_H
#define GPUSDRPIPELINE_FILTERDRIVER_H

#include "Factories.h"
#include "Result.h"
#include "driver/IFilterDriver.h"
#include "driver/ISteppingDriver.h"

class FilterDriver final : public IFilterDriver {
 public:
  explicit FilterDriver(ISteppingDriver* steppingDriver) noexcept;

  void setDriverInput(Sink* sink) noexcept final;
  void setDriverOutput(Source* source) noexcept final;

  [[nodiscard]] Status connect(Source* source, size_t sourcePort, Sink* sink, size_t sinkPort) noexcept final;
  [[nodiscard]] Status setupNode(Node* node, const char* functionInGraph) noexcept final;
  [[nodiscard]] Status setupSourcePort(Source* source, size_t sourcePort, const IBufferCopier* sourceOutputMemCopier) noexcept final;
  [[nodiscard]] size_t preferredInputBufferSize(size_t port) noexcept final;

  void iterateOverConnections(
      void* context,
      void (*connectionIterator)(
          IDriver* driver,
          void* context,
          Source* source,
          size_t sourcePort,
          Sink* sink,
          size_t sinkPort) noexcept) noexcept final;
  void iterateOverNodes(
      void* context,
      void (*nodeIterator)(IDriver* driver, void* context, Node* node) noexcept) noexcept final;
  void iterateOverNodeAttributes(
      Node* node,
      void* context,
      void (*nodeAttrIterator)(
          IDriver* driver,
          Node* node,
          void* context,
          const char* attrName,
          const char* attrVal) noexcept) noexcept final;

  [[nodiscard]] Result<IBuffer> requestBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] Status commitBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  [[nodiscard]] Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;
  size_t getNodeName(Node* node, char* name, size_t nameBufLen, bool* foundOut) noexcept final;

 private:
  struct CallbackInfo {
    CallbackInfo(
        void* context,
        FilterDriver* thisFilterDriver,
        void (*connectionIterator)(
            IDriver* driver,
            void* context,
            Source* source,
            size_t sourcePort,
            Sink* sink,
            size_t sinkPort) noexcept);
    CallbackInfo(void* context, FilterDriver* thisFilterDriver, void (*nodeIterator)(IDriver* driver, void* context, Node* node) noexcept);
    CallbackInfo(
        void* context,
        FilterDriver* thisFilterDriver,
        void (*nodeAttrIterator)(
            IDriver* driver,
            Node* node,
            void* context,
            const char* attrName,
            const char* attrVal) noexcept);

    void* const context;
    FilterDriver* const filterDriver;
    void (*const connectionIterator)(
        IDriver* driver,
        void* context,
        Source* source,
        size_t sourcePort,
        Sink* sink,
        size_t sinkPort) noexcept;

    void (*const nodeIterator)(IDriver* driver, void* context, Node* node) noexcept;
    void (*const nodeAttrIterator)(
        IDriver* driver,
        Node* node,
        void* context,
        const char* attrName,
        const char* attrVal) noexcept;
  };

 private:
  ConstRef<ISteppingDriver> mSteppingDriver;

  Ref<Sink> mInputDelegate;
  Ref<Source> mOutputDelegate;

 private:
  static void translateConnectionIt(
      IDriver* driver,
      void* context,
      Source* source,
      size_t sourcePort,
      Sink* sink,
      size_t sinkPort) noexcept;

  static void translateNodeIt(IDriver* driver, void* context, Node* node) noexcept;

  static void translateNodeAttrIt(
      IDriver* driver,
      Node* node,
      void* context,
      const char* attrName,
      const char* attrVal) noexcept;

  REF_COUNTED(FilterDriver);
};

#endif  // GPUSDRPIPELINE_FILTERDRIVER_H
