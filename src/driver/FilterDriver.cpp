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

#include "FilterDriver.h"

/*
 * getInput() returns a Sink.
 * User calls Sink.requestBuffer().
 *
 * setInput(Sink)
 * setOutput(Source)
 *
 * Wraps the node, and is changeable at run time.
 */

class SinkWrapper final : public virtual Sink {
 public:
  SinkWrapper() = default;
  explicit SinkWrapper(Sink* sink)
      : mSink(sink) {}

  void setSink(Sink* sink) noexcept { mSink = sink; }

  [[nodiscard]] Result<IBuffer> requestBuffer(size_t port, size_t byteCount) noexcept final {
    return mSink->requestBuffer(port, byteCount);
  }

  [[nodiscard]] Status commitBuffer(size_t port, size_t byteCount) noexcept final {
    return mSink->commitBuffer(port, byteCount);
  }

  [[nodiscard]] size_t preferredInputBufferSize(size_t port) noexcept final {
    return mSink->preferredInputBufferSize(port);
  }

 private:
  Ref<Sink> mSink;

  REF_COUNTED(SinkWrapper);
};

class SourceWrapper final : public virtual Source {
  SourceWrapper() = default;
  explicit SourceWrapper(Source* source)
      : mSource(source) {}

  void setSource(Source* source) noexcept { mSource = source; }

  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final {
    GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
    return mSource->getOutputDataSize(port);
  }

  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final {
    GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
    return mSource->getOutputSizeAlignment(port);
  }

  Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final {
    return mSource->readOutput(portOutputBuffers, numPorts);
  }

 private:
  Ref<Source> mSource;

  REF_COUNTED(SourceWrapper);
};

FilterDriver::FilterDriver(ISteppingDriver* steppingDriver) noexcept
    : mSteppingDriver(steppingDriver) {}

void FilterDriver::setDriverInput(Sink* sink) noexcept { mInputDelegate = sink; }
void FilterDriver::setDriverOutput(Source* source) noexcept { mOutputDelegate = source; }

Status FilterDriver::connect(Source* source, size_t sourcePort, Sink* sink, size_t sinkPort) noexcept {
  return mSteppingDriver->connect(source, sourcePort, sink, sinkPort);
}

Status FilterDriver::setupNode(Node* node, const char* functionInGraph) noexcept {
  return mSteppingDriver->setupNode(node, functionInGraph);
}

Status FilterDriver::setupSourcePort(
    Source* source,
    size_t sourcePort,
    const IBufferCopier* sourceOutputMemCopier) noexcept {
  return mSteppingDriver->setupSourcePort(source, sourcePort, sourceOutputMemCopier);
}

void FilterDriver::iterateOverConnections(
    void* context,
    void (*connectionIterator)(
        IDriver* driver,
        void* context,
        Source* source,
        size_t sourcePort,
        Sink* sink,
        size_t sinkPort) noexcept) noexcept {
  auto info = CallbackInfo(context, this, connectionIterator);
  mSteppingDriver->iterateOverConnections(&info, translateConnectionIt);

  if (mInputDelegate != nullptr) {
    connectionIterator(this, context, this, 0, mInputDelegate.get(), 0);
  }

  if (mOutputDelegate != nullptr) {
    connectionIterator(this, context, mOutputDelegate.get(), 0, this, 0);
  }
}

void FilterDriver::iterateOverNodes(
    void* context,
    void (*nodeIterator)(IDriver* driver, void* context, Node* node) noexcept) noexcept {
  auto info = CallbackInfo(context, this, nodeIterator);
  mSteppingDriver->iterateOverNodes(&info, translateNodeIt);
}

// TODO: Connect external ports to input/output delegate ports

void FilterDriver::iterateOverNodeAttributes(
    Node* node,
    void* context,
    void (*nodeAttrIterator)(
        IDriver* driver,
        Node* node,
        void* context,
        const char* attrName,
        const char* attrVal) noexcept) noexcept {
  auto info = CallbackInfo(context, this, nodeAttrIterator);

  mSteppingDriver->iterateOverNodeAttributes(node, &info, translateNodeAttrIt);

  if (node == nullptr) {
    return;
  }

  if (node->asSink() == mInputDelegate.get()) {
    nodeAttrIterator(this, node, context, "inputNode", "true");
  }

  if (node->asSource() == mOutputDelegate.get()) {
    nodeAttrIterator(this, node, context, "outputNode", "true");
  }
}

Result<IBuffer> FilterDriver::requestBuffer(size_t port, size_t byteCount) noexcept {
  if (mInputDelegate == nullptr) {
    gsloge("Cannot use FilterDriver as a Sink until a node is set via setDriverInput()");
    return ERR_RESULT(Status_InvalidState);
  }

  return mInputDelegate->requestBuffer(port, byteCount);
}

Status FilterDriver::commitBuffer(size_t port, size_t byteCount) noexcept {
  if (mInputDelegate == nullptr) {
    gsloge("Cannot use FilterDriver as a Sink until a node is set via setDriverInput()");
    return Status_InvalidState;
  }

  FWD_IF_ERR(mInputDelegate->commitBuffer(port, byteCount));
  return mSteppingDriver->doFilter();
}

size_t FilterDriver::getOutputDataSize(size_t port) noexcept {
  if (mOutputDelegate == nullptr) {
    gslogw("Cannot use FilterDriver as a Source until a node is set via setDriverOutput()");
    return 0;
  }

  if (mOutputDelegate->getOutputDataSize(port) == 0) {
    RET_IF_ERR(mSteppingDriver->doFilter(), 0);
  }

  return mOutputDelegate->getOutputDataSize(port);
}

size_t FilterDriver::getOutputSizeAlignment(size_t port) noexcept {
  if (mOutputDelegate == nullptr) {
    gslogw("Cannot use FilterDriver as a Source until a node is set via setDriverOutput()");
    return 1;
  }

  return mOutputDelegate->getOutputSizeAlignment(port);
}

Status FilterDriver::readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept {
  if (mOutputDelegate == nullptr) {
    gsloge("Cannot use FilterDriver as a Source until a node is set via setDriverOutput()");
    return Status_InvalidState;
  }

  bool allOutputPortsHaveData = true;
  for (size_t port = 0; port < numPorts; port++) {
    if (mOutputDelegate->getOutputDataSize(port) == 0) {
      allOutputPortsHaveData = false;
      break;
    }
  }

  if (!allOutputPortsHaveData) {
    FWD_IF_ERR(mSteppingDriver->doFilter());
  }

  return mOutputDelegate->readOutput(portOutputBuffers, numPorts);
}

FilterDriver::CallbackInfo::CallbackInfo(
    void* context,
    FilterDriver* thisFilterDriver,
    void (*connectionIterator)(
        IDriver* driver,
        void* context,
        Source* source,
        size_t sourcePort,
        Sink* sink,
        size_t sinkPort) noexcept)
    : context(context),
      filterDriver(thisFilterDriver),
      connectionIterator(connectionIterator),
      nodeIterator(nullptr),
      nodeAttrIterator(nullptr) {}

FilterDriver::CallbackInfo::CallbackInfo(
    void* context,
    FilterDriver* thisFilterDriver,
    void (*nodeIterator)(IDriver* driver, void* context, Node* node) noexcept)
    : context(context),
      filterDriver(thisFilterDriver),
      connectionIterator(nullptr),
      nodeIterator(nodeIterator),
      nodeAttrIterator(nullptr) {}

FilterDriver::CallbackInfo::CallbackInfo(
    void* context,
    FilterDriver* thisFilterDriver,
    void (*nodeAttrIterator)(
        IDriver* driver,
        Node* node,
        void* context,
        const char* attrName,
        const char* attrVal) noexcept)
    : context(context),
      filterDriver(thisFilterDriver),
      connectionIterator(nullptr),
      nodeIterator(nullptr),
      nodeAttrIterator(nodeAttrIterator) {}

void FilterDriver::translateConnectionIt(
    [[maybe_unused]] IDriver* driver,
    void* context,
    Source* source,
    size_t sourcePort,
    Sink* sink,
    size_t sinkPort) noexcept {
  auto info = reinterpret_cast<CallbackInfo*>(context);
  GS_REQUIRE_OR_ABORT(info->connectionIterator != nullptr, "Connection iterator not set");
  info->connectionIterator(info->filterDriver, info->context, source, sourcePort, sink, sinkPort);
}

void FilterDriver::translateNodeIt([[maybe_unused]] IDriver* driver, void* context, Node* node) noexcept {
  auto info = reinterpret_cast<CallbackInfo*>(context);
  GS_REQUIRE_OR_ABORT(info->nodeIterator != nullptr, "Node iterator not set");
  info->nodeIterator(info->filterDriver, info->context, node);
}

void FilterDriver::translateNodeAttrIt(
    [[maybe_unused]] IDriver* driver,
    Node* node,
    void* context,
    const char* attrName,
    const char* attrVal) noexcept {
  auto info = reinterpret_cast<CallbackInfo*>(context);
  GS_REQUIRE_OR_ABORT(info->nodeAttrIterator != nullptr, "Node attribute iterator not set");
  info->nodeAttrIterator(info->filterDriver, node, info->context, attrName, attrVal);
}

size_t FilterDriver::getNodeName(Node* node, char* name, size_t nameBufLen, bool* foundOut) noexcept {
  return mSteppingDriver->getNodeName(node, name, nameBufLen, foundOut);
}

size_t FilterDriver::preferredInputBufferSize(size_t port) noexcept {
  return mInputDelegate->preferredInputBufferSize(port);
}
