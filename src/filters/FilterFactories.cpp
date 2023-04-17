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

#include "filters/FilterFactories.h"

#include "Factories.h"

using namespace std;

static unordered_map<string, ImmutableRef<INodeFactory>> gNodeFactories;

static mutex gLock;

GS_C_LINKAGE Result<Node> createNode(const char* name, const char* jsonParameters) noexcept {
  Ref<INodeFactory> factory;

  DO_OR_RET_ERR_RESULT({
    lock_guard<mutex> l(gLock);

    auto it = gNodeFactories.find(name);
    if (it == gNodeFactories.end()) {
      return ERR_RESULT(Status_NotFound);
    }

    factory = it->second;

    return factory->create(jsonParameters);
  });
}

GS_C_LINKAGE Result<Filter> createFilter(const char* name, const char* jsonParameters) noexcept {
  Ref<Node> node;
  UNWRAP_OR_FWD_RESULT(node, createNode(name, jsonParameters));

  ConstRef<Filter> filter = node->asFilter();
  if (filter == nullptr) {
    const char* isSourceStr = node->asSource() != nullptr ? "yes" : "no";
    const char* isSinkStr = node->asSink() != nullptr ? "yes" : "no";
    const char* isComponentStr = node->asDriver() != nullptr ? "yes" : "no";

    gsloge(
        "[%s] was created, but is not a Filter. Source? [%s] Sink? [%s] Component? [%s].",
        name,
        isSourceStr,
        isSinkStr,
        isComponentStr);

    return ERR_RESULT(Status_InvalidArgument);
  }

  return makeRefResultNonNull(filter);
}

GS_C_LINKAGE Result<Source> createSource(const char* name, const char* jsonParameters) noexcept {
  Ref<Node> node;
  UNWRAP_OR_FWD_RESULT(node, createNode(name, jsonParameters));

  ConstRef<Source> source = node->asSource();
  if (source == nullptr) {
    const char* isFilterStr = node->asFilter() != nullptr ? "yes" : "no";
    const char* isSinkStr = node->asSink() != nullptr ? "yes" : "no";
    const char* isComponentStr = node->asDriver() != nullptr ? "yes" : "no";

    gsloge(
        "[%s] was created, but is not a Source. Filter? [%s] Sink? [%s] Component? [%s].",
        name,
        isFilterStr,
        isSinkStr,
        isComponentStr);

    return ERR_RESULT(Status_InvalidArgument);
  }

  return makeRefResultNonNull(source);
}

GS_C_LINKAGE Result<Sink> createSink(const char* name, const char* jsonParameters) noexcept {
  Ref<Node> node;
  UNWRAP_OR_FWD_RESULT(node, createNode(name, jsonParameters));

  ConstRef<Sink> sink = node->asSink();
  if (sink == nullptr) {
    const char* isSourceStr = node->asSource() != nullptr ? "yes" : "no";
    const char* isFilterStr = node->asFilter() != nullptr ? "yes" : "no";
    const char* isComponentStr = node->asDriver() != nullptr ? "yes" : "no";

    gsloge(
        "[%s] was created, but is not a Sink. Source? [%s] Filter? [%s] Component? [%s].",
        name,
        isSourceStr,
        isFilterStr,
        isComponentStr);

    return ERR_RESULT(Status_InvalidArgument);
  }

  return makeRefResultNonNull(sink);
}

GS_C_LINKAGE bool hasNodeFactory(const char* name) noexcept {
  lock_guard<mutex> l(gLock);
  DO_OR_RET_STATUS(return gNodeFactories.find(name) != gNodeFactories.end());
}

GS_C_LINKAGE Status registerNodeFactory(const char* name, INodeFactory* filterFactory) noexcept {
  DO_OR_RET_STATUS({
    lock_guard<mutex> l(gLock);

    gNodeFactories.erase(name);

    if (filterFactory != nullptr) {
      gNodeFactories.emplace(name, ImmutableRef<INodeFactory>(filterFactory));
    }

    return Status_Success;
  });
}

GS_C_LINKAGE Status registerDefaultFilterFactories() noexcept {
  Ref<IFactories> factories;
  UNWRAP_OR_FWD_STATUS(factories, getFactoriesSingleton());

  FWD_IF_ERR(registerNodeFactory("AacWriter", factories->getAacFileWriterFactory()));
  FWD_IF_ERR(registerNodeFactory("AddConst", factories->getAddConstFactory()));
  FWD_IF_ERR(registerNodeFactory("AddConstToVectorLength", factories->getAddConstToVectorLengthFactory()));
  FWD_IF_ERR(registerNodeFactory("Component", factories->getFilterDriverFactory()));
  FWD_IF_ERR(registerNodeFactory("Cosine", factories->getCosineSourceFactory()));
  FWD_IF_ERR(registerNodeFactory("File", factories->getFileReaderFactory()));
  FWD_IF_ERR(registerNodeFactory("Fir", factories->getFirFactory()));
  FWD_IF_ERR(registerNodeFactory("HackRfSource", factories->getHackrfSourceFactory()));
  FWD_IF_ERR(registerNodeFactory("Int8ToFloat", factories->getInt8ToFloatFactory()));
  FWD_IF_ERR(registerNodeFactory("Magnitude", factories->getMagnitudeFactory()));
  FWD_IF_ERR(registerNodeFactory("MultiplyCCC", factories->getMultiplyFactory()));
  FWD_IF_ERR(registerNodeFactory("QuadDemod", factories->getQuadDemodFactory()));

  return Status_Success;
}
