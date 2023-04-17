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

#ifndef GPUSDRPIPELINE_CUDAMEMCPYFILTERFACTORY_H
#define GPUSDRPIPELINE_CUDAMEMCPYFILTERFACTORY_H

#include <ParseJson.h>

#include <nlohmann/json.hpp>

#include "../CudaMemcpyFilter.h"
#include "filters/FilterFactories.h"

class CudaMemcpyFilterFactory final : public ICudaMemcpyFilterFactory {
 public:
  explicit CudaMemcpyFilterFactory(IFactories* factories)
      : mFactories(factories) {}

  Result<Node> create(const char* jsonParameters) noexcept override {
    nlohmann::json params;
    UNWRAP_OR_FWD_RESULT(params, parseJson(jsonParameters));

    const std::string commandQueueId = params["commandQueue"];
    const std::string from = params["from"];
    const std::string to = params["to"];

    cudaMemcpyKind memcpyKind;
    const std::string device = "device";
    const std::string host = "host";

    bool fromIsHost;
    bool toIsHost;

    if (from == host) {
      fromIsHost = true;
    } else if (from == device) {
      fromIsHost = false;
    } else {
      gsloge(
          "Unknown cuda memcpy 'from' location [%s]. Expected %s or %s.",
          from.c_str(),
          host.c_str(),
          device.c_str());
      return ERR_RESULT(Status_ParseError);
    }

    if (to == host) {
      toIsHost = true;
    } else if (to == device) {
      toIsHost = false;
    } else {
      gsloge("Unknown cuda memcpy 'to' location [%s]. Expected %s or %s.", to.c_str(), host.c_str(), device.c_str());
    }

    if (fromIsHost && toIsHost) {
      memcpyKind = cudaMemcpyHostToHost;
    } else if (fromIsHost && !toIsHost) {
      memcpyKind = cudaMemcpyHostToDevice;
    } else if (!fromIsHost && !toIsHost) {
      memcpyKind = cudaMemcpyDeviceToDevice;
    } else {
      memcpyKind = cudaMemcpyDeviceToHost;
    }

    Ref<ICudaCommandQueue> commandQueue;
    UNWRAP_OR_FWD_RESULT(
        commandQueue,
        mFactories->getCommandQueueFactory()->getCudaCommandQueue(commandQueueId.c_str()));

    return ResultCast<Node>(createCudaMemcpy(memcpyKind, commandQueue.get()));
  }

  Result<Filter> createCudaMemcpy(cudaMemcpyKind memcpyKind, ICudaCommandQueue* commandQueue) noexcept final {
    return CudaMemcpyFilter::create(memcpyKind, commandQueue, mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(CudaMemcpyFilterFactory);
};

#endif  // GPUSDRPIPELINE_CUDAMEMCPYFILTERFACTORY_H
