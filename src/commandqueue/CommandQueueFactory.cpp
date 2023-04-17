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

#include "CommandQueueFactory.h"

#include <mutex>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "GSErrors.h"
#include "ParseJson.h"
#include "commandqueue/ICudaCommandQueue.h"

using namespace std;

static unordered_map<string, Ref<ICudaCommandQueue>> gCudaCommandQueues;
static recursive_mutex gMutex;

CommandQueueFactory::CommandQueueFactory(IFactories* factories) noexcept
    : mFactories(factories) {}

Status CommandQueueFactory::create(const char* queueId, const char* parameterJson) noexcept {
  lock_guard<recursive_mutex> l(gMutex);

  nlohmann::json parameters;
  UNWRAP_OR_FWD_STATUS(parameters, parseJson(parameterJson, nlohmann::json::value_t::object));
  GS_REQUIRE_OR_RET_STATUS_FMT(
      parameters.contains("queueType"),
      "Required queueType parameter is missing from JSON: %s\nParsed as JSON: %s",
      parameterJson,
      parameters.dump().c_str());
  auto queueTypeJson = parameters["queueType"];
  GS_REQUIRE_OR_RET_STATUS_FMT(
      queueTypeJson.is_string(),
      "queueType must be a string, but is [%s]",
      queueTypeJson.type_name());

  string queueType = queueTypeJson.get<string>();

  if (queueType == "cuda") {
    ICudaCommandQueueFactory* cudaFactory = mFactories->getCudaCommandQueueFactory();
    if (exists(queueId)) {
      return Status_Success;
    }

    static constexpr int64_t defaultCudaDevice = 0;
    const int32_t cudaDevice = tryGetJson<int32_t>(parameters, "cudaDevice", defaultCudaDevice);
    Ref<ICudaCommandQueue> commandQueue;
    UNWRAP_OR_FWD_STATUS(commandQueue, cudaFactory->create(cudaDevice));

    auto insertIt = gCudaCommandQueues.emplace(queueId, commandQueue);
    if (!insertIt.second) {
      gsloge("Failed to add CUDA queue [%s]", queueId);
      return Status_UnknownError;
    }

    return Status_Success;
  } else {
    gsloge("Unknown queueType [%s]", queueType.c_str());
    return Status_NotFound;
  }

  return Status_UnknownError;
}

bool CommandQueueFactory::exists(const char* queueId) noexcept {
  lock_guard<recursive_mutex> l(gMutex);

  return gCudaCommandQueues.find(queueId) != gCudaCommandQueues.end();
}

Result<ICudaCommandQueue> CommandQueueFactory::getCudaCommandQueue(const char* queueId) noexcept {
  lock_guard<recursive_mutex> l(gMutex);

  auto it = gCudaCommandQueues.find(queueId);
  if (it == gCudaCommandQueues.end()) {
    return ERR_RESULT(Status_NotFound);
  }

  return makeRefResultNonNull(it->second.get());
}
