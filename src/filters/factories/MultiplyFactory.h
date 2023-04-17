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

#ifndef GPUSDRPIPELINE_MULTIPLYFACTORY_H
#define GPUSDRPIPELINE_MULTIPLYFACTORY_H

#include <nlohmann/json.hpp>

#include "../Multiply.h"
#include "Factories.h"
#include "GSLog.h"
#include "ParseJson.h"
#include "commandqueue/ICudaCommandQueue.h"
#include "filters/FilterFactories.h"

class MultiplyFactory final : public ICudaFilterFactory {
 public:
  explicit MultiplyFactory(IFactories* factories)
      : mFactories(factories) {}

  Result<Node> create(const char* jsonParameters) noexcept final {
    DO_OR_RET_ERR_RESULT({
      nlohmann::json parameters;
      UNWRAP_OR_FWD_RESULT(parameters, parseJson(jsonParameters));

      static const std::string commandQueueKey = "commandQueue";
      GS_REQUIRE_OR_RET_RESULT_FMT(parameters.contains(commandQueueKey), "%s must be set", commandQueueKey.c_str());

      std::string commandQueueId = parameters[commandQueueKey].get<std::string>();

      Ref<ICudaCommandQueue> commandQueue;
      UNWRAP_OR_FWD_RESULT(commandQueue, mFactories->getCommandQueueFactory()->getCudaCommandQueue(commandQueueId.c_str()));

      return ResultCast<Node>(createFilter(commandQueue.get()));
    });
  }

  Result<Filter> createFilter(ICudaCommandQueue* commandQueue) noexcept final {
    return MultiplyCcc::create(commandQueue, mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(MultiplyFactory);
};

#endif  // GPUSDRPIPELINE_MULTIPLYFACTORY_H
