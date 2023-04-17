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

#ifndef GPUSDRPIPELINE_FIRFACTORY_H
#define GPUSDRPIPELINE_FIRFACTORY_H

#include "filters/FilterFactories.h"
#include "filters/Fir.h"

class FirFactory final : public IFirFactory {
 public:
  explicit FirFactory(IFactories* factories) noexcept
      : mFactories(factories) {}

  Result<Node> create(const char* jsonParameters) noexcept final {
    nlohmann::json params;
    UNWRAP_OR_FWD_RESULT(params, parseJson(jsonParameters));

    const std::string commandQueueId = params["commandQueue"];
    Ref<ICudaCommandQueue> commandQueue;
    UNWRAP_OR_FWD_RESULT(
        commandQueue,
        mFactories->getCommandQueueFactory()->getCudaCommandQueue(commandQueueId.c_str()));

    SampleType tapType;
    SampleType elementType;

    std::vector<float> taps = params["taps"];

    UNWRAP_OR_FWD_RESULT(tapType, parseSampleType(params["tapType"].get<std::string>()));
    UNWRAP_OR_FWD_RESULT(elementType, parseSampleType(params["elementType"].get<std::string>()));

    return ResultCast<Node>(createFir(
        tapType,
        elementType,
        params["decimation"],
        taps.data(),
        taps.size(),
        commandQueue.get()));
  }

  Result<Filter> createFir(
      SampleType tapType,
      SampleType elementType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      ICudaCommandQueue* commandQueue) noexcept final {
    return Fir::create(tapType, elementType, decimation, taps, tapCount, commandQueue, mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(FirFactory);
};

#endif  // GPUSDRPIPELINE_FIRFACTORY_H
