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

#ifndef GPUSDRPIPELINE_COSINESOURCEFACTORY_H
#define GPUSDRPIPELINE_COSINESOURCEFACTORY_H

#include "../ComplexCosineSource.h"
#include "../CosineSource.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class CosineSourceFactory final : public ICosineSourceFactory {
 public:
  explicit CosineSourceFactory(IFactories* factories)
      : mFactories(factories) {}

  Result<Node> create(const char* jsonParameters) noexcept final {
    nlohmann::json params;
    UNWRAP_OR_FWD_RESULT(params, parseJson(jsonParameters));

    const std::string commandQueueId = params["commandQueue"];
    Ref<ICudaCommandQueue> commandQueue;
    UNWRAP_OR_FWD_RESULT(
        commandQueue,
        mFactories->getCommandQueueFactory()->getCudaCommandQueue(commandQueueId.c_str()));

    SampleType sampleType;
    UNWRAP_OR_FWD_RESULT(sampleType, parseSampleType(params["sampleType"]));

    return ResultCast<Node>(createCosineSource(
        sampleType,
        params["sampleRate"],
        params["frequency"],
        commandQueue.get()));
  }

  Result<Source> createCosineSource(
      SampleType sampleType,
      float sampleRate,
      float frequency,
      ICudaCommandQueue* commandQueue) noexcept final {
    switch (sampleType) {
      case SampleType_FloatComplex:
        return
            ComplexCosineSource::create(sampleRate, frequency, commandQueue, mFactories);

      case SampleType_Float:
        return CosineSource::create(sampleRate, frequency, commandQueue, mFactories);

      default:
        gsloge("Sample type [%u] is not supported for cosine", sampleType);
        return ERR_RESULT(Status_InvalidArgument);
    }
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(CosineSourceFactory);
};

#endif  // GPUSDRPIPELINE_COSINESOURCEFACTORY_H
