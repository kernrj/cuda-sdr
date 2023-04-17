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

#ifndef GPUSDRPIPELINE_QUADDEMODFACTORY_H
#define GPUSDRPIPELINE_QUADDEMODFACTORY_H

#include <filters/QuadAmDemod.h>

#include "../QuadFmDemod.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class QuadDemodFactory final : public IQuadDemodFactory {
 public:
  explicit QuadDemodFactory(IFactories* factories)
      : mFactories(factories) {}

  Result<Node> create(const char* jsonParameters) noexcept final {
    nlohmann::json params;
    UNWRAP_OR_FWD_RESULT(params, parseJson(jsonParameters));

    const auto modulationString = params["modulation"].get<std::string>();
    Modulation modulation;
    struct {
      std::string modulationId;
      Modulation modulation;
    } modulations[] = {
        {.modulationId = "fm", .modulation = Modulation_Fm},
        {.modulationId = "am", .modulation = Modulation_Am},
    };

    if (modulationString == "fm") {
      modulation = Modulation_Fm;
    } else if (modulationString == "am") {
      modulation = Modulation_Am;
    } else {
      std::ostringstream supportedModulations;

      const size_t modulationsLength = sizeof(modulations) / sizeof(modulations[0]);
      for (size_t i = 0; i < modulationsLength; i++) {
        if (i > 0) {
          supportedModulations << ", ";
        }

        supportedModulations << "'" << modulations[i].modulationId << "'";
      }

      gsloge(
          "Modulation [%s] is not supported. Supported modulations: %s",
          modulationString.c_str(),
          supportedModulations.str().c_str());
      return ERR_RESULT(Status_NotFound);
    }

    const auto sampleRate = params["sampleRate"].get<float>();
    const auto fskDeviation = modulation == Modulation_Fm ? params["fskDeviation"].get<float>() : 0.0f;
    const auto commandQueueId = params["commandQueue"].get<std::string>();
    Ref<ICudaCommandQueue> commandQueue;
    UNWRAP_OR_FWD_RESULT(
        commandQueue,
        mFactories->getCommandQueueFactory()->getCudaCommandQueue(commandQueueId.c_str()));

    Ref<Filter> quadDemodFilter;
    UNWRAP_OR_FWD_RESULT(quadDemodFilter, createQuadDemod(modulation, sampleRate, fskDeviation, commandQueue.get()));
    return makeRefResultNonNull<Node>(quadDemodFilter.get());
  }

  Result<Filter> createQuadDemod(
      Modulation modulation,
      float rfSampleRate,
      float fskDeviation,
      ICudaCommandQueue* commandQueue) noexcept final {
    switch (modulation) {
      case Modulation_Am:
        gslogd("Creating quad am demodulator");
        return QuadAmDemod::create(commandQueue, mFactories);

      case Modulation_Fm: {
        const float gain = getQuadDemodGain(rfSampleRate, fskDeviation);
        gslogd(
            "Creating quad fm demodulator. Sample rate [%f] Deviation [%f] Gain [%f]",
            rfSampleRate,
            fskDeviation,
            gain);

        return QuadFmDemod::create(gain, commandQueue, mFactories);
      }

      default:
        gsloge("Modulation [%d] is not supported", modulation);
        return ERR_RESULT(Status_InvalidArgument);
    }
  }

  static float getQuadDemodGain(float inputSampleRate, float fskDeviation) noexcept {
    return inputSampleRate / (2.0f * M_PIf * fskDeviation * 5);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(QuadDemodFactory);
};

#endif  // GPUSDRPIPELINE_QUADDEMODFACTORY_H
