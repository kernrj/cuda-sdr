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

  Result<Filter> create(
      Modulation modulation,
      float rfSampleRate,
      float fskDeviation,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept final {
    switch (modulation) {
      case Modulation_Am:
        gslogd("Creating quad am demodulator");
        return QuadAmDemod::create(cudaDevice, cudaStream, mFactories);

      case Modulation_Fm: {
        const float gain = getQuadDemodGain(rfSampleRate, fskDeviation);
        gslogd(
            "Creating quad fm demodulator. Sample rate [%f] Deviation [%f] Gain [%f]",
            rfSampleRate,
            fskDeviation,
            gain);

        return QuadFmDemod::create(gain, cudaDevice, cudaStream, mFactories);
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
