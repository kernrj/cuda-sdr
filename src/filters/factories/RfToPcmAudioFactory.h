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

#ifndef GPUSDRPIPELINE_RFTOPCMAUDIOFACTORY_H
#define GPUSDRPIPELINE_RFTOPCMAUDIOFACTORY_H

#include <filters/RfToPcmAudio.h>

#include "Factories.h"
#include "filters/FilterFactories.h"

class RfToPcmAudioFactory final : public IRfToPcmAudioFactory {
 public:
  explicit RfToPcmAudioFactory(IFactories* factories)
      : mFactories(factories) {}

  Result<Filter> create(
      float rfSampleRate,
      Modulation modulation,
      size_t rfLowPassDecim,
      size_t audioLowPassDecim,
      float centerFrequency,
      float channelFrequency,
      float channelWidth,
      float fskDevationIfFm,
      float rfLowPassDbAttenuation,
      float audioLowPassDbAttenuation,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept final {
    return RfToPcmAudio::create(
        rfSampleRate,
        modulation,
        rfLowPassDecim,
        audioLowPassDecim,
        centerFrequency,
        channelFrequency,
        channelWidth,
        fskDevationIfFm,
        rfLowPassDbAttenuation,
        audioLowPassDbAttenuation,
        cudaDevice,
        cudaStream,
        mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(RfToPcmAudioFactory);
};
#endif  // GPUSDRPIPELINE_RFTOPCMAUDIOFACTORY_H
