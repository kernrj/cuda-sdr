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

#ifndef GPUSDRPIPELINE_RFTOPCMAUDIO_H
#define GPUSDRPIPELINE_RFTOPCMAUDIO_H

#include <remez/remez.h>

#include "Factories.h"
#include "Modulation.h"

class RfToPcmAudioFactory final : public IRfToPcmAudioFactory {
 public:
  explicit RfToPcmAudioFactory(IFactories* factories) noexcept;

  Result<Node> create(const char* jsonParameters) noexcept final;

  Result<Filter> createRfToPcm(
      float rfSampleRate,
      Modulation modulation,
      size_t rfLowPassDecimation,
      size_t audioLowPassDecimation,
      float centerFrequency,
      float channelFrequency,
      float channelWidth,
      float fskDeviationIfFm,
      float rfLowPassDbAttenuation,
      float audioLowPassDbAttenuation,
      const char* commandQueueId) noexcept final;

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(RfToPcmAudioFactory);
};

#endif  // GPUSDRPIPELINE_RFTOPCMAUDIOFACTORY_H
