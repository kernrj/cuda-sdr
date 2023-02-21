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

#ifndef GPUSDRPIPELINE_HACKRFSOURCEFACTORY_H
#define GPUSDRPIPELINE_HACKRFSOURCEFACTORY_H

#include "../HackrfSource.h"
#include "filters/FilterFactories.h"

class HackrfSourceFactory final : public IHackrfSourceFactory {
 public:
  explicit HackrfSourceFactory(IFactories* factories) noexcept
      : mFactories(factories) {}

  Result<IHackrfSource> createHackrfSource(
      int32_t deviceIndex,
      uint64_t frequency,
      double sampleRate,
      size_t maxBufferCountBeforeDropping) noexcept final {
    Ref<IHackrfSource> hackrfSource;
    UNWRAP_OR_FWD_RESULT(
        hackrfSource,
        HackrfSource::create(frequency, sampleRate, maxBufferCountBeforeDropping, mFactories));

    FWD_IN_RESULT_IF_ERR(hackrfSource->selectDeviceByIndex(deviceIndex));

    return makeRefResultNonNull(hackrfSource.get());
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(HackrfSourceFactory);
};

#endif  // GPUSDRPIPELINE_FIRFACTORY_H
