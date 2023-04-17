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

#include <nlohmann/json.hpp>

#include "../HackrfSource.h"
#include "ParseJson.h"
#include "filters/FilterFactories.h"

class HackrfSourceFactory final : public IHackrfSourceFactory {
 public:
  explicit HackrfSourceFactory(IFactories* factories) noexcept
      : mFactories(factories) {}

  Result<Node> create(const char* jsonParameters) noexcept final {
    nlohmann::json params;

    UNWRAP_OR_FWD_RESULT(params, parseJson(jsonParameters));

    const int32_t deviceIndex = params["deviceIndex"].get<int32_t>();
    const uint64_t centerFrequency = params["centerFrequency"].get<uint32_t>();
    const double sampleRate = params["sampleRate"].get<float>();
    size_t maxBufferCountBeforeDropping = 3;
    const std::string keyMaxBufferCountBeforeDropping = "maxBufferCountBeforeDropping";
    if (params.contains(keyMaxBufferCountBeforeDropping)) {
      UNWRAP_OR_FWD_RESULT(maxBufferCountBeforeDropping, getJsonOrErr<size_t>(params, keyMaxBufferCountBeforeDropping));
    }

    return ResultCast<Node>(createHackrfSource(deviceIndex, centerFrequency, sampleRate, maxBufferCountBeforeDropping));
  }

  Result<IHackrfSource> createHackrfSource(
      int32_t deviceIndex,
      uint64_t centerFrequency,
      double sampleRate,
      size_t maxBufferCountBeforeDropping) noexcept final {
    StealableRef<IHackrfSource> hackrfSource;
    UNWRAP_OR_FWD_RESULT(
        hackrfSource,
        HackrfSource::create(centerFrequency, sampleRate, maxBufferCountBeforeDropping, mFactories));

    FWD_IN_RESULT_IF_ERR(hackrfSource->selectDeviceByIndex(deviceIndex));

    return makeRefResultNonNull(hackrfSource.steal());
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(HackrfSourceFactory);
};

#endif  // GPUSDRPIPELINE_FIRFACTORY_H
