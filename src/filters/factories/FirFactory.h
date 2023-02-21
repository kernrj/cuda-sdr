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

  Result<Filter> createFir(
      SampleType tapType,
      SampleType elementType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept final {
    return Fir::create(tapType, elementType, decimation, taps, tapCount, cudaDevice, cudaStream, mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(FirFactory);
};

#endif  // GPUSDRPIPELINE_FIRFACTORY_H
