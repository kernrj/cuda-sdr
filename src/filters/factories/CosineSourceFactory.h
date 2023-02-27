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
  Result<Source> createCosineSource(
      SampleType sampleType,
      float sampleRate,
      float frequency,
      int32_t cudaDevice,
      cudaStream_t cudaStream) noexcept final {
    switch (sampleType) {
      case SampleType_FloatComplex:
        return makeRefResultNonNull<Source>(new (std::nothrow) ComplexCosineSource(sampleRate, frequency, cudaDevice, cudaStream));

      case SampleType_Float:
        return makeRefResultNonNull<Source>(new (std::nothrow) CosineSource(sampleRate, frequency, cudaDevice, cudaStream));

      default:
        gsloge("Sample type [%u] is not supported for cosine", sampleType);
        return ERR_RESULT(Status_InvalidArgument);
    }
  }

  REF_COUNTED(CosineSourceFactory);
};

#endif  // GPUSDRPIPELINE_COSINESOURCEFACTORY_H
