//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_COSINESOURCEFACTORY_H
#define GPUSDRPIPELINE_COSINESOURCEFACTORY_H

#include "../CosineSource.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class CosineSourceFactory : public ICosineSourceFactory {
 public:
  ~CosineSourceFactory() override = default;
  std::shared_ptr<Source> createCosineSource(
      float sampleRate,
      float frequency,
      int32_t cudaDevice,
      cudaStream_t cudaStream) override {
    return std::make_shared<CosineSource>(sampleRate, frequency, cudaDevice, cudaStream);
  }
};

#endif  // GPUSDRPIPELINE_COSINESOURCEFACTORY_H
