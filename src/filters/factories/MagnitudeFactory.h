//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_MAGNITUDEFACTORY_H
#define GPUSDRPIPELINE_MAGNITUDEFACTORY_H

#include "../Magnitude.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class MagnitudeFactory : public ICudaFilterFactory {
 public:
  explicit MagnitudeFactory(IFactories* factories)
      : mFactories(factories) {}
  ~MagnitudeFactory() override = default;

  std::shared_ptr<Filter> createFilter(int32_t cudaDevice, cudaStream_t cudaStream) override {
    return std::make_shared<Magnitude>(cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_MAGNITUDEFACTORY_H
