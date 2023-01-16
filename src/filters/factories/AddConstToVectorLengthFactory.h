//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_ADDCONSTTOVECTORLENGTHFACTORY_H
#define GPUSDRPIPELINE_ADDCONSTTOVECTORLENGTHFACTORY_H

#include "../AddConstToVectorLength.h"
#include "filters/FilterFactories.h"

class AddConstToVectorLengthFactory : public IAddConstToVectorLengthFactory {
 public:
  explicit AddConstToVectorLengthFactory(IFactories* factories)
      : mFactories(factories) {}
  ~AddConstToVectorLengthFactory() override = default;
  std::shared_ptr<Filter> createAddConstToVectorLength(
      float addValueToMagnitude,
      int32_t cudaDevice,
      cudaStream_t cudaStream) override {
    return std::make_shared<AddConstToVectorLength>(addValueToMagnitude, cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_ADDCONSTTOVECTORLENGTHFACTORY_H
