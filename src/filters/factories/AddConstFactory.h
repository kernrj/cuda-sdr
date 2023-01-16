//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_ADDCONSTFACTORY_H
#define GPUSDRPIPELINE_ADDCONSTFACTORY_H

#include "../AddConst.h"
#include "filters/FilterFactories.h"

class AddConstFactory : public IAddConstFactory {
 public:
  explicit AddConstFactory(IFactories* factories)
      : mFactories(factories) {}
  ~AddConstFactory() override = default;
  std::shared_ptr<Filter> createAddConst(float addValue, int32_t cudaDevice, cudaStream_t cudaStream) override {
    return std::make_shared<AddConst>(addValue, cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_ADDCONSTFACTORY_H
