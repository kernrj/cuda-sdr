//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_MULTIPLYFACTORY_H
#define GPUSDRPIPELINE_MULTIPLYFACTORY_H

#include "../Multiply.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class MultiplyFactory : public ICudaFilterFactory {
 public:
  explicit MultiplyFactory(IFactories* factories)
      : mFactories(factories) {}
  ~MultiplyFactory() override = default;

  std::shared_ptr<Filter> createFilter(int32_t cudaDevice, cudaStream_t cudaStream) override {
    return std::make_shared<MultiplyCcc>(cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_MULTIPLYFACTORY_H
