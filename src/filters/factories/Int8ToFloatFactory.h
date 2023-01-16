//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_INT8TOFLOATFACTORY_H
#define GPUSDRPIPELINE_INT8TOFLOATFACTORY_H

#include "../Int8ToFloat.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class Int8ToFloatFactory : public ICudaFilterFactory {
 public:
  explicit Int8ToFloatFactory(IFactories* factories)
      : mFactories(factories) {}
  ~Int8ToFloatFactory() override = default;

  std::shared_ptr<Filter> createFilter(int32_t cudaDevice, cudaStream_t cudaStream) override {
    return std::make_shared<Int8ToFloat>(cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_INT8TOFLOATFACTORY_H
