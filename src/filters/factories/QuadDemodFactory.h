//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_QUADDEMODFACTORY_H
#define GPUSDRPIPELINE_QUADDEMODFACTORY_H

#include "../QuadDemod.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class QuadDemodFactory : public IQuadDemodFactory {
 public:
  explicit QuadDemodFactory(IFactories* factories)
      : mFactories(factories) {}
  ~QuadDemodFactory() override = default;

  std::shared_ptr<Filter> create(float gain, int32_t cudaDevice, cudaStream_t cudaStream) override {
    return std::make_shared<QuadDemod>(gain, cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_QUADDEMODFACTORY_H
