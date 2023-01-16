//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_FIRFACTORY_H
#define GPUSDRPIPELINE_FIRFACTORY_H

#include "filters/FilterFactories.h"
#include "filters/Fir.h"

class FirFactory : public IFirFactory {
 public:
  FirFactory(IFactories* factories)
      : mFactories(factories) {}
  ~FirFactory() override = default;
  std::shared_ptr<Filter> createFir(
      FirType firType,
      size_t decimation,
      const float* taps,
      size_t tapCount,
      int32_t cudaDevice,
      cudaStream_t cudaStream) override {
    return std::make_shared<Fir>(firType, decimation, taps, tapCount, cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_FIRFACTORY_H
