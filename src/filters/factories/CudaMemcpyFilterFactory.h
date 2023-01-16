//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_CUDAMEMCPYFILTERFACTORY_H
#define GPUSDRPIPELINE_CUDAMEMCPYFILTERFACTORY_H

#include "../CudaMemcpyFilter.h"
#include "filters/FilterFactories.h"

class CudaMemcpyFilterFactory : public ICudaMemcpyFilterFactory {
 public:
  explicit CudaMemcpyFilterFactory(IFactories* factories)
      : mFactories(factories) {}
  ~CudaMemcpyFilterFactory() override = default;

  std::shared_ptr<Filter> createCudaMemcpy(cudaMemcpyKind memcpyKind, int32_t cudaDevice, cudaStream_t cudaStream)
      override {
    return std::make_shared<CudaMemcpyFilter>(memcpyKind, cudaDevice, cudaStream, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_CUDAMEMCPYFILTERFACTORY_H
