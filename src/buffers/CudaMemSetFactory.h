//
// Created by Rick Kern on 1/14/23.
//

#ifndef GPUSDRPIPELINE_CUDAMEMSETFACTORY_H
#define GPUSDRPIPELINE_CUDAMEMSETFACTORY_H

#include "CudaMemSet.h"
#include "buffers/ICudaMemSetFactory.h"

class CudaMemSetFactory : public ICudaMemSetFactory {
 public:
  ~CudaMemSetFactory() override = default;

  std::shared_ptr<IMemSet> create(int32_t cudaDevice, cudaStream_t cudaStream) noexcept override {
    return std::make_shared<CudaMemSet>(cudaDevice, cudaStream);
  }
};

#endif  // GPUSDRPIPELINE_CUDAMEMSETFACTORY_H
