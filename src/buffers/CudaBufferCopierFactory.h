//
// Created by Rick Kern on 1/4/23.
//

#ifndef GPUSDR_CUDABUFFERCOPIERFACTORY_H
#define GPUSDR_CUDABUFFERCOPIERFACTORY_H

#include "buffers/ICudaBufferCopierFactory.h"
class CudaBufferCopierFactory : public ICudaBufferCopierFactory {
 public:
  ~CudaBufferCopierFactory() override = default;

  std::shared_ptr<IBufferCopier> createBufferCopier(
      int32_t cudaDevice,
      cudaStream_t cudaStream,
      cudaMemcpyKind memcpyKind) override;
};

#endif  // GPUSDR_CUDABUFFERCOPIERFACTORY_H
