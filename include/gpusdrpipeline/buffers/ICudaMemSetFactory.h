//
// Created by Rick Kern on 1/14/23.
//

#ifndef GPUSDRPIPELINE_ICUDAMEMSETFACTORY_H
#define GPUSDRPIPELINE_ICUDAMEMSETFACTORY_H

#include <cuda_runtime.h>
#include <gpusdrpipeline/GSErrors.h>
#include <gpusdrpipeline/buffers/IMemSet.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>

class ICudaMemSetFactory {
 public:
  virtual ~ICudaMemSetFactory() noexcept = default;

  virtual std::shared_ptr<IMemSet> create(int32_t cudaDevice, cudaStream_t cudaStream) noexcept = 0;
};

#endif  // GPUSDRPIPELINE_ICUDAMEMSETFACTORY_H
