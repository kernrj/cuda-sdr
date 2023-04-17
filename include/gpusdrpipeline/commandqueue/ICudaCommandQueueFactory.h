//
// Created by Rick Kern on 3/30/23.
//

#ifndef GPUSDRPIPELINE_ICUDACOMMANDQUEUEFACTORY_H
#define GPUSDRPIPELINE_ICUDACOMMANDQUEUEFACTORY_H

#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/commandqueue/ICudaCommandQueue.h>

class ICudaCommandQueueFactory : public IRef {
 public:
  virtual Result<ICudaCommandQueue> create(int32_t cudaDevice) noexcept = 0;

  ABSTRACT_IREF(ICudaCommandQueueFactory);
};

#endif  // GPUSDRPIPELINE_ICUDACOMMANDQUEUEFACTORY_H
