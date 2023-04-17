/*
 * Copyright 2023 Rick Kern <kernrj@gmail.com>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef GPUSDRPIPELINE_ICOMMANDQUEUEFACTORY_H
#define GPUSDRPIPELINE_ICOMMANDQUEUEFACTORY_H

#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/commandqueue/ICommandQueue.h>

/*
 * Things done on a CQ:
 * Run a kernel
 * Allocate memory
 *
 * CPU can wait on a CQ
 *
 * Events
 *
 * A CQ is a thread:
 * - Could be a cuda stream, opencl command queue, CPU thread
 *
 * A CQ is tied to a "device"
 * - GPU
 * - CPU (could be multiple in a system)
 *
 * Things tied to a device:
 * - CQ
 * - Memory allocation (device may (e.g. CPU/GPU) or may not (e.g. FPGA) support memory allocation.)
 *
 * Memory can be allocated on a device, but can be scheduled on a CQ.
 *
 *
 *
 */

class ICommandQueueFactory : public virtual IRef {
 public:
  /**
   * Creates the command queue. Subclasses of ICommandQueueFactory provide getter methods.
   * @param queueId A unique ID for the new queue.
   * @return Status_Success if the command queue was created, or an error code otherwise.
   */
  [[nodiscard]] virtual Status create(const char* queueId, const char* parameterJson) noexcept = 0;
  [[nodiscard]] virtual bool exists(const char* queueId) noexcept = 0;

  [[nodiscard]] virtual Result<ICudaCommandQueue> getCudaCommandQueue(const char* queueId) noexcept = 0;

  ABSTRACT_IREF(ICommandQueueFactory);
};

#endif  // GPUSDRPIPELINE_ICOMMANDQUEUEFACTORY_H
