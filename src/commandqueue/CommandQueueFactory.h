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

#ifndef GPUSDRPIPELINE_COMMANDQUEUEFACTORY_H
#define GPUSDRPIPELINE_COMMANDQUEUEFACTORY_H

#include "Factories.h"
#include "commandqueue/ICommandQueueFactory.h"

class CommandQueueFactory final : public virtual ICommandQueueFactory {
 public:
  explicit CommandQueueFactory(IFactories* factories) noexcept;

  Status create(const char* queueId, const char* parameterJson) noexcept final;
  bool exists(const char* queueId) noexcept final;
  Result<ICudaCommandQueue> getCudaCommandQueue(const char* queueId) noexcept final;

 private:
  ConstRef<IFactories> mFactories;

  REF_COUNTED(CommandQueueFactory);
};

#endif  // GPUSDRPIPELINE_COMMANDQUEUEFACTORY_H
