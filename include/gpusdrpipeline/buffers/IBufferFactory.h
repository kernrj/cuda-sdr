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

#ifndef GPUSDR_IBUFFERFACTORY_H
#define GPUSDR_IBUFFERFACTORY_H

#include <gpusdrpipeline/IRef.h>
#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/buffers/IBuffer.h>
#include <gpusdrpipeline/buffers/IRelocatable.h>
#include <gpusdrpipeline/buffers/IResizable.h>

class IBufferFactory : public virtual IRef {
 public:
  [[nodiscard]] virtual Result<IBuffer> createBuffer(size_t size) noexcept = 0;

  ABSTRACT_IREF(IBufferFactory);
};

#endif  // GPUSDR_IBUFFERFACTORY_H
