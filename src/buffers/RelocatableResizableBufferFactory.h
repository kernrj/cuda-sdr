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

#ifndef GPUSDR_RELOCATABLERESIZABLEBUFFERFACTORY_H
#define GPUSDR_RELOCATABLERESIZABLEBUFFERFACTORY_H

#include "Factories.h"
#include "buffers/IAllocator.h"
#include "buffers/IBufferCopier.h"
#include "buffers/IBufferRangeFactory.h"
#include "buffers/IRelocatableResizableBufferFactory.h"

class RelocatableResizableBufferFactory final : public IRelocatableResizableBufferFactory {
 public:
  explicit RelocatableResizableBufferFactory(
      IAllocator* allocator,
      const IBufferCopier* bufferCopier,
      const IBufferRangeFactory* bufferRangeFactory) noexcept;
  Result<IRelocatableResizableBuffer> createRelocatableBuffer(size_t size) const noexcept final;

 private:
  ConstRef<IAllocator> mAllocator;
  ConstRef<const IBufferCopier> mBufferCopier;
  ConstRef<const IBufferRangeFactory> mBufferRangeFactory;

  REF_COUNTED(RelocatableResizableBufferFactory);
};

#endif  // GPUSDR_RELOCATABLERESIZABLEBUFFERFACTORY_H
