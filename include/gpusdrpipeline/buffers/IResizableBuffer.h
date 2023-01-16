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

#ifndef GPUSDR_IRESIZABLEBUFFER_H
#define GPUSDR_IRESIZABLEBUFFER_H

#include <gpusdrpipeline/buffers/IBuffer.h>
#include <gpusdrpipeline/buffers/IResizable.h>

class IResizableBuffer : public virtual IBuffer, public IResizable {
 public:
  ~IResizableBuffer() override = default;

  void ensureMinSize(size_t minSize) {
    if (range()->capacity() < minSize) {
      resize(minSize, nullptr);
    }
  }
};

#endif  // GPUSDR_IRESIZABLEBUFFER_H