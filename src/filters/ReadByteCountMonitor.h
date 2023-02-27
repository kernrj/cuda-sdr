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
#ifndef GPUSDRPIPELINE_READBYTECOUNTMONITOR_H
#define GPUSDRPIPELINE_READBYTECOUNTMONITOR_H

#include <vector>

#include "Result.h"
#include "filters/IReadByteCountMonitor.h"

class ReadByteCountMonitor final : public IReadByteCountMonitor {
 public:
  explicit ReadByteCountMonitor(Filter* filter) noexcept;

  size_t getByteCountRead(size_t port) noexcept final;

  [[nodiscard]] Result<IBuffer> requestBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] Status commitBuffer(size_t port, size_t byteCount) noexcept final;
  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;
  [[nodiscard]] Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;
  [[nodiscard]] size_t preferredInputBufferSize(size_t port) noexcept final;

 private:
  ConstRef<Filter> mFilter;
  std::vector<size_t> mUsedDataBeforeReadOnLastIteration;
  std::vector<size_t> mTotalByteCountRead;

  REF_COUNTED(ReadByteCountMonitor);
};

#endif  // GPUSDRPIPELINE_READBYTECOUNTMONITOR_H
