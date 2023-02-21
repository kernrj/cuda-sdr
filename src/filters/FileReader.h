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

#ifndef GPUSDR_FILEREADER_H
#define GPUSDR_FILEREADER_H

#include "filters/Filter.h"

class FileReader final : public virtual Source {
 public:
  explicit FileReader(const char* fileName) noexcept;

  [[nodiscard]] size_t getOutputDataSize(size_t port) noexcept final;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) noexcept final;

  Status readOutput(IBuffer** portOutputBuffers, size_t numPorts) noexcept final;

 private:
  const std::string mFileName;
  FILE* const mFile;

  ~FileReader() final;
  REF_COUNTED_NO_DESTRUCTOR(FileReader);
};

#endif  // GPUSDR_FILEREADER_H
