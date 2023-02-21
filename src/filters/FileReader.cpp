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

#include "FileReader.h"

#include <cerrno>
#include <cstring>

#include "GSErrors.h"

using namespace std;

FileReader::FileReader(const char* fileName) noexcept
    : mFileName(fileName),
      mFile(fopen64(fileName, "rb")) {
  if (mFile == nullptr) {
    GS_FAIL("Failed to open file [" << fileName << "]: Error [" << errno << "]: " << strerror(errno));
  }
}

FileReader::~FileReader() {
  if (mFile != nullptr) {
    fclose(mFile);
  }
}

size_t FileReader::getOutputDataSize(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return 65536;
}

size_t FileReader::getOutputSizeAlignment(size_t port) noexcept {
  GS_REQUIRE_OR_RET_FMT(0 == port, 0, "Output port [%zu] is out of range", port);
  return 1;
}

Status FileReader::readOutput(IBuffer** portOutputBuffers, size_t portCount) noexcept {
  GS_REQUIRE_OR_RET_STATUS(portCount != 0, "One output port is required");

  IBuffer* outputBuffer = portOutputBuffers[0];
  size_t readByteCount = fread(outputBuffer->writePtr(), 1, outputBuffer->range()->remaining(), mFile);

  FWD_IF_ERR(outputBuffer->range()->increaseEndOffset(readByteCount));

  return Status_Success;
}
