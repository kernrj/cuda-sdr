//
// Created by Rick Kern on 1/6/23.
//

#include "FileReader.h"

#include <cerrno>
#include <cstring>

#include "GSErrors.h"

using namespace std;

FileReader::FileReader(const string& fileName)
    : mFileName(fileName),
      mFile(fopen64(fileName.c_str(), "rb")) {
  if (mFile == nullptr) {
    THROW("Failed to open file [" << fileName << "]: Error [" << errno << "]: " << strerror(errno));
  }
}

FileReader::~FileReader() noexcept {
  if (mFile != nullptr) {
    fclose(mFile);
  }
}

size_t FileReader::getOutputDataSize(size_t port) { return 65536; }
size_t FileReader::getOutputSizeAlignment(size_t port) { return 1; }

void FileReader::readOutput(const vector<shared_ptr<IBuffer>>& portOutputs) {
  if (portOutputs.empty()) {
    THROW("One output port is required");
  }

  const shared_ptr<IBuffer>& outputBuffer = portOutputs[0];
  size_t readByteCount = fread(outputBuffer->writePtr(), 1, outputBuffer->range()->remaining(), mFile);

  outputBuffer->range()->increaseEndOffset(readByteCount);
}
