//
// Created by Rick Kern on 1/6/23.
//

#ifndef GPUSDR_FILEREADER_H
#define GPUSDR_FILEREADER_H

#include "filters/Filter.h"

class FileReader : public Source {
 public:
  explicit FileReader(const std::string& fileName);
  ~FileReader() noexcept override;

  [[nodiscard]] size_t getOutputDataSize(size_t port) override;
  [[nodiscard]] size_t getOutputSizeAlignment(size_t port) override;

  void readOutput(const std::vector<std::shared_ptr<IBuffer>>& portOutputs) override;

 private:
  const std::string mFileName;
  FILE* const mFile;
};

#endif  // GPUSDR_FILEREADER_H
