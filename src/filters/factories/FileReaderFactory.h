//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_FILEREADERFACTORY_H
#define GPUSDRPIPELINE_FILEREADERFACTORY_H

#include "../FileReader.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class FileReaderFactory : public IFileReaderFactory {
 public:
  ~FileReaderFactory() override = default;
  std::shared_ptr<Source> createFileReader(const char* fileName) override {
    return std::make_shared<FileReader>(fileName);
  }
};

#endif  // GPUSDRPIPELINE_FILEREADERFACTORY_H
