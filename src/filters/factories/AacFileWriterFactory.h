//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_AACFILEWRITERFACTORY_H
#define GPUSDRPIPELINE_AACFILEWRITERFACTORY_H

#include "../AacFileWriter.h"
#include "Factories.h"
#include "filters/FilterFactories.h"

class AacFileWriterFactory : public IAacFileWriterFactory {
 public:
  explicit AacFileWriterFactory(IFactories* factories)
      : mFactories(factories) {}
  ~AacFileWriterFactory() override = default;
  std::shared_ptr<Sink> createAacFileWriter(const char* outputFileName, int32_t sampleRate, int32_t bitRate) override {
    return std::make_shared<AacFileWriter>(outputFileName, sampleRate, bitRate, mFactories);
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_AACFILEWRITERFACTORY_H
