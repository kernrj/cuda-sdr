//
// Created by Rick Kern on 1/8/23.
//

#ifndef GPUSDRPIPELINE_HACKRFSOURCEFACTORY_H
#define GPUSDRPIPELINE_HACKRFSOURCEFACTORY_H

#include "../HackrfSource.h"
#include "filters/FilterFactories.h"

class HackrfSourceFactory : public IHackrfSourceFactory {
 public:
  explicit HackrfSourceFactory(IFactories* factories)
      : mFactories(factories) {}
  ~HackrfSourceFactory() override = default;

  std::shared_ptr<IHackrfSource> createHackrfSource(
      int32_t deviceIndex,
      uint64_t frequency,
      double sampleRate,
      size_t maxBufferCountBeforeDropping) override {
    auto hackrfSource = std::make_shared<HackrfSource>(frequency, sampleRate, maxBufferCountBeforeDropping, mFactories);
    hackrfSource->selectDeviceByIndex(deviceIndex);

    return hackrfSource;
  }

 private:
  IFactories* const mFactories;
};

#endif  // GPUSDRPIPELINE_FIRFACTORY_H
