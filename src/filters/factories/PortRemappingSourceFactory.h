//
// Created by Rick Kern on 4/15/23.
//

#ifndef GPUSDRPIPELINE_PORTREMAPPINGSOURCEFACTORY_H
#define GPUSDRPIPELINE_PORTREMAPPINGSOURCEFACTORY_H

#include "filters/FilterFactories.h"
#include "filters/PortRemappingSource.h"

class PortRemappingSourceFactory final : public IPortRemappingSourceFactory {
 public:
  explicit PortRemappingSourceFactory(IFactories* factories) : mFactories(factories) {}

  Result<IPortRemappingSource> create() noexcept final {
    return PortRemappingSource::create(mFactories);
  }

 private:
  ConstRef<IFactories> mFactories;
  REF_COUNTED(PortRemappingSourceFactory);
};

#endif  // GPUSDRPIPELINE_PORTREMAPPINGSOURCEFACTORY_H
