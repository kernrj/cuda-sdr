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

#ifndef GPUSDRPIPELINE_IPORTREMAPPINGSOURCE_H
#define GPUSDRPIPELINE_IPORTREMAPPINGSOURCE_H

#include <gpusdrpipeline/filters/Filter.h>

class IPortRemappingSource : public virtual Source {
 public:
  virtual void addPortMapping(size_t outerPort, Source* innerSource, size_t innerSourcePort) noexcept = 0;

  ABSTRACT_IREF(IPortRemappingSource);
};

#endif  // GPUSDRPIPELINE_IPORTREMAPPINGSOURCE_H
