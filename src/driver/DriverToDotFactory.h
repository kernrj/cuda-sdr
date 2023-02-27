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

#ifndef GPUSDRPIPELINE_DRIVERTODOTFACTORY_H
#define GPUSDRPIPELINE_DRIVERTODOTFACTORY_H

#include "DriverToDot.h"
#include "driver/IDriverToDiagramFactory.h"

class DriverToDotFactory final : public IDriverToDiagramFactory {
 public:
  [[nodiscard]] Result<IDriverToDiagram> create() const final {
    return makeRefResultNonNull<IDriverToDiagram>(new (std::nothrow) DriverToDot());
  }

  REF_COUNTED(DriverToDotFactory);
};

#endif  // GPUSDRPIPELINE_DRIVERTODOTFACTORY_H
