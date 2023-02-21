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

#ifndef GPUSDRPIPELINE_IDRIVERTODIAGRAM_H
#define GPUSDRPIPELINE_IDRIVERTODIAGRAM_H

#include <gpusdrpipeline/Result.h>
#include <gpusdrpipeline/driver/IDriver.h>

class IDriverToDiagram : public virtual IRef {
 public:
  /**
   * Generates a text-format diagram.
   *
   * @param driver Convert this to a diagram
   * @param name
   * @param diagramBuffer Populated with the diagram. It may be null if diagramSize is also 0.
   * @param diagramSize The size of the diagramBuffer.
   * @return The number of bytes the full diagram string occupies, not including any null-terminator value.
   */
  virtual Result<size_t> convertToDot(
      IDriver* driver,
      const char* name,
      char* diagramBuffer,
      size_t diagramSize) noexcept = 0;

  ABSTRACT_IREF(IDriverToDiagram);
};

#endif  // GPUSDRPIPELINE_IDRIVERTODIAGRAM_H
