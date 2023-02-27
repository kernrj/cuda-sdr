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

#ifndef GPUSDRPIPELINE_IREADBYTECOUNTMONITOR_H
#define GPUSDRPIPELINE_IREADBYTECOUNTMONITOR_H

#include <gpusdrpipeline/filters/Filter.h>

#include <cstddef>

class IReadByteCountMonitor : public Filter {
 public:
  [[nodiscard]] virtual size_t getByteCountRead(size_t port) noexcept = 0;

  ABSTRACT_IREF(IReadByteCountMonitor);
};

#endif  // GPUSDRPIPELINE_IREADBYTECOUNTMONITOR_H
