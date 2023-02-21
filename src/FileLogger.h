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

#ifndef GPUSDRPIPELINE_FILELOGGER_H
#define GPUSDRPIPELINE_FILELOGGER_H

#include <chrono>

#include "GSLog.h"

class FileLogger final : public ILogger {
 public:
  FileLogger(FILE* file, bool closeWhenDone) noexcept;

  void log(LogLevel logLevel, const char* msgFmt, va_list args) noexcept final;

 private:
  FILE* const mFile;
  const bool mCloseWhenDone;

 private:
  ~FileLogger() final;
  REF_COUNTED_NO_DESTRUCTOR(FileLogger);
};

#endif  // GPUSDRPIPELINE_FILELOGGER_H
