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

#include "FileLogger.h"

#include <sys/time.h>

#include <cinttypes>
#include <cstdio>
#include <ctime>

static const char* paddedLogLevelName(LogLevel logLevel) {
  switch (logLevel) {
    case GSLOG_TRACE: return "TRACE";
    case GSLOG_DEBUG: return "DEBUG";
    case GSLOG_INFO: return " INFO";
    case GSLOG_WARN: return " WARN";
    case GSLOG_ERROR: return "ERROR";
    case GSLOG_FATAL: return "FATAL";
    default: return "?????";
  }
}

FileLogger::FileLogger(FILE* file, bool closeWhenDone) noexcept
    : mFile(file),
      mCloseWhenDone(closeWhenDone) {}

FileLogger::~FileLogger() {
  if (mCloseWhenDone && mFile != nullptr) {
    fclose(mFile);
  }
}

void FileLogger::log(LogLevel logLevel, const char* msgFmt, va_list args) noexcept {
  if (mFile == nullptr) {
    fprintf(stderr, "(NULL LOGGER) - ");
    vfprintf(stderr, msgFmt, args);
    putc('\n', stderr);
    return;
  }

  timespec now;
  clock_gettime(CLOCK_REALTIME, &now);

  tm nowTm;

#ifdef _MSC_VER
  ::gmtime_s(&nowTm, &now.tv_sec);
#else
  ::gmtime_r(&now.tv_sec, &nowTm);
#endif

  /*
   * Min is 26:
   * Year (4) + month (2) + day (2) + hour (2) + minute (2) + second (2) + microseconds (6) + literals "-- ::" (5)
   *   + null-terminator (1)
   */
  static char timeStr[32];
  strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", &nowTm);
  int64_t microseconds = now.tv_nsec / 1000;
  fprintf(mFile, "%s.%06" PRId64 " - %s - ", timeStr, microseconds, paddedLogLevelName(logLevel));
  timeStr[sizeof(timeStr) - 1] = 0;

  vfprintf(mFile, msgFmt, args);
  fputc('\n', mFile);
}
