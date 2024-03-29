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

#ifndef GPUSDRPIPELINE_GSLOG_H
#define GPUSDRPIPELINE_GSLOG_H

#include <gpusdrpipeline/GSDefs.h>
#include <gpusdrpipeline/IRef.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>

using LogLevel = uint32_t;
enum LogLevel_ {
  GSLOG_TRACE,
  GSLOG_DEBUG,
  GSLOG_INFO,
  GSLOG_WARN,
  GSLOG_ERROR,
  GSLOG_FATAL,
};

class ILogger : public virtual IRef {
 public:
  virtual void log(LogLevel level, const char* msgFmt, va_list args) noexcept = 0;

  ABSTRACT_IREF(ILogger);
};

GS_EXPORT [[nodiscard]] const char* gslogLevelName(LogLevel level) noexcept;

GS_EXPORT void gsvlog(LogLevel level, GS_FMT_STR(const char* fmt), va_list args) noexcept;

GS_EXPORT void gslogSetLogger(ILogger* logger) noexcept;
GS_EXPORT void gslogSetVerbosity(LogLevel level) noexcept;

GS_EXPORT GS_FMT_ATTR(1, 2) void gslogt(GS_FMT_STR(const char* fmt), ...) noexcept;               /// Trace-level log
GS_EXPORT GS_FMT_ATTR(1, 2) void gslogd(GS_FMT_STR(const char* fmt), ...) noexcept;               /// Debug-level log
GS_EXPORT GS_FMT_ATTR(1, 2) void gslogi(GS_FMT_STR(const char* fmt), ...) noexcept;               /// Info-level log
GS_EXPORT GS_FMT_ATTR(1, 2) void gslogw(GS_FMT_STR(const char* fmt), ...) noexcept;               /// Warn-level log
GS_EXPORT GS_FMT_ATTR(1, 2) void gsloge(GS_FMT_STR(const char* fmt), ...) noexcept;               /// Error-level log
GS_EXPORT [[noreturn]] GS_FMT_ATTR(1, 2) void gslogf(GS_FMT_STR(const char* fmt), ...) noexcept;  /// Fatal-level log

#endif  // GPUSDRPIPELINE_GSLOG_H
