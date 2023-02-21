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

#include "GSLog.h"

#include <cstdio>
#include <mutex>

#include "FileLogger.h"

using namespace std;

static Ref<ILogger> gLogger;
static LogLevel gLogLevel;

static mutex gLogLock;
static once_flag gInitOnceFlag;

static void initLogger() noexcept {
#ifdef DEBUG
  gLogLevel = GSLOG_DEBUG;
#else
  gLogLevel = GSLOG_INFO;
#endif

  gLogger = new (nothrow) FileLogger(stdout, /* closeWhenDone= */ false);
}

static void ensureInit() noexcept { call_once(gInitOnceFlag, initLogger); }

static ImmutableRef<ILogger> getLogger() noexcept {
  lock_guard<mutex> l(gLogLock);
  call_once(gInitOnceFlag, initLogger);

  return gLogger;
}

GS_C_LINKAGE const char* gslogLevelName(LogLevel level) noexcept {
  switch (level) {
    case GSLOG_TRACE:
      return "trace";
    case GSLOG_DEBUG:
      return "debug";
    case GSLOG_INFO:
      return "info";
    case GSLOG_WARN:
      return "warn";
    case GSLOG_ERROR:
      return "error";
    case GSLOG_FATAL:
      return "fatal";
    default:
      return "unknown_log_level";
  }
}

GS_C_LINKAGE void gsSetLogger(ILogger* logger) noexcept {
  ensureInit();

  lock_guard<mutex> l(gLogLock);
  gLogger = logger;
}

GS_C_LINKAGE void gsSetLogVerbosity(LogLevel logLevel) noexcept {
  ensureInit();

  lock_guard<mutex> l(gLogLock);
  gLogLevel = logLevel;
}

GS_C_LINKAGE void gslog(LogLevel level, GS_FMT_STR(const char* fmt), ...) noexcept {
  if (level < gLogLevel) {
    return;
  }

  va_list args;
  va_start(args, fmt);

  getLogger()->log(level, fmt, args);

  va_end(args);
}

GS_C_LINKAGE void gsvlog(LogLevel level, GS_FMT_STR(const char* fmt), va_list args) noexcept {
  if (level < gLogLevel) {
    return;
  }

  getLogger()->log(level, fmt, args);
}
