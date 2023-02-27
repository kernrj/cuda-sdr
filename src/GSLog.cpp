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

  gLogger = new (nothrow) FileLogger(stderr, /* closeWhenDone= */ false);
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
      return "TRACE";
    case GSLOG_DEBUG:
      return "DEBUG";
    case GSLOG_INFO:
      return "INFO";
    case GSLOG_WARN:
      return "WARN";
    case GSLOG_ERROR:
      return "ERROR";
    case GSLOG_FATAL:
      return "FATAL";
    default:
      return "UNKNOWN";
  }
}

GS_C_LINKAGE void gslogSetLogger(ILogger* logger) noexcept {
  ensureInit();

  lock_guard<mutex> l(gLogLock);
  gLogger = logger;
}

GS_C_LINKAGE void gslogSetVerbosity(LogLevel logLevel) noexcept {
  ensureInit();

  lock_guard<mutex> l(gLogLock);
  gLogLevel = logLevel;
}

static void gslog(LogLevel level, GS_FMT_STR(const char* fmt), ...) noexcept {
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

GS_C_LINKAGE void gslogt(GS_FMT_STR(const char* fmt), ...) noexcept {
  if (gLogLevel > GSLOG_TRACE) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  getLogger()->log(GSLOG_TRACE, fmt, args);
  va_end(args);
}

GS_C_LINKAGE void gslogd(GS_FMT_STR(const char* fmt), ...) noexcept {
  if (gLogLevel > GSLOG_DEBUG) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  getLogger()->log(GSLOG_DEBUG, fmt, args);
  va_end(args);
}

GS_C_LINKAGE void gslogi(GS_FMT_STR(const char* fmt), ...) noexcept {
  if (gLogLevel > GSLOG_INFO) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  getLogger()->log(GSLOG_INFO, fmt, args);
  va_end(args);
}

GS_C_LINKAGE void gslogw(GS_FMT_STR(const char* fmt), ...) noexcept {
  if (gLogLevel > GSLOG_WARN) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  getLogger()->log(GSLOG_WARN, fmt, args);
  va_end(args);
}

GS_C_LINKAGE void gsloge(GS_FMT_STR(const char* fmt), ...) noexcept {
  if (gLogLevel > GSLOG_ERROR) {
    return;
  }

  va_list args;
  va_start(args, fmt);
  getLogger()->log(GSLOG_ERROR, fmt, args);
  va_end(args);
}

GS_C_LINKAGE void gslogf(GS_FMT_STR(const char* fmt), ...) noexcept {
  va_list args;
  va_start(args, fmt);
  getLogger()->log(GSLOG_FATAL, fmt, args);
  va_end(args);

  abort();
}
