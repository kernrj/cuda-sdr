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

#ifndef GPUSDRPIPELINE_GSDEFS_H
#define GPUSDRPIPELINE_GSDEFS_H

#include <gpusdrpipeline/gpusdrpipeline_export.h>

// Used on abstract classes extending from IRef.
#define ABSTRACT_IREF(CLASS_NAME__)  \
 protected:                          \
  CLASS_NAME__() noexcept = default; \
  ~CLASS_NAME__() override = default;

#define REF_COUNTED_NO_DESTRUCTOR(CLASS_TYPE__)                                \
 public:                                                                       \
  void ref() const noexcept final { mRefCt.ref(); }                            \
  void unref() const noexcept final { mRefCt.unref(); }                        \
                                                                               \
 private:                                                                      \
  static void mSelfDeleter(CLASS_TYPE__* selfPtr) noexcept { delete selfPtr; } \
  RefCt<CLASS_TYPE__> mRefCt { this, mSelfDeleter }

// Used for concrete classes extending from IRef.
#define REF_COUNTED(REF_CT_CLASS_TYPE__)  \
 private:                                 \
  ~REF_CT_CLASS_TYPE__() final = default; \
  REF_COUNTED_NO_DESTRUCTOR(REF_CT_CLASS_TYPE__)

#define GS_C_LINKAGE extern "C"

#define GS_EXPORT GS_C_LINKAGE GS_PUBLIC

#if _MSC_VER >= 1400
#include <sal.h>
#if _MSC_VER == 1400
#define GS_FMT_STR(p) __format_string p
#else  // _MSC_VER != 1400 (> 1400)
#define GS_FMT_STR(p) _Printf_format_string_ p
#endif  // _MSC_VER == 1400
#else
#define GS_FMT_STR(p) p
#endif  // _MSC_VER >= 1400

#ifdef __GNUC__
#define GS_FMT_ATTR(FMT_OFFSET, PARAM_OFFSET) __attribute__((format(printf, FMT_OFFSET, PARAM_OFFSET)))
#else
#define GS_FMT_ATTR(FMT_OFFSET, PARAM_OFFSET)
#endif  // __GNUC__

#endif  // GPUSDRPIPELINE_GSDEFS_H
