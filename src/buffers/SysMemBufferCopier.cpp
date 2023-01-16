//
// Created by Rick Kern on 1/4/23.
//

#include "SysMemBufferCopier.h"

#include <cstring>

void SysMemBufferCopier::copy(void* dst, const void* src, size_t length) { memcpy(dst, src, length); }
