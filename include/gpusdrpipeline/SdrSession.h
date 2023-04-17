//
// Created by Rick Kern on 4/1/23.
//

#ifndef GPUSDRPIPELINE_SDRSESSION_H
#define GPUSDRPIPELINE_SDRSESSION_H

#include <gsdrpipeline/Status.h>
#include <stdint.h>

extern "C" {
Status createSession(uint8_t* data, size_t dataLength);
}

#endif  // GPUSDRPIPELINE_SDRSESSION_H
