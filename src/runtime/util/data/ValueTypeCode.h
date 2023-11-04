//
// Created by saeed on 04.11.23.
//

#ifndef CATDB_VALUETYPECODE_H
#define CATDB_VALUETYPECODE_H

#include <cinttypes>

enum class ValueTypeCode : uint8_t {
    SI8, SI32, SI64, // signed integers (intX_t)
    UI8, UI32, UI64, // unsigned integers (uintx_t)
    F32, F64, // floating point (float, double)
    INVALID, // only for JSON enum conversion
    // TODO Support bool as well, but poses some challenges (e.g. sizeof).
//    UI1 // boolean (bool)
};
#endif //CATDB_VALUETYPECODE_H
