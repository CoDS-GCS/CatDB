//
// Created by saeed on 04.11.23.
//

#ifndef CATDB_VALUETYPEUTILS_H
#define CATDB_VALUETYPEUTILS_H


#pragma once

#include <ValueTypeCode.h>

#include <iostream>
#include <string>

#include <cinttypes>
#include <cstddef>

// Intended for use with TEMPLATE_TEST_CASE in the test cases, but fits nicely
// here where everything else value-type-related resides, as that helps to keep
// changes to the list of supported data types local.
#define ALL_VALUE_TYPES \
    int8_t, int32_t, int64_t, \
    uint8_t, uint32_t, uint64_t, \
    float, double

struct ValueTypeUtils {

    static size_t sizeOf(ValueTypeCode type);

    static void printValue(std::ostream & os, ValueTypeCode type, const void * array, size_t pos);

    template<typename ValueType>
    static const ValueTypeCode codeFor;

    template<typename ValueType>
    static const std::string cppNameFor;

    template<typename ValueType>
    static const std::string irNameFor;

    static const std::string cppNameForCode(ValueTypeCode type);

    static const std::string irNameForCode(ValueTypeCode type);
};

template<> const ValueTypeCode ValueTypeUtils::codeFor<int8_t>;
template<> const ValueTypeCode ValueTypeUtils::codeFor<int32_t>;
template<> const ValueTypeCode ValueTypeUtils::codeFor<int64_t>;
template<> const ValueTypeCode ValueTypeUtils::codeFor<uint8_t>;
template<> const ValueTypeCode ValueTypeUtils::codeFor<uint32_t>;
template<> const ValueTypeCode ValueTypeUtils::codeFor<uint64_t>;
template<> const ValueTypeCode ValueTypeUtils::codeFor<float>;
template<> const ValueTypeCode ValueTypeUtils::codeFor<double>;

template<> const std::string ValueTypeUtils::cppNameFor<int8_t>;
template<> const std::string ValueTypeUtils::cppNameFor<int32_t>;
template<> const std::string ValueTypeUtils::cppNameFor<int64_t>;
template<> const std::string ValueTypeUtils::cppNameFor<uint8_t>;
template<> const std::string ValueTypeUtils::cppNameFor<uint32_t>;
template<> const std::string ValueTypeUtils::cppNameFor<uint64_t>;
template<> const std::string ValueTypeUtils::cppNameFor<float>;
template<> const std::string ValueTypeUtils::cppNameFor<double>;
template<> const std::string ValueTypeUtils::cppNameFor<bool>;

template<> const std::string ValueTypeUtils::irNameFor<int8_t>;
template<> const std::string ValueTypeUtils::irNameFor<int32_t>;
template<> const std::string ValueTypeUtils::irNameFor<int64_t>;
template<> const std::string ValueTypeUtils::irNameFor<uint8_t>;
template<> const std::string ValueTypeUtils::irNameFor<uint32_t>;
template<> const std::string ValueTypeUtils::irNameFor<uint64_t>;
template<> const std::string ValueTypeUtils::irNameFor<float>;
template<> const std::string ValueTypeUtils::irNameFor<double>;


#endif //CATDB_VALUETYPEUTILS_H
