//
// Created by saeed on 04.11.23.
//

#ifndef CATDB_RANGE_H
#define CATDB_RANGE_H

#pragma once

#include <memory>

// Unused for now. This can be used to track sub allocations of matrices
struct Range {
    size_t r_start;
    size_t c_start;
    size_t r_len;
    size_t c_len;

    explicit Range() : r_start(0), c_start(0), r_len(0), c_len(0) { }
    explicit Range(size_t r1, size_t c1, size_t r2, size_t c2) : r_start(r1), c_start(c1), r_len(r2), c_len(c2) { }

    bool operator==(const Range* other) const {
        return((other != nullptr) && (r_start == other->r_start && c_start == other->c_start && r_len == other->r_len &&
                                      c_len == other->c_len));
    }

    bool operator==(const Range other) const {
        return(r_start == other.r_start && c_start == other.c_start && r_len == other.r_len &&
               c_len == other.c_len);
    }

    [[nodiscard]] std::unique_ptr<Range> clone() const { return std::make_unique<Range>(*this); }
};

#endif //CATDB_RANGE_H
