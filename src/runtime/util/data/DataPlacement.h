//
// Created by saeed on 04.11.23.
//

#pragma once

#include "IAllocationDescriptor.h"
#include "Range.h"
//#include <runtime/local/context/DaphneContext.h>

#include <atomic>

/**
 * The DataPlacement struct binds an allocation descriptor to a range description and stores an ID of
 * an instantiated object.
 */
struct DataPlacement {
    size_t dp_id;

    // used to generate object IDs
    static std::atomic_size_t instance_count;

    std::unique_ptr<IAllocationDescriptor> allocation{};

    std::unique_ptr<Range> range{};

    DataPlacement() = delete;
    DataPlacement(std::unique_ptr<IAllocationDescriptor> _a, std::unique_ptr<Range> _r) : dp_id(instance_count++),
                                                                                          allocation(std::move(_a)), range(std::move(_r)) { }
};