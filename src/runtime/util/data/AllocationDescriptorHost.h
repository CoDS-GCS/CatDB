//
// Created by saeed on 04.11.23.
//

#pragma once

#include "DataPlacement.h"
#include <memory>

class AllocationDescriptorHost : public IAllocationDescriptor {
    ALLOCATION_TYPE type = ALLOCATION_TYPE::HOST;
    std::shared_ptr<std::byte> data{};

public:
    ~AllocationDescriptorHost() override = default;
    [[nodiscard]] ALLOCATION_TYPE getType() const override { return type; }
    void createAllocation(size_t size, bool zero) override { }
    std::string getLocation() const override
    { return "Host"; }
    std::shared_ptr<std::byte> getData() override { return data; }
    void transferTo(std::byte* src, size_t size) override { }
    void transferFrom(std::byte* dst, size_t size) override {}
    [[nodiscard]] std::unique_ptr<IAllocationDescriptor> clone() const override {
        return std::make_unique<AllocationDescriptorHost>(*this);
    }
    bool operator==(const IAllocationDescriptor* other) const override { return (getType() == other->getType()); }
};

