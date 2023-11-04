//
// Created by saeed on 02.11.23.
//

#include <runtime/local/io/DaphneSerializer.h>
#include "Frame.h"
#include <ostream>

std::ostream & operator<<(std::ostream & os, const Frame & obj)
{
    obj.print(os);
    return os;
}

size_t Frame::serialize(std::vector<char> &buf) const {
    return DaphneSerializer<Frame>::serialize(this, buf);
}
