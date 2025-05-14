#!/bin/sh

find xrtailor -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.c" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \) \
    ! -path "xrtailor/include/xrtailor/external/*" \
    ! -path "xrtailor/src/external/*" \
    -exec clang-format-12 -i {} +
    # -exec clang-format-12 --dry-run --Werror {} +