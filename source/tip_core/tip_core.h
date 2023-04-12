#pragma once

#include <hedley.h>

#define TIP_EXPORT HEDLEY_PUBLIC
#define TIP_IMPORT HEDLEY_IMPORT

#ifdef tip_core_EXPORTS // This should be defined automatically by CMake when building the library.
#  define TIP_CORE_API TIP_EXPORT
#else
#  define TIP_CORE_API TIP_IMPORT
#endif