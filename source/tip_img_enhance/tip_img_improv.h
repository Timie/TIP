#pragma once

#include <tip_core.h>

#ifdef tip_img_improv_EXPORTS // This should be defined automatically by CMake when building the library.
#  define TIP_IMG_IMPROV_API TIP_EXPORT
#else
#  define TIP_IMG_IMPROV_API TIP_IMPORT
#endif