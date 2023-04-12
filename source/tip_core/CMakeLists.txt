

add_library(tip_core ${TIP_LIB_TYPE}
    tip/core/TipCore.hpp
    tip/core/TipCore.cpp
)


target_include_directories(
	tip_core
	PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${HEDLEY_include_dir}>
    $<INSTALL_INTERFACE:include>
)


target_link_libraries(tip_core
    ${OpenCV_LIBS}
    Eigen3::Eigen
)

