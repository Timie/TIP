
add_library(tip_img_enhance ${TIP_LIB_TYPE}
    tip/img_enhance/FogRemovalEdgePreservingSmothing.hpp
    tip/img_enhance/FogRemovalEdgePreservingSmothing.cpp
    tip/img_enhance/DarkChannelHazeRemoval.hpp
    tip/img_enhance/DarkChannelHazeRemoval.cpp
    tip/img_enhance/FusionForWeakIllumination.hpp
    tip/img_enhance/FusionForWeakIllumination.cpp
    tip/img_enhance/FastGuidedFilter.hpp
    tip/img_enhance/FastGuidedFilter.cpp)


target_include_directories(
	tip_img_enhance
	PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${HEDLEY_include_dir}>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tip_img_enhance
    tip_core
    ${OpenCV_LIBS}
    Eigen3::Eigen
)


