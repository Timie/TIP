
add_executable(tip_img_enhance_demo
    tip_img_enhance_demo.cpp)
    
target_link_libraries(tip_img_enhance_demo
    tip_img_enhance
	${OPENCV_libs}
	${IMGUI_libs}
)
