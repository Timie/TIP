#pragma once


#include <opencv2/core.hpp>
#include <cstdint>

#include "tip_img_improv.h"


// TODO: move this to separate library
namespace tip::img_enhance::fast_guided_filter
{
    // See Fast Guided Filter by Kaiming He and Jian Sun
    TIP_IMG_IMPROV_API void matte_alpha_channel_rgb_full(
        const cv::Mat_<cv::Vec3f> &image,
        const cv::Mat_<float> &maskToFilter,
        cv::Mat_<float> &filteredMask,
        int halfWindowSize = 7
    );

    // This is suitable for subsampling approach
    // filteredMask = elementwise(image * meanA) + meanB
    TIP_IMG_IMPROV_API void matte_alpha_channel_rgb_full(
        const cv::Mat_<cv::Vec3f> &image,
        const cv::Mat_<float> &maskToFilter,
        cv::Mat_<cv::Vec3f> &meanA,
        cv::Mat_<float> &meanB,
        int halfWindowSize = 7
    );
}
