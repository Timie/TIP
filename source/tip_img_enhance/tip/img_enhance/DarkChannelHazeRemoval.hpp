#pragma once


#include <opencv2/core.hpp>
#include <cstdint>

#include "tip_img_improv.h"


namespace tip::img_enhance::dark_channel
{
    TIP_IMG_IMPROV_API
    void remove_haze(
        const cv::Mat_<cv::Vec3f> &inputBgrImage,
        cv::Mat_<cv::Vec3f> &outputBgrImage,
        int darkChannelNeighHalfSize = 7,
        float atmoMinTopRatio = 0.01f,
        float darkChannelMultiplier = 0.95f,
        float minTransmission = 0.1f,
        int transmissionRefinementWindowHalfSize = -1);

    /**
     * Calculates dark channel from the image.
     * 
     * Dark channel is just simple min over channels of the image in the neigbourhood.
    */
    TIP_IMG_IMPROV_API
    void compute_dark_channel(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        cv::Mat_<float> &darkChannel,
        int neighbourhoodHalfSize = 7
    );

    TIP_IMG_IMPROV_API
    void compute_dark_channel(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        cv::Mat_<float> &darkChannel,
        const cv::Vec3f& atmosphericLight,
        int neighbourhoodHalfSize = 7
    );

    TIP_IMG_IMPROV_API
    cv::Vec3f estimate_atmospheric_light_top_dch(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &darkChannel,
        float topRatio = 0.01f
    );

    TIP_IMG_IMPROV_API
    cv::Vec3f estimate_atmospheric_light_min_dch(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &darkChannel,
        float minDarkChannelValue
    );

    // TODO: Move this to a general place
    /**
     * n = zero based index => 0 = highest value, 1 = second highest value.
    */
    TIP_IMG_IMPROV_API
    float find_nth_largest_value(
        const cv::Mat_<float> &arr,
        int n,
        int minBins = 255);

    TIP_IMG_IMPROV_API
    void compute_transmission(
        const cv::Mat_<float> &darkChannel,
        cv::Mat_<float> &transmission,
        float darkChannelMultiplier = 0.95 // omega in original paper
    );

    /// Recovering riadiance 
    TIP_IMG_IMPROV_API
    void recover_radiance(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &transmission,
        const cv::Vec3f& atmosphericLight,
        cv::Mat_<cv::Vec3f> &outputBGRImage,
        float minTransmission = 0.1 // t0 in original paper
    );

    TIP_IMG_IMPROV_API
    void recover_radiance(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &dampedTransmission, // t1
        const cv::Mat_<float> &transmission, // t2
        const cv::Vec3f& atmosphericLight,
        cv::Mat_<cv::Vec3f> &outputBGRImage,
        float minTransmission = 0.1 // t0 in original paper
    );
}