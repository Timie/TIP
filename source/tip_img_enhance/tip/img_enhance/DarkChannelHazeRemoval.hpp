#pragma once


#include <opencv2/core.hpp>
#include <cstdint>

#include "tip_img_improv.h"


/**
 * @brief Contains implementation of Single Image Haze Removal Using Dark Channel Prior
 * by Kaiming He, Jian Sun, and Xiaoou Tang (2011).
 * 
 * Note that soft-matting step is replaced with fast guided filter due to processing speed reasons.
*/
namespace tip::img_enhance::dark_channel
{
    /**
     * Applies the algorithm "Single Image Haze Removal Using Dark Channel Prior" to the
     * image.
     * 
     * Note that the order of colour channels (BGR or RGB) is not important.
    */
    TIP_IMG_IMPROV_API
    void remove_haze(
        const cv::Mat_<cv::Vec3f> &inputBgrImage,
        cv::Mat_<cv::Vec3f> &outputBgrImage,
        int darkChannelNeighHalfSize = 7,
        float atmoMinTopRatio = 0.01f,
        float darkChannelMultiplier = 0.95f,
        float minTransmission = 0.1f,
        int transmissionRefinementWindowHalfSize = -1);

    TIP_IMG_IMPROV_API
    void estimate_transmission(
        const cv::Mat_<cv::Vec3f> &inputBgrImage,
        cv::Mat_<float> &transmission,
        cv::Vec3f &atmosphericLight,
        int darkChannelNeighHalfSize = 7,
        float atmoMinTopRatio = 0.01f,
        float darkChannelMultiplier = 0.95f);


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

    /**
     * Similar to other overload, but this also assumes existing atmospheric light.
     * The image colour is firstly divided by the atmospheric light before computing
     * the dark channel.
    */
    TIP_IMG_IMPROV_API
    void compute_dark_channel(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        cv::Mat_<float> &darkChannel,
        const cv::Vec3f& atmosphericLight,
        int neighbourhoodHalfSize = 7
    );

    /**
     * Estimates atmospheric light by averaging image pixels with the highest dark
     * channel value - the pixels are selected if they have to the top
     * "topRatio * total pixel count" pixels when ordered by dark channel value.
    */
    TIP_IMG_IMPROV_API
    cv::Vec3f estimate_atmospheric_light_top_dch(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &darkChannel,
        float topRatio = 0.01f
    );

    /**
     * Estimates atmospheric light by averaging image pixels with the dark channel
     * equal or higher than minDarkChannelValue.
    */
    TIP_IMG_IMPROV_API
    cv::Vec3f estimate_atmospheric_light_min_dch(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &darkChannel,
        float minDarkChannelValue
    );

    /**
     * Calculates the transmission map from the tark channel.
    */
    TIP_IMG_IMPROV_API
    void compute_transmission(
        const cv::Mat_<float> &darkChannel,
        cv::Mat_<float> &transmission,
        float darkChannelMultiplier = 0.95 // omega in original paper
    );

    /**
     * Recovers the radiance based on the equation (22) in the original paper.
     * 
     * J[X] = (I[X] - A)/(max(t[x], t0) + A)
     * 
     * This is a simpler radiance recovery algorithm.
    */
    TIP_IMG_IMPROV_API
    void recover_radiance(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &transmission,
        const cv::Vec3f& atmosphericLight,
        cv::Mat_<cv::Vec3f> &outputBGRImage,
        float minTransmission = 0.1 // t0 in original paper
    );


    /**
     * Recovers the radiance based on the equation (13) in the original paper.
     * J[x] = (I[x] - A(1 - t2[x])) / max(t1[x], t0)
    */
    TIP_IMG_IMPROV_API
    void recover_radiance(
        const cv::Mat_<cv::Vec3f> &inputBGRImage,
        const cv::Mat_<float> &dampedTransmission, // t1
        const cv::Mat_<float> &transmission, // t2
        const cv::Vec3f& atmosphericLight,
        cv::Mat_<cv::Vec3f> &outputBGRImage,
        float minTransmission = 0.1 // t0 in original paper
    );

    // TODO: Move this to a general place
    /**
     * Finds the approximate value of the nth largest element in the matrix.
     * This may be inaccurate in certain scenarios, especially when the histogram
     * of the arr is very skewed or irregular.
     * 
     * n = zero based index => 0 = highest value, 1 = second highest value, etc...
    */
    TIP_IMG_IMPROV_API
    float find_nth_largest_value(
        const cv::Mat_<float> &arr,
        int n,
        int minBins = 255);
}