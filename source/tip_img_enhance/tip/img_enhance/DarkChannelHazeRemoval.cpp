#include "DarkChannelHazeRemoval.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cassert>
#include <limits>

#include <tip/core/TipCore.hpp>
#include "tip/img_enhance/FastGuidedFilter.hpp"

void
tip::img_enhance::dark_channel::
remove_haze(
    const cv::Mat_<cv::Vec3f> &inputBgrImage,
    cv::Mat_<cv::Vec3f> &outputBgrImage,
    int darkChannelNeighHalfSize,
    float atmoMinTopRatio,
    float darkChannelMultiplier,
    float minTransmission,
    int transmissionRefinementWindowHalfSize)
{
    if(transmissionRefinementWindowHalfSize < 0)
    {
        transmissionRefinementWindowHalfSize = darkChannelNeighHalfSize;
    }
    
    cv::Vec3f atmosphericLight;
    cv::Mat_<float> transmission;
    estimate_transmission(
        inputBgrImage,
        transmission,
        atmosphericLight,
        darkChannelNeighHalfSize,
        atmoMinTopRatio,
        darkChannelMultiplier
    );

    // TODO: Apply soft matting and bilateral filter to tranmission.
    cv::Mat_<float> adjustedTransmission;
    tip::img_enhance::fast_guided_filter::matte_alpha_channel_rgb_fast(
        inputBgrImage, transmission, transmission, transmissionRefinementWindowHalfSize);
    
    // TODO: This seems to introduce bit of value quantisation to the transmission map.
    // TODO: Also, with provided parameters it smoothens the transmission map. Note that this seems to actually improve
    //       the clarity of fine details in the resulting image. We may try replacing it with different filters (box, guasss, dilate),
    //       or just calculating transmission map at half the resolution.
    cv::bilateralFilter(transmission, adjustedTransmission, -1, 0.3, 5, cv::BORDER_REPLICATE);

    recover_radiance(
       inputBgrImage,
       adjustedTransmission,
       atmosphericLight,
       outputBgrImage,
       minTransmission);
}

void
tip::img_enhance::dark_channel::
compute_dark_channel(
    const cv::Mat_<cv::Vec3f> &inputBGRImage, 
    cv::Mat_<float> &darkChannel, 
    int neighbourhoodHalfSize)
{
    assert(neighbourhoodHalfSize >= 0);
    assert(neighbourhoodHalfSize < std::numeric_limits<decltype(neighbourhoodHalfSize)>::max() / 2 - 1);

    if(inputBGRImage.empty())
    {
        darkChannel = cv::Mat();
        return;
    }

    // Prepare the output
    darkChannel.create(inputBGRImage.size());

    // apply min over the channels
    tip::core::applyElementwise(inputBGRImage, darkChannel,
    [](const cv::Vec3f& colour, float &darkChannelVal) noexcept {
        darkChannelVal = std::min(colour[0], std::min(colour[1], colour[2]));
    });

    // apply min in the neighbourhood = same as OpenCV's erode
    if(neighbourhoodHalfSize > 0)
    {
        const auto boxNeighbourhoodElem = 
            cv::getStructuringElement(
                cv::MorphShapes::MORPH_RECT,
                cv::Size(neighbourhoodHalfSize * 2 + 1,
                         neighbourhoodHalfSize * 2 + 1));
        cv::erode(
            darkChannel, darkChannel,
            boxNeighbourhoodElem, cv::Point(-1,-1), 
            1, cv::BorderTypes::BORDER_REPLICATE);
    }
}

void
tip::img_enhance::dark_channel::
compute_dark_channel(
    const cv::Mat_<cv::Vec3f> &inputBGRImage,
    cv::Mat_<float> &darkChannel,
    const cv::Vec3f& atmosphericLight,
    int neighbourhoodHalfSize)
{
    assert(neighbourhoodHalfSize >= 0);
    assert(neighbourhoodHalfSize < std::numeric_limits<int>::max() / 2 - 1);

    if(inputBGRImage.empty())
    {
        darkChannel = cv::Mat();
        return;
    }

    // Prepare the output
    darkChannel.create(inputBGRImage.size());

    // apply min over the channels
    tip::core::applyElementwise(inputBGRImage, darkChannel,
    [atmosphericLight](const cv::Vec3f& colour, float &darkChannelVal) noexcept {
        const cv::Vec3f normColour {
            colour[0] / atmosphericLight[0],
            colour[1] / atmosphericLight[1],
            colour[2] / atmosphericLight[2],
        };
        darkChannelVal = std::min(normColour[0], std::min(normColour[1], normColour[2]));
    });

    // apply min in the neighbourhood = same as OpenCV's erode
    if(neighbourhoodHalfSize > 0)
    {
        const auto boxNeighbourhoodElem = 
            cv::getStructuringElement(
                cv::MorphShapes::MORPH_RECT,
                cv::Size(neighbourhoodHalfSize * 2 + 1,
                         neighbourhoodHalfSize * 2 + 1));
        cv::erode(
            darkChannel, darkChannel,
            boxNeighbourhoodElem, cv::Point(-1,-1), 
            1, cv::BorderTypes::BORDER_REPLICATE);
    }
}

cv::Vec3f 
tip::img_enhance::dark_channel::
estimate_atmospheric_light_top_dch(
    const cv::Mat_<cv::Vec3f> &inputBGRImage,
    const cv::Mat_<float> &darkChannel,
    float topRatio)
{
    assert(!inputBGRImage.empty());
    assert(inputBGRImage.size() == darkChannel.size());
    assert(topRatio > 0 && topRatio <= 1);

    // Get the pixels with highest value of dark channel
    float minAtmoLightDarkChannelVal = 0;
    if(topRatio < 1)
    {
        int nthLargestIndex = static_cast<int>(std::round(darkChannel.total() * topRatio));
        minAtmoLightDarkChannelVal = 
            find_nth_largest_value(darkChannel, nthLargestIndex);
    }

    return estimate_atmospheric_light_min_dch(inputBGRImage, darkChannel, minAtmoLightDarkChannelVal);
}


cv::Vec3f
tip::img_enhance::dark_channel::
estimate_atmospheric_light_min_dch(
    const cv::Mat_<cv::Vec3f> &inputBGRImage,
    const cv::Mat_<float> &darkChannel,
    float minDarkChannelValue)
{
    assert(!inputBGRImage.empty());
    assert(inputBGRImage.size() == darkChannel.size());

    // Iterate over all pixels, and average the BGR value from the pixels which has high dark channel value.
    cv::Vec3f atmoLightSum {0.0f, 0.0f};
    int totalCount = 0;
    tip::core::applyElementwise(inputBGRImage, darkChannel,
    [minDarkChannelValue, &atmoLightSum, &totalCount](const cv::Vec3f& colour, const float &darkChannelVal) noexcept {
        if(darkChannelVal >= minDarkChannelValue)
        {
            atmoLightSum[0] += colour[0];
            atmoLightSum[1] += colour[1];
            atmoLightSum[2] += colour[2];
            ++totalCount;
        }
    });

    assert(totalCount > 0);
    return atmoLightSum / float(totalCount);
}


float
tip::img_enhance::dark_channel::
find_nth_largest_value(
    const cv::Mat_<float> &arr,
    int n,
    int minBins)
{
    assert(!arr.empty());
    assert(n >= 0 && n < arr.total());
    assert(minBins > 0);

    // TODO: If we use 8-bit unsigned int for input image, and dark channel, this can be much-much faster.

    // instead of doing partial sort, we can just calculate histogram,
    // and read value from there.

    // We split the range to max(minBins, arr.total() / n * 10) bins.
    assert(arr.total() <= std::numeric_limits<int>::max());
    const int binCount = std::max(minBins, static_cast<int>(arr.total()) * 10 / n);
    // In order to calculate histogram, that is accurate enough, we calculate min-max range and 
    // do the histogram just for that range. 
    double minRange, maxRange;
    cv::minMaxIdx(arr, &minRange, &maxRange);
    const float minRangeF = static_cast<float>(minRange);
    const float maxRangeF = static_cast<float>(maxRange);
    if(minRangeF == maxRangeF) // image is uniform
    {
        return minRangeF;
    }
    // we need to adjust histogram boundaries, because upper one is exclusive
    const float binWidth = (maxRangeF - minRangeF) / (binCount - 1); // subtracting one bin, as we need one more above to maxRange
    const float histLowBound = minRange;
    const float histUpperBound = maxRange + binWidth;

    // Calculate the histogram
    const std::vector<cv::Mat> imgsForHistogram = {arr};
    const std::vector<int> channelsForHistogram = {0};
    const std::vector<int> histogramSizes = {binCount};
    const std::vector<float> histogramRanges
        = {histLowBound, histUpperBound};

    cv::Mat histogram;
    cv::calcHist(
        imgsForHistogram, channelsForHistogram,
        cv::noArray(), histogram,
        histogramSizes, histogramRanges);
    
    // now go from the highest value of the histogram, and calculate the number of elements.
    int elementCount = 0;
    for(int histIdx = binCount; histIdx >= 0; --histIdx)
    {
        int count = static_cast<int>(std::round(histogram.at<float>(histIdx)));
        elementCount += count;
        if(elementCount > n)
        {
            return minRangeF + histIdx * binWidth;
        }
    }

    // this should never happen though.
    assert(false);
    return 0;
}


TIP_IMG_IMPROV_API
void
tip::img_enhance::dark_channel::
compute_transmission(
    const cv::Mat_<float> &darkChannel,
    cv::Mat_<float> &transmission,
    float darkChannelMultiplier)
{
    assert(darkChannelMultiplier >= 0);
    // TODO: possible speedup by splitting this, or doing by hand?
    transmission = 1 - darkChannel * darkChannelMultiplier;
}


TIP_IMG_IMPROV_API
void
tip::img_enhance::dark_channel::
recover_radiance(
    const cv::Mat_<cv::Vec3f> &inputBGRImage,
    const cv::Mat_<float> &transmission,
    const cv::Vec3f& atmosphericLight,
    cv::Mat_<cv::Vec3f> &outputBGRImage,
    float minTransmission)
{
    assert(inputBGRImage.size() == transmission.size());

    if(inputBGRImage.empty())
    {
        outputBGRImage = cv::Mat();
        return;
    }

    outputBGRImage.create(inputBGRImage.size());

    // apply J(x) = (I(x) - A)/(min(t1(x), t0)) + A
    tip::core::applyElementwise(inputBGRImage, transmission, outputBGRImage,
        [atmosphericLight, minTransmission]
        (const cv::Vec3f& colour, const float& t1, cv::Vec3f &outputColour) noexcept 
        {
            const float effectiveInvTransmission = 1 / std::max(t1, minTransmission);
            outputColour = (colour - atmosphericLight) * effectiveInvTransmission + atmosphericLight;
        });
}


TIP_IMG_IMPROV_API
void
tip::img_enhance::dark_channel::
recover_radiance(
    const cv::Mat_<cv::Vec3f> &inputBGRImage,
    const cv::Mat_<float> &dampedTransmission, // t1
    const cv::Mat_<float> &transmission, // t2
    const cv::Vec3f& atmosphericLight,
    cv::Mat_<cv::Vec3f> &outputBGRImage,
    float minTransmission)
{
    assert(inputBGRImage.size() == dampedTransmission.size());
    assert(inputBGRImage.size() == transmission.size());

    if(inputBGRImage.empty())
    {
        outputBGRImage = cv::Mat();
        return;
    }

    outputBGRImage.create(inputBGRImage.size());

    // apply J[x] = (I[x] - A(1 - t2[x])) / max(t1[x], t0)
    tip::core::applyElementwise(inputBGRImage, dampedTransmission, transmission, outputBGRImage,
        [atmosphericLight, minTransmission]
        (const cv::Vec3f& colour, const float& t1, const float& t2, cv::Vec3f &outputColour) noexcept 
        {
            const cv::Vec3f invAtmlight =  atmosphericLight * (1 - t2);
            const float effectiveInvT0 = 1 / std::max(t1, minTransmission);
            outputColour = (colour - invAtmlight) * effectiveInvT0;
        });
}


TIP_IMG_IMPROV_API
void 
tip::img_enhance::dark_channel::
estimate_transmission(
    const cv::Mat_<cv::Vec3f> &inputBgrImage,
    cv::Mat_<float> &transmission,
    cv::Vec3f &atmosphericLight,
    int darkChannelNeighHalfSize,
    float atmoMinTopRatio,
    float darkChannelMultiplier)
{
    cv::Mat_<float> initDarkChannel;
    compute_dark_channel(
        inputBgrImage,
        initDarkChannel,
        darkChannelNeighHalfSize
    );

    atmosphericLight = estimate_atmospheric_light_top_dch(
        inputBgrImage,
        initDarkChannel,
        atmoMinTopRatio
    );

    cv::Mat_<float> darkChannel;
    compute_dark_channel(
        inputBgrImage,
        darkChannel,
        atmosphericLight,
        darkChannelNeighHalfSize
    );

    compute_transmission(
        darkChannel,
        transmission,
        darkChannelMultiplier
    );
}