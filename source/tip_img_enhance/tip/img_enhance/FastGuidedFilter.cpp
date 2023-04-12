#include "FastGuidedFilter.hpp"

#include <opencv2/imgproc.hpp>
#include <limits>
#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "tip/core/TipCore.hpp"

namespace
{
    cv::Vec<float, 9> squareAndFlatten(const cv::Vec3f& v)
    {
        cv::Vec<float, 9> result;
        // v * vT store row major.
        result[0] = v[0] * v[0];
        result[1] = v[0] * v[1];
        result[2] = v[0] * v[2];
        result[3] = v[1] * v[0];
        result[4] = v[1] * v[1];
        result[5] = v[1] * v[2];
        result[6] = v[2] * v[0];
        result[7] = v[2] * v[1];
        result[8] = v[2] * v[2];

        return result;
    }

    cv::Vec3f covIpDividedByVarIPlusEps(
        const cv::Vec3f& covIp,
        const cv::Vec<float, 9> &varI)
    {
        // covIp / (varI + eps) = ((varI + eps)^-1) * covIp
        float eps = 0.0001f;//std::numeric_limits<float>::epsilon();
        Eigen::Matrix3f varIMat; // + eps
        varIMat << 
            varI[0] + eps, varI[1],       varI[2],
            varI[3],       varI[4] + eps, varI[5],
            varI[6],       varI[7],       varI[8] + eps;

        // because varIMat is positive definite, we can use llt decomposition, which should be faster.
        varIMat = varIMat.llt().solve(Eigen::Matrix3f::Identity()).eval();
        
        Eigen::Vector3f result = varIMat * Eigen::Vector3f(covIp[0], covIp[1], covIp[2]);
        return cv::Vec3f(result[0], result[1], result[2]);
    }

    void
    calculatePixelwiseXXT(
        const cv::Mat_<cv::Vec3f> &image,
        cv::Mat_<cv::Vec<float, 9>> &xxt)
    {
        xxt.create(image.size());

        tip::core::applyElementwise(
            image, xxt,
            [](const cv::Vec3f &img, cv::Vec<float, 9>& result)
            {
                result = ::squareAndFlatten(img);
            }
        );
    }

    void
    calculateImageVariance(
        const cv::Mat_<cv::Vec3f> &imageAveraged,
        const cv::Mat_<cv::Vec<float, 9>> &xxtAveraged,
        cv::Mat_<cv::Vec<float, 9>> &imgPxVariance)
    {
        assert(imageAveraged.size() == xxtAveraged.size());

        imgPxVariance.create(imageAveraged.size());
        tip::core::applyElementwise(
            imageAveraged, xxtAveraged, imgPxVariance,
            [](const cv::Vec3f& meanI, const cv::Vec<float, 9>& corr_i, cv::Vec<float, 9>& result)
            {
                result = corr_i - ::squareAndFlatten(meanI);
            }
        );
    }

    void
    calculateImageMaskCovariance(
        const cv::Mat_<cv::Vec3f> &imageMaskAveraged,
        const cv::Mat_<cv::Vec3f> &imageAveraged,
        const cv::Mat_<float> &maskAveraged,
        cv::Mat_<cv::Vec3f> &imageMaskCovariance)
    {
        assert(imageMaskAveraged.size() == imageAveraged.size());
        assert(imageMaskAveraged.size() == maskAveraged.size());

        imageMaskCovariance.create(imageMaskAveraged.size());
        tip::core::applyElementwise(
            imageMaskAveraged, imageAveraged, maskAveraged, imageMaskCovariance,
            [](const cv::Vec3f& meanIP, const cv::Vec3f& meanI, const float& meanP, cv::Vec3f& result)
            {
                result = meanIP - meanI * meanP;
            }
        );
    }

    void
    calculateA(
        const cv::Mat_<cv::Vec<float, 9>> &imgVariance,
        const cv::Mat_<cv::Vec3f> &imageMaskCovariance,
        cv::Mat_<cv::Vec3f> &A)
    {
        assert(imgVariance.size() == imageMaskCovariance.size());
        
        A.create(imageMaskCovariance.size());

        tip::core::applyElementwise(
            imgVariance, imageMaskCovariance, A,
            [](const cv::Vec<float, 9>& var_i, const cv::Vec3f& cov_ip, cv::Vec3f &result)
            {
                // (var_i + I * eps) ^ -1 * cov_ip
                result = covIpDividedByVarIPlusEps(cov_ip, var_i);
            }
        );
    }

    void
    calculateB(
        const cv::Mat_<float> &maskAveraged, 
        const cv::Mat_<cv::Vec3f> &A, 
        const cv::Mat_<cv::Vec3f> &imageAveraged,
        cv::Mat_<float> &B)
    {
        assert(maskAveraged.size() == A.size());
        assert(maskAveraged.size() == imageAveraged.size());

        B.create(maskAveraged.size());

        tip::core::applyElementwise(
            maskAveraged, A, imageAveraged, B,
            [](const float &meanP, const cv::Vec3f &a, const cv::Vec3f& meanI, float &result)
            {
                result = meanP - a.dot(meanI);
            }
        );
    }
}

void 
tip::img_enhance::fast_guided_filter::
matte_alpha_channel_rgb_full(
    const cv::Mat_<cv::Vec3f> &image,
    const cv::Mat_<float> &maskToFilter,
    cv::Mat_<float> &filteredMask,
    int halfWindowSize)
{
    assert(image.size() == maskToFilter.size());
    cv::Mat_<cv::Vec3f> meanA;
    cv::Mat_<float> meanB;
    matte_alpha_channel_rgb_full(
        image,
        maskToFilter,
        meanA,
        meanB,
        halfWindowSize
    );

    filteredMask.create(image.size());
    tip::core::applyElementwise(
        meanA, image, meanB, filteredMask,
        [](const cv::Vec3f& a, const cv::Vec3f &i, float& b, float& pNew)
        {
            pNew = a.dot(i) + b;
        }
    );
}

void 
tip::img_enhance::fast_guided_filter::
matte_alpha_channel_rgb_full(
    const cv::Mat_<cv::Vec3f> &image,
    const cv::Mat_<float> &maskToFilter, 
    cv::Mat_<cv::Vec3f> &meanA,
    cv::Mat_<float> &meanB,
    int halfWindowSize)
{
    assert(image.size() == maskToFilter.size());
    assert(halfWindowSize >= 0 && halfWindowSize < std::numeric_limits<int>::max() / 2 - 1);

    cv::Mat_<cv::Vec3f> meanI;
    int windowSize = halfWindowSize * 2 + 1;
    cv::boxFilter(
        image, meanI, CV_32F,
        cv::Size(windowSize, windowSize),
        cv::Point(-1, -1), true,
        cv::BorderTypes::BORDER_REPLICATE);
    
    cv::Mat_<float> meanP;
    cv::boxFilter(
        maskToFilter, meanP, CV_32F,
        cv::Size(windowSize, windowSize),
        cv::Point(-1, -1), true,
        cv::BorderTypes::BORDER_REPLICATE);

    cv::Mat_<cv::Vec<float, 9>> xxt; // I * I => corr_i => var_i
    calculatePixelwiseXXT(image, xxt);
    cv::boxFilter(
        xxt, xxt, CV_32F,
        cv::Size(windowSize, windowSize),
        cv::Point(-1, -1), true,
        cv::BorderTypes::BORDER_REPLICATE);

    calculateImageVariance(meanI, xxt, xxt);

    cv::Mat_<cv::Vec3f> IP; // I * p => mean(I * p) => cov_ip
    IP.create(image.size());
    tip::core::applyElementwise(
        image, maskToFilter, IP,
        [](const cv::Vec3f &i, const float &maskToFilter, cv::Vec3f &result)
        {
            result = i * maskToFilter;
        }
    );

    cv::boxFilter(
        IP, IP, CV_32F,
        cv::Size(windowSize, windowSize),
        cv::Point(-1, -1), true,
        cv::BorderTypes::BORDER_REPLICATE);

    calculateImageMaskCovariance(
        IP,
        meanI,
        meanP,
        IP
    );

    calculateA(
        xxt,
        IP,
        meanA);

    calculateB(
        meanP,
        meanA,
        meanI,
        meanB
    );

    cv::boxFilter(
        meanA, meanA, CV_32F,
        cv::Size(windowSize, windowSize),
        cv::Point(-1, -1), true,
        cv::BorderTypes::BORDER_REPLICATE);

    cv::boxFilter(
        meanB, meanB, CV_32F,
        cv::Size(windowSize, windowSize),
        cv::Point(-1, -1), true,
        cv::BorderTypes::BORDER_REPLICATE);
}