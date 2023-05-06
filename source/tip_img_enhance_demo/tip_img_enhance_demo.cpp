#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tip/img_enhance/DarkChannelHazeRemoval.hpp"
#include "tip/img_enhance/FastGuidedFilter.hpp"

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    std::unique_ptr<char[]> buf( new char[ size ] );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}



int main(int argc, const char *argv[])
{
    try
    {
        /* code */

        std::string imagePath = "/home/adam/Development/Data/MyHazyDatasets/IMG_20230407_192145938.jpg";

        std::cout << "Loading input image from path '" << imagePath << "'." << std::endl;

        const cv::Mat input = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        // cv::Mat inputSmaller;
        // cv::resize(input, inputSmaller, cv::Size(0, 0), 1, 1);
        cv::Mat_<cv::Vec3f> inputF;
        input.convertTo(inputF, CV_32F, 1.0 / 255);


        std::cout << "Writing output files to '" << std::filesystem::current_path().c_str() << "'." << std::endl;

        // Try various parameters
        for(int darkChannelWindowHalfSize : {5, 10, 20})
        {
            cv::Vec3f atmosphericLight;
            cv::Mat_<float> transmission;
            tip::img_enhance::dark_channel::estimate_transmission(
                inputF,
                transmission,
                atmosphericLight,
                darkChannelWindowHalfSize,
                0.02f,
                0.95f
            );

            {
                std::string initTransmissionFileName = string_format("init_tranmission_%d.png",
                    darkChannelWindowHalfSize);
                std::cout << "Saving '" << initTransmissionFileName << "'." << std::endl;

                if(!cv::imwrite(initTransmissionFileName, transmission * 255))
                {
                    throw std::runtime_error(string_format("Failed to save image '%s'.", initTransmissionFileName));
                }
            }

            for(int refinmentWinHalfSize : {darkChannelWindowHalfSize, darkChannelWindowHalfSize * 2, darkChannelWindowHalfSize * 4})
            {
                for(int refinementImgBlurKernelSize : {0, 1})
                {
                    cv::Mat_<cv::Vec3f> inputFBlurred;
                    if(refinementImgBlurKernelSize == 0)
                    {
                        inputFBlurred = inputF;
                    }
                    else
                    {
                        cv::boxFilter(
                            inputF, inputFBlurred, CV_32F,
                            cv::Size(refinementImgBlurKernelSize * 2 + 1, refinementImgBlurKernelSize * 2 + 1),
                            cv::Point(-1, -1), true,
                            cv::BorderTypes::BORDER_REPLICATE);
                    }
                    
                    for(int refinementIters : {1, 2, 4})
                    {
                        for(int refinementAlgo : {0, 1, 2})
                        {
                            cv::Mat_<float> refinedTransmission = transmission.clone();
                            for(int iterIdx = 0; iterIdx < refinementIters; ++iterIdx)
                            {
                                switch(refinementAlgo)
                                {
                                    case 0:
                                        tip::img_enhance::fast_guided_filter::matte_alpha_channel_rgb_full(
                                            inputFBlurred,
                                            refinedTransmission,
                                            refinedTransmission,
                                            refinmentWinHalfSize
                                        );
                                        break;
                                    case 1:
                                        tip::img_enhance::fast_guided_filter::matte_alpha_channel_rgb_fast(
                                            inputFBlurred,
                                            refinedTransmission,
                                            refinedTransmission,
                                            refinmentWinHalfSize
                                        );
                                        break;
                                    case 2:
                                        tip::img_enhance::fast_guided_filter::matte_alpha_channel_rgb_simple(
                                            inputFBlurred,
                                            refinedTransmission,
                                            refinedTransmission,
                                            refinmentWinHalfSize
                                        );
                                        break;
                                }
                            }

                            {
                                std::string refinedTransmissionFileName = string_format("ref_trans_%d_%d_%d_%d_%d.png",
                                    darkChannelWindowHalfSize, refinmentWinHalfSize, refinementImgBlurKernelSize, refinementIters, refinementAlgo);

                                std::cout << "Saving '" << refinedTransmissionFileName << "'." << std::endl;
                                if(!cv::imwrite(refinedTransmissionFileName, refinedTransmission * 255))
                                {
                                    throw std::runtime_error(string_format("Failed to save image '%s'.", refinedTransmissionFileName));
                                }
                            }

                            cv::Mat_<float> bluredTransmission;
                            cv::bilateralFilter(refinedTransmission, bluredTransmission, -1, 0.3, 5, cv::BORDER_REPLICATE);


                            {
                                std::string bluredTransmissionFileName = string_format("blr_trans_%d_%d_%d_%d_%d.png",
                                    darkChannelWindowHalfSize, refinmentWinHalfSize, refinementImgBlurKernelSize, refinementIters, refinementAlgo);

                                std::cout << "Saving '" << bluredTransmissionFileName << "'." << std::endl;
                                if(!cv::imwrite(bluredTransmissionFileName, bluredTransmission * 255))
                                {
                                    throw std::runtime_error(string_format("Failed to save image '%s'.", bluredTransmissionFileName));
                                }
                            }

                            for(float minTransmission : {0.1f, 0.2f})
                            {
                                cv::Mat_<cv::Vec3f> outputBgrImage;
                                tip::img_enhance::dark_channel::recover_radiance(
                                    inputF,
                                    bluredTransmission,
                                    atmosphericLight,
                                    outputBgrImage,
                                    minTransmission);

                                std::string outputImageFileName = string_format("output_%d_%d_%d_%d_%d_%f.png",
                                    darkChannelWindowHalfSize, refinmentWinHalfSize, refinementImgBlurKernelSize, refinementIters, refinementAlgo, minTransmission);

                                std::cout << "Saving '" << outputImageFileName << "'." << std::endl;
                                if(!cv::imwrite(outputImageFileName, outputBgrImage * 255))
                                {
                                    throw std::runtime_error(string_format("Failed to save image '%s'.", outputImageFileName));
                                }
                            }
                        }
                    }
                }
            }
        }
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;

        return -1;
    }
    catch(...)
    {
        std::cerr << "Unexpected exception!" << std::endl;
        return -2;
    }
    
}