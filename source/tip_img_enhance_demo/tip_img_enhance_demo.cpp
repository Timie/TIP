#include <iostream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "tip/img_enhance/DarkChannelHazeRemoval.hpp"


int main(int argc, const char *argv[])
{
    try
    {
        /* code */

        std::string imagePath = "/home/adam/Development/Data/MyHazyDatasets/IMG_20230407_192145938.jpg";

        const cv::Mat input = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        cv::Mat_<cv::Vec3f> inputF;
        input.convertTo(inputF, CV_32F, 1.0 / 255);

        cv::Mat_<cv::Vec3f> recoveredImage;
        tip::img_enhance::dark_channel::remove_haze(
            inputF,
            recoveredImage,
            10, 0.02f, 0.95f, 0.2f, 20);
        cv::imshow("Input Image", inputF);
        cv::imshow("Output Image", recoveredImage);

        cv::waitKey(-1);
        std::cout << "All good! :)\n";
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