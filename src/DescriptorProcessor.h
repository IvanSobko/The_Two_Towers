#ifndef THE_TWO_TOWERS_DESCRIPTORPROCESSOR_H
#define THE_TWO_TOWERS_DESCRIPTORPROCESSOR_H

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/xfeatures2d.hpp"

class DescriptorProcessor {

public:
    DescriptorProcessor(std::string objectPath, std::string scenesPath);

    void setHaussian(int haussian);

    void displayMatches();

    void process();

    void saveMetricsToFile(std::string &filename);

    size_t getSize();

private:
    std::vector<std::string> const findSceneImages(const std::string &path);

    void detectAndCompute(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor);

    size_t mSize = 0;
    int mMinHaussian = 100;
    cv::Mat mObjectImg;
    std::vector<std::pair<std::string, cv::Mat>> mSceneImgs;

    cv::Ptr<cv::xfeatures2d::SURF> mSURFDetector = nullptr;
    cv::Mat mObjectDescriptor;
    std::vector<cv::Mat> mSceneDescriptors;

    std::vector<cv::KeyPoint> mObjectKeypoints;
    std::vector<std::vector<cv::KeyPoint>> mSceneKeypoints;
    std::vector<std::pair<std::string, cv::Mat>> mReadyImgs;
    cv::Ptr<cv::FlannBasedMatcher> mMatcher = nullptr;

    struct Metrics {
        std::string filename;
        double pointsInsideAvg;
        double distanceAvg;
        int64_t duration;
        int width;
        int height;

        std::vector<std::string> toVector() const;
    };
    std::vector<Metrics> mMetrics;
};


#endif //THE_TWO_TOWERS_DESCRIPTORPROCESSOR_H
