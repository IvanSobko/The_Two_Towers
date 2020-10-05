#include "DescriptorProcessor.h"

#include <filesystem>
#include <vector>

#include "CsvWriter.h"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


DescriptorProcessor::DescriptorProcessor(std::string objectPath, std::string scenesPath) {
    mObjectImg = imread(objectPath, cv::IMREAD_GRAYSCALE);
    std::vector<std::string> sceneImgFiles = findSceneImages(scenesPath);
    for (const auto &file: sceneImgFiles) {
        cv::Mat res = imread(scenesPath + file, cv::IMREAD_GRAYSCALE);
        if (!res.data) {
            printf("Image %s not read", file.c_str());
            continue;
        }
        mSceneImgs.push_back({file, res});
    }
    mSize = mSceneImgs.size();
    mSceneDescriptors.resize(mSize);
    mSceneKeypoints.resize(mSize);
    mSURFDetector = cv::xfeatures2d::SURF::create(mMinHaussian);
    mMatcher = cv::FlannBasedMatcher::create();
}

void DescriptorProcessor::displayMatches() {
    int num = 0;
    while (true) {
        cv::imshow(mReadyImgs[num].first, mReadyImgs[num].second);
        auto pressedKey = cv::waitKeyEx(0);
        if (pressedKey == 113) { // q to exit
            break;
        } else if (pressedKey == 63234) { // left arrow
            num = std::max(0, --num);
        } else if (pressedKey == 63235) { // right arrow
            num = std::min(static_cast<int>(mReadyImgs.size() - 1), ++num);
        }
        cv::destroyAllWindows();
    }
}

void DescriptorProcessor::process() {
    detectAndCompute(mObjectImg, mObjectKeypoints, mObjectDescriptor);
    for (int i = 0; i < mSize; i++) {
        const std::string &filename = mSceneImgs[i].first;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        //-- Step 1: Detect the keypoints using SURF Detector
        detectAndCompute(mSceneImgs[i].second, mSceneKeypoints[i], mSceneDescriptors[i]);

        std::vector<std::vector<cv::DMatch>> knn_matches;

        if (mSceneDescriptors[i].empty()) {
            printf("\nCouldn't find feature points for: %s\n", filename.c_str());
            continue;
        }

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        mMatcher->knnMatch(mObjectDescriptor, mSceneDescriptors[i], knn_matches, 2);

        //-- Step 3: Filter matches using the Lowe's ratio test
        const float ratio_thresh = 0.7f;
        std::vector<cv::DMatch> good_matches;
        for (const auto &match : knn_matches) {
            if (match[0].distance < ratio_thresh * match[1].distance) {
                good_matches.push_back(match[0]);
            }
        }

        //-- Step 4: Draw matches
        cv::Mat img_matches;
        drawMatches(mObjectImg, mObjectKeypoints, mSceneImgs[i].second, mSceneKeypoints[i],
                    good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //-- Step 5: Localize the object
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;
        for (const auto &match : good_matches) {
            obj.push_back(mObjectKeypoints[match.queryIdx].pt);
            scene.push_back(mSceneKeypoints[i][match.trainIdx].pt);
        }

        cv::Mat H = findHomography(obj, scene, cv::RANSAC);
        if (H.empty()) {
            printf("\nHomography is empty: %i\n", i);
            continue;
        }

        //-- Step 6: Get the corners from the object.jpeg
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cv::Point2f(0, 0);
        obj_corners[1] = cv::Point2f((float) mObjectImg.cols, 0);
        obj_corners[2] = cv::Point2f((float) mObjectImg.cols, (float) mObjectImg.rows);
        obj_corners[3] = cv::Point2f(0, (float) mObjectImg.rows);
        std::vector<cv::Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, H);

        //-- Step 7: Draw lines between the corners of scene
        line(img_matches, scene_corners[0] + cv::Point2f(mObjectImg.cols, 0),
             scene_corners[1] + cv::Point2f(mObjectImg.cols, 0), cv::Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[1] + cv::Point2f(mObjectImg.cols, 0),
             scene_corners[2] + cv::Point2f(mObjectImg.cols, 0), cv::Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[2] + cv::Point2f(mObjectImg.cols, 0),
             scene_corners[3] + cv::Point2f(mObjectImg.cols, 0), cv::Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[3] + cv::Point2f(mObjectImg.cols, 0),
             scene_corners[0] + cv::Point2f(mObjectImg.cols, 0), cv::Scalar(0, 255, 0), 4);

        printf("\rFile %s is ready [%i/%zu are ready(%i skipped).]", filename.c_str(),
               i + 1, mSize, (i - static_cast<int>(mReadyImgs.size())));

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        mReadyImgs.push_back({filename, img_matches});

        //time for some metrics
        //first metric

        double pointsInside = 0;
        for (const auto &p: scene) {
            double check = pointPolygonTest(scene_corners, p, false);
            if (check >= 0) {
                pointsInside += 1;
            }
        }
        double pointsRatio = pointsInside / scene.size();

        //second metric
        double sumDistance = 0;
        for (const auto &match: good_matches) {
            sumDistance += match.distance;
        }
        double distanceRatio = sumDistance / good_matches.size();
        //third metric
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);

        //size
        int width = mSceneImgs[i].second.cols;
        int height = mSceneImgs[i].second.rows;

        Metrics data({filename, pointsRatio, distanceRatio, time.count(), width, height});
        mMetrics.push_back(data);
    }
}

void DescriptorProcessor::detectAndCompute(cv::Mat image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor) {
    mSURFDetector->detectAndCompute(image, cv::noArray(), keypoints, descriptor);
    if (descriptor.type() != CV_32F) {
        descriptor.convertTo(descriptor, CV_32F);
    }
}

std::vector<std::string> const DescriptorProcessor::findSceneImages(const std::string &path) {
    std::vector<std::string> res;
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        std::string filename = entry.path().filename().string();
        if (filename.find("scene") != std::string::npos) {
            res.push_back(filename);
        }

    }
    std::sort(res.begin(), res.end(), [](std::string vec1, std::string vec2) {
        return vec1 < vec2;
    });
    return res;
}

size_t DescriptorProcessor::getSize() {
    return mSize;
}

void DescriptorProcessor::setHaussian(int haussian) {
    mMinHaussian = haussian;
}

void DescriptorProcessor::saveMetricsToFile(std::string &filename) {
    CSVWriter writer(filename);
    std::vector<std::string> headers = {"Filename", "Average matched points", "Average distance", "Average process time (μs)", "Size"};
    writer.addDataInRow(headers.begin(), headers.end());
    for(auto & m: mMetrics){
        std::vector<std::string> dataToAdd = m.toVector();
        writer.addDataInRow(dataToAdd.begin(), dataToAdd.end());
    }
}

std::vector<std::string> DescriptorProcessor::Metrics::toVector() const {
    std::vector<std::string> res;
    res.push_back(filename);
    res.push_back(std::to_string(pointsInsideAvg));
    res.push_back(std::to_string(distanceAvg));
    res.push_back(std::to_string(duration));
    res.push_back(std::to_string(width) + "x" + std::to_string(height));
    return res;
}
