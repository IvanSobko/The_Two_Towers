#include <iostream>
#include <vector>
#include <map>
#include <filesystem>
#include <chrono>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

//      відносна кількість правильно суміщених ознак
//      похибка локалізації (відстань між реальним розміщенням предмета в кадрі та розпізнаним)
//      відносний час обробки фото в залежності від розміру зображення



std::vector<std::string> const findSceneImages(const std::string &path) {
    std::vector<std::string> res;
    int i = 0;
    for (const auto &entry : std::filesystem::directory_iterator(path)) {
        i += 1;
        if (i % 3 != 0) {
            continue;
        }
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

void displayImages(std::vector<std::pair<std::string, cv::Mat>> images) {
    int num = 0;
    while (true) {
        cv::imshow(images[num].first, images[num].second);
        auto pressedKey = cv::waitKeyEx(0);
        if (pressedKey == 113) { // q to exit
            break;
        } else if (pressedKey == 63234) { // left arrow
            num = std::max(0, --num);
        } else if (pressedKey == 63235) { // right arrow
            num = std::min(static_cast<int>(images.size() - 1), ++num);
        }
        cv::destroyAllWindows();

    }
}

void processDescriptors(cv::Ptr<cv::xfeatures2d::SURF> detector, cv::Mat image,
                        std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptor) {
    detector->detectAndCompute(image, cv::noArray(), keypoints, descriptor);
    if (descriptor.type() != CV_32F) {
        descriptor.convertTo(descriptor, CV_32F);
    }
}


int main() {
    cv::Mat object = imread("../img/object.jpeg", cv::IMREAD_GRAYSCALE);

    std::vector<std::pair<std::string, cv::Mat>> sceneImages;
    std::string path = "../img/";
    std::vector<std::string> sceneImgFiles = findSceneImages(path);//{"../img/scene_1.jpeg"};
    for (const auto &file: sceneImgFiles) {
        cv::Mat res = imread(path + file, cv::IMREAD_GRAYSCALE);
        if (!res.data) {
            printf("Image %s not read", file.c_str());
            continue;
        }
        sceneImages.push_back({file, res});
    }

    size_t size = sceneImages.size();
    printf("Total photos to process: %zu.\n", size);
    if (!size) {
        return 1;
    }

    int minHessian = 400;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian);
    cv::Mat objectDescriptor;
    std::vector<cv::Mat> sceneDescriptors;
    sceneDescriptors.resize(size);
    std::vector<cv::KeyPoint> objectKeypoints;
    std::vector<std::vector<cv::KeyPoint>> sceneKeypoints;
    sceneKeypoints.resize(size);
    std::vector<std::pair<std::string, cv::Mat>> readyImg;
    cv::Ptr<cv::FlannBasedMatcher> matcher = cv::FlannBasedMatcher::create();

    std::map<std::string, double> pointsInsideAvg;
    std::map<std::string, double> distanceAvg;
    std::map<std::string, int64_t> duration;

    processDescriptors(detector, object, objectKeypoints, objectDescriptor);


    for (int i = 0; i < size; i++) {
        const std::string &filename = sceneImages[i].first;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        //-- Step 1: Detect the keypoints using SURF Detector
        processDescriptors(detector, sceneImages[i].second, sceneKeypoints[i], sceneDescriptors[i]);
        printf("\rDetected and computed %i photos.", i + 1);


        std::vector<std::vector<cv::DMatch>> knn_matches;

        if (sceneDescriptors[i].empty()) {
            printf("\nCouldn't find feature points for: %s\n", filename.c_str());
            continue;
        }

        //-- Step 2: Matching descriptor vectors with a FLANN based matcher
        matcher->knnMatch(objectDescriptor, sceneDescriptors[i], knn_matches, 2);

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
        drawMatches(object, objectKeypoints, sceneImages[i].second, sceneKeypoints[i],
                    good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        //-- Step 5: Localize the object
        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;
        for (const auto &match : good_matches) {
            obj.push_back(objectKeypoints[match.queryIdx].pt);
            scene.push_back(sceneKeypoints[i][match.trainIdx].pt);
        }

        cv::Mat H = findHomography(obj, scene, cv::RANSAC);
        if (H.empty()) {
            printf("\nHomography is empty: %i\n", i);
            continue;
        }

        //-- Step 6: Get the corners from the object.jpeg
        std::vector<cv::Point2f> obj_corners(4);
        obj_corners[0] = cv::Point2f(0, 0);
        obj_corners[1] = cv::Point2f((float) object.cols, 0);
        obj_corners[2] = cv::Point2f((float) object.cols, (float) object.rows);
        obj_corners[3] = cv::Point2f(0, (float) object.rows);
        std::vector<cv::Point2f> scene_corners(4);

        perspectiveTransform(obj_corners, scene_corners, H);

        //-- Step 7: Draw lines between the corners of scene
        line(img_matches, scene_corners[0] + cv::Point2f(object.cols, 0),
             scene_corners[1] + cv::Point2f(object.cols, 0), cv::Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[1] + cv::Point2f(object.cols, 0),
             scene_corners[2] + cv::Point2f(object.cols, 0), cv::Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[2] + cv::Point2f(object.cols, 0),
             scene_corners[3] + cv::Point2f(object.cols, 0), cv::Scalar(0, 255, 0), 4);
        line(img_matches, scene_corners[3] + cv::Point2f(object.cols, 0),
             scene_corners[0] + cv::Point2f(object.cols, 0), cv::Scalar(0, 255, 0), 4);

        printf("\rFile %s is ready [%i/%zu are ready(%i skipped).]", filename.c_str(),
               i + 1, size, (i - static_cast<int>(readyImg.size())));

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        readyImg.push_back({filename, img_matches});

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
        pointsInsideAvg.insert({filename, pointsRatio});

        //second metric
        double sumDistance = 0;
        for (const auto &match: good_matches) {
            sumDistance += match.distance;
        }
        double distanceRatio = sumDistance / good_matches.size();
        distanceAvg.insert({filename, distanceRatio});

        //third metric
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
        duration.insert({filename, time.count()});
    }

    for (const auto &p: distanceAvg) {
        printf("\nFor image %s average point distance: %f", p.first.c_str(), p.second);
    }
    displayImages(readyImg);

    return 0;
}