#ifndef CAMERA_FUSION_FEATUREPOINTMATCHER2D_H
#define CAMERA_FUSION_FEATUREPOINTMATCHER2D_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <sstream>
#include <vector>
#include <thread>
#include <condition_variable>
#include <queue>
#include <functional>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include "lidarData.hpp"
#include "YoloV3.h"
#include "dataStructures.h"
#include "yaml-cpp/yaml.h"

// forward declaration to avoid include cycle
class EgoVehicle;

class FeaturePointMatcher2D {
public:
    FeaturePointMatcher2D();
    ~FeaturePointMatcher2D();

    void DetectImageKeypoints(std::string detectorType, cv::Mat* img, std::string order);
    void DescriptorImageKeypoints(std::string detectorType, std::string descriptorType, cv::Mat* img, std::string order);
    void MatchFeaturePoints(std::vector<std::string> detectorTypes, std::vector<std::string> descriptorTypes,
                            std::vector<std::string> matcherTypes, cv::Mat* img_prev, cv::Mat* img_next,
                            const std::string descriptorFamily, const std::string selectorType);

    // basic functionality
    void detKeypointsHarris(std::vector<cv::KeyPoint> *keypoints, cv::Mat *img, const bool bVis);
    void detKeypointsShiTomasi(std::vector<cv::KeyPoint> *keypoints, cv::Mat *img, const bool bVis);
    void descKeypoints(cv::Mat* img, std::string detectorType, std::string descriptorType, std::string order);
    void matchDescriptors(const std::string descriptorFamily, const std::string matcherType,
                          const std::string selectorType, std::string detectorType, std::string descriptorType);
    void seEgoVehicleInterface(EgoVehicle *egoVehicle);
    std::vector<cv::DMatch> GetMatches();
    void Process(std::string detectorType, std::string descriptorType, std::string matcherType,
                 cv::Mat* img_prev, cv::Mat* img_next, const std::string descriptorFamily, const std::string selectorType);
    void PushKepoyints(std::shared_ptr<std::vector<cv::KeyPoint>> keypoints, std::string order, std::string detectorType);
    void PushDescriptors(std::shared_ptr<cv::Mat> descriptors, std::string order, std::string descriptorType);
    std::shared_ptr<cv::Mat>* GetDescriptors(std::string order, std::string descriptorType);
    std::shared_ptr<std::vector<cv::KeyPoint>>* GetKepoyints(std::string order, std::string detectorType);
    void PushMatches(std::shared_ptr<std::vector<cv::DMatch>> kptMatches, std::string detectorType, std::string descriptorType);
    void GetBestMatches();
    std::vector<cv::KeyPoint> GetKeypointsPrev();
    std::vector<cv::KeyPoint> GetKeypointsNext();
    cv::Mat GetDescriptorsPrev();
    cv::Mat GetDescriptorsNext();

private:

    std::shared_ptr<std::vector<cv::KeyPoint>> keypointsPrev_;
    std::shared_ptr<std::vector<cv::KeyPoint>> keypointsNext_;
    std::shared_ptr<cv::Mat> descriptorsPrev_;
    std::shared_ptr<cv::Mat> descriptorsNext_;
    std::shared_ptr<std::vector<cv::DMatch>> kptMatches_;

    std::string matcherType;
    bool bLimitKpts_{false};

    EgoVehicle *egoVehicle_;

    // concurrent programing;
    std::mutex mutex_;
    std::mutex mtx_;

    // vector of running threads
    std::vector <std::thread> threads_;

    // map result with keypoint, descriptor and matches types
    std::map<std::string, std::shared_ptr<std::vector<cv::KeyPoint>>> keypointsMapPrev_;
    std::map<std::string, std::shared_ptr<cv::Mat>> descriptorsMapPrev_;
    std::map<std::string, std::shared_ptr<std::vector<cv::KeyPoint>>> keypointsMapNext_;
    std::map<std::string, std::shared_ptr<cv::Mat>> descriptorsMapNext_;
    std::map<std::string, std::pair<std::string, std::shared_ptr<std::vector<cv::DMatch>>>> matchesMap_;

    // best matches detector and descriptor type
    std::string detector_;
    std::string descriptor_;


};


#endif //CAMERA_FUSION_FEATUREPOINTMATCHER2D_H
