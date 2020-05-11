#ifndef CAMERA_FUSION_EGOVEHICLE_H
#define CAMERA_FUSION_EGOVEHICLE_H

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
#include "FeaturePointMatcher2D.h"
#include "yaml-cpp/yaml.h"


class EgoVehicle {
public:
    EgoVehicle();
    ~EgoVehicle();
    void SetParameters();
    void SetDataBuffer(DataFrame frame);
    void EraseFirstElement();
    std::string GetImgBasePath();
    std::string GetImgFileType();
    std::string GetYoloClassesFile();
    std::string GetYoloModelConfiguration();
    std::string GetYoloModelWeights();
    std::string GetLidarPrefix();
    std::string GetLidarFileType();
    std::vector<std::string> GetDetectorTypes();
    std::vector<std::string> GetDescriptorTypes();
    std::vector<std::string> GetMatcherTypes();
    std::vector<std::string> GetSelectorTypes();
    std::vector<DataFrame> GetDataBuffer();

    void UpdateLidarWithROI();
    void GetDetectedObjects();
    void GetMatches();
    void PushMatches();
    void Matching3DboundingBoxes();
    void ParralelExecutionGetMatches();
    void TrackAssociation();
    void ComputeTTCLidar(const double sensorFrameRate, bool bvis);
    void ClearTracks();

private:
    // calibration data for camera and lidar
    cv::Mat P_rect_00;                                   // 3x4 projection matrix after rectification
    cv::Mat R_rect_00;                                   // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT;                                          // rotation matrix and translation vector

    std::vector<DataFrame> dataBuffer;                   // buffer for image
    bool bVis{false};                                    // visualize results
    const float shrinkFactor{ 0.10 };

    // configuration paths
    std::string imgBasePath;
    std::string imgFileType;
    std::string yoloClassesFile;
    std::string yoloModelConfiguration;
    std::string yoloModelWeights;
    std::string lidarPrefix;
    std::string lidarFileType;

    // object detection models
    YoloV3 yoloV3_;                                         // object detection model

    // 2D feature matches
    FeaturePointMatcher2D matcher2D_;

    // descriptors and detectors used
    std::vector<std::string> detectorTypes;
    std::vector<std::string> descriptorTypes;
    std::vector<std::string> matcherTypes;
    std::vector<std::string> selectorTypes;
    std::string descriptorFamily;


};

#endif //CAMERA_FUSION_EGOVEHICLE_H
