#ifndef CAMERA_FUSION_OBJECTDETECTION_H
#define CAMERA_FUSION_OBJECTDETECTION_H

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/dnn.hpp>

#include "dataStructures.h"

class ObjectDetection {

public:

    ObjectDetection() {}

    ~ObjectDetection() {};

    virtual void LoadModel() = 0;
    virtual void Inference(std::vector<BoundingBox> *boundingBoxes, cv::Mat *img) = 0;


protected:
    cv::Mat* img_;
    std::vector<std::string> classes_;
    std::string modelConfiguration_;
    std::string modelWeights_;
    std::vector<BoundingBox>* boundingBoxes_;
    std::string classesFile_;
};

#endif //CAMERA_FUSION_OBJECTDETECTION_H
