#ifndef CAMERA_FUSION_YOLOV3_H
#define CAMERA_FUSION_YOLOV3_H

#include "ObjectDetection.h"
#include "yaml-cpp/yaml.h"

class YoloV3: public ObjectDetection {
public:

    YoloV3();

    ~YoloV3();

    void LoadModel();
    void GetBoundingBoxes();
    void PerformNonMaximaSuppression();
    void Inference(std::vector<BoundingBox> *boundingBoxes, cv::Mat *img);

private:
    float confThreshold_{ 0.2 };
    float nmsThreshold_{ 0.4 };
    cv::Mat blob_;
    cv::dnn::Net net_;
    std::vector<cv::Mat> netOutput_;
    std::vector<int> classIds_;
    std::vector<float> confidences_;
    std::vector<cv::Rect> boxes_;
};



#endif //CAMERA_FUSION_YOLOV3_H
