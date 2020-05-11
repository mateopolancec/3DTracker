#include "YoloV3.h"


YoloV3::YoloV3()  {

    const YAML::Node config = YAML::LoadFile(std::string("../dat/yolo/yolo_params.yaml"));
    this->classesFile_ = config["object_detection"]["yoloClassesFile"].as<std::string>();
    this->modelConfiguration_ = config["object_detection"]["yoloModelConfiguration"].as<std::string>();
    this->modelWeights_ = config["object_detection"]["yoloModelWeights"].as<std::string>();
}


YoloV3::~YoloV3() {}


void YoloV3::LoadModel() {

    // load class names from file
    std::ifstream ifs(classesFile_.c_str());
    std::string line;

    while (getline(ifs, line)) { classes_.push_back(line); }

    // load neural network
    net_ = cv::dnn::readNetFromDarknet(modelConfiguration_, modelWeights_) ;
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // generate 4D blob from input image
    constexpr bool swapRB{ false};
    constexpr bool crop{ false };
    constexpr double scalefactor{ 1 / 255.0 };

    cv::Size size{ cv::Size(416, 416) };
    cv::Scalar mean{ cv::Scalar(0, 0, 0) };
    cv::dnn::blobFromImage(*img_, blob_, scalefactor, size, mean, swapRB, crop);
}


void YoloV3::GetBoundingBoxes() {
    // Get names of output layers
    std::vector<cv::String> names;
    std::vector<int> outLayers{ net_.getUnconnectedOutLayers() }; // get  indices of  output layers, i.e.  layers
    // with unconnected outputs
    std::vector<cv::String> layersNames{ net_.getLayerNames() };  // get  names of all layers in the network
    names.resize(outLayers.size());

    // Get the names of the output layers in names
    for (size_t i{ 0 }; i < outLayers.size(); ++i) { names[i] = layersNames[outLayers[i] - 1]; }

    // invoke forward propagation through network
    net_.setInput(blob_);
    net_.forward(netOutput_, names);

    // Scan through all bounding boxes and keep only the ones with high confidence
    for (size_t i{ 0 }; i < netOutput_.size(); ++i) {

        float *data{ reinterpret_cast<float*>(netOutput_[i].data) };

        for (int j = 0; j < netOutput_[i].rows; ++j, data += netOutput_[i].cols) {
            cv::Mat scores{ netOutput_[i].row(j).colRange(5, netOutput_[i].cols) };
            cv::Point classId;
            double confidence;

            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classId);

            if (confidence > confThreshold_) {
                cv::Rect box;
                int cx{ static_cast<int>(data[0] * img_->cols) };
                int cy{ static_cast<int>(data[1] * img_->rows) };

                box.width = static_cast<int>(data[2] * img_->cols);
                box.height = static_cast<int>(data[3] * img_->rows);

                box.x = cx - box.width / 2;  // left
                box.y = cy - box.height / 2; // top

                boxes_.push_back(box);
                classIds_.push_back(classId.x);
                confidences_.push_back((float)confidence);
            }
        }
    }
}


void YoloV3::PerformNonMaximaSuppression() {

    std::vector<int> indices;
    int counter = 0;
    cv::dnn::NMSBoxes(boxes_, confidences_, confThreshold_, nmsThreshold_, indices);

    for (auto it = indices.begin(); it != indices.end(); ++it) {
        BoundingBox bBox;
        bBox.roi = boxes_[*it];
        bBox.classID = classIds_[*it];
        bBox.confidence = confidences_[*it];
        bBox.boxID = counter;
        counter += 1;
        boundingBoxes_->push_back(std::move(bBox));
    }
}


void YoloV3::Inference(std::vector<BoundingBox> *boundingBoxes, cv::Mat *img) {
    this->boundingBoxes_ = boundingBoxes;
    this->img_ = img;
    LoadModel();
    GetBoundingBoxes();
    PerformNonMaximaSuppression();
}

