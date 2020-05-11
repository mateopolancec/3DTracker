#include "EgoVehicle.h"
#include "camFusion.hpp"


EgoVehicle::EgoVehicle() {
    matcher2D_.seEgoVehicleInterface(this);
}


EgoVehicle::~EgoVehicle() {

}


void EgoVehicle::SetParameters() {

    const YAML::Node config = YAML::LoadFile(std::string("../config/params.yaml"));

    imgBasePath = config["camera"]["imgBasePath"].as<std::string>();
    imgFileType = config["camera"]["imgFileType"].as<std::string>();
    lidarPrefix = config["lidar"]["lidarPrefix"].as<std::string>();
    lidarFileType = config["lidar"]["lidarFileType"].as<std::string>();
    detectorTypes = config["detectorTypes"].as<std::vector<std::string>>();
    descriptorTypes = config["descriptorTypes"].as<std::vector<std::string>>();
    matcherTypes = config["matcherTypes"].as<std::vector<std::string>>();
    selectorTypes = config["selectorTypes"].as<std::vector<std::string>>();

    descriptorFamily = "DES_BINARY";

    P_rect_00 = cv::Mat(3, 4, cv::DataType<double>::type);
    R_rect_00 = cv::Mat(4, 4, cv::DataType<double>::type);
    RT = cv::Mat(4, 4, cv::DataType<double>::type);

    RT.at<double>(0, 0) = 7.533745e-03;
    RT.at<double>(0, 1) = -9.999714e-01;
    RT.at<double>(0, 2) = -6.166020e-04;
    RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02;
    RT.at<double>(1, 1) = 7.280733e-04;
    RT.at<double>(1, 2) = -9.998902e-01;
    RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01;
    RT.at<double>(2, 1) = 7.523790e-03;
    RT.at<double>(2, 2) = 1.480755e-02;
    RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0;
    RT.at<double>(3, 1) = 0.0;
    RT.at<double>(3, 2) = 0.0;
    RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01;
    R_rect_00.at<double>(0, 1) = 9.837760e-03;
    R_rect_00.at<double>(0, 2) = -7.445048e-03;
    R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03;
    R_rect_00.at<double>(1, 1) = 9.999421e-01;
    R_rect_00.at<double>(1, 2) = -4.278459e-03;
    R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03;
    R_rect_00.at<double>(2, 1) = 4.351614e-03;
    R_rect_00.at<double>(2, 2) = 9.999631e-01;
    R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0;
    R_rect_00.at<double>(3, 1) = 0;
    R_rect_00.at<double>(3, 2) = 0;
    R_rect_00.at<double>(3, 3) = 1;

    P_rect_00.at<double>(0, 0) = 7.215377e+02;
    P_rect_00.at<double>(0, 1) = 0.000000e+00;
    P_rect_00.at<double>(0, 2) = 6.095593e+02;
    P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00;
    P_rect_00.at<double>(1, 1) = 7.215377e+02;
    P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00;
    P_rect_00.at<double>(2, 0) = 0.000000e+00;
    P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00;
    P_rect_00.at<double>(2, 3) = 0.000000e+00;


}

/* GETTERS AND SETTERS */

std::string EgoVehicle::GetImgBasePath() {
    return imgBasePath;
}

std::string EgoVehicle::GetImgFileType() {
    return imgFileType;
}

std::string EgoVehicle::GetYoloClassesFile() {
    return yoloClassesFile;
}

std::string EgoVehicle::GetYoloModelConfiguration() {
    return yoloModelConfiguration;
}

std::string EgoVehicle::GetYoloModelWeights() {
    return yoloModelWeights;
}

std::string EgoVehicle::GetLidarPrefix() {
    return lidarPrefix;
}

std::string EgoVehicle::GetLidarFileType() {
    return lidarFileType;
}

std::vector<std::string> EgoVehicle::GetDetectorTypes() {
    return detectorTypes;
}

std::vector<std::string> EgoVehicle::GetDescriptorTypes() {
    return descriptorTypes;
}

std::vector<std::string> EgoVehicle::GetMatcherTypes() {
    return matcherTypes;
}

std::vector<std::string> EgoVehicle::GetSelectorTypes() {
    return selectorTypes;
}

void EgoVehicle::SetDataBuffer(DataFrame frame) {
    dataBuffer.emplace_back(std::move(frame));
}

void EgoVehicle::EraseFirstElement() {
    dataBuffer.erase(dataBuffer.begin());
}

void EgoVehicle::UpdateLidarWithROI() {
    clusterLidarWithROI(dataBuffer.back().boundingBoxes.get(),
                        dataBuffer.back().lidarPoints.get(),
                        shrinkFactor, P_rect_00, R_rect_00, RT);
}

void EgoVehicle::GetDetectedObjects() {
    yoloV3_.Inference(dataBuffer.back().boundingBoxes.get(), dataBuffer.back().cameraImg.get());
}


std::vector<DataFrame> EgoVehicle::GetDataBuffer() {
    return dataBuffer;
}

void EgoVehicle::ParralelExecutionGetMatches() {

}

void EgoVehicle::GetMatches() {

    matcher2D_.MatchFeaturePoints(detectorTypes, descriptorTypes, matcherTypes,
                                  dataBuffer.front().cameraImg.get(), dataBuffer.back().cameraImg.get(),
                                  descriptorFamily, selectorTypes[0]);
}

void EgoVehicle::PushMatches() {

    if(this->dataBuffer.front().kptMatches->empty()) {
        *this->dataBuffer.back().kptMatches = matcher2D_.GetMatches();
        *this->dataBuffer.front().keypoints = matcher2D_.GetKeypointsPrev();
        *this->dataBuffer.back().keypoints = matcher2D_.GetKeypointsNext();
        *this->dataBuffer.front().descriptors = matcher2D_.GetDescriptorsPrev();
        *this->dataBuffer.back().descriptors = matcher2D_.GetDescriptorsNext();
    }
    else {
        if(this->dataBuffer.back().kptMatches->size() < matcher2D_.GetMatches().size()) {
            *this->dataBuffer.back().kptMatches = matcher2D_.GetMatches();
            *this->dataBuffer.back().keypoints = matcher2D_.GetKeypointsNext();
            *this->dataBuffer.back().descriptors = matcher2D_.GetDescriptorsNext();
        }
    }
}


void EgoVehicle::TrackAssociation() {

    std::map<int, int> bbBestMatches;

    matchBoundingBoxes(dataBuffer.back().kptMatches.get(),
                       bbBestMatches,
                       &dataBuffer.front(),
                       &dataBuffer.back());

    // associate bounding boxes between current and previous frame using keypoint matches
    // store matches in current data frame
    dataBuffer.back().bbBestMatches_ = bbBestMatches;
}

void EgoVehicle::ComputeTTCLidar(const double sensorFrameRate, bool bvis) {

    for (auto it1{(dataBuffer.back().bbBestMatches_.begin())}; it1 != dataBuffer.back().bbBestMatches_.end(); ++it1) {
        // find bounding boxes associates with current match

        BoundingBox prevBB, currBB;
        int boxID;
        double ttcLidar;

        for (auto it2{dataBuffer.back().boundingBoxes->begin()};
             it2 != dataBuffer.back().boundingBoxes->end(); ++it2) {

            // check wether current match partner corresponds to this BB
            if (it1->second == it2->boxID) {
                currBB = (*it2);
                currBB.lidarPoints = (*it2).lidarPoints;
                boxID = it2->boxID;
            }
        }

        for (auto it3{dataBuffer.front().boundingBoxes->begin()};
            it3 != dataBuffer.front().boundingBoxes->end(); ++it3) {
            // check wether current match partner corresponds to this BB
            if (it1->first == it3->boxID) {
                prevBB = (*it3);
                prevBB.lidarPoints = (*it3).lidarPoints;
            }
        }

        // compute TTC for current match
        // only compute TTC if we have Lidar points

        if (currBB.lidarPoints.size() > 0 &&
            prevBB.lidarPoints.size() > 0) {// only compute TTC if we have Lidar points

            computeTTCLidar(prevBB.lidarPoints, currBB.lidarPoints, sensorFrameRate, ttcLidar);

            if (bvis) {
                cv::Mat visImg{ dataBuffer.back().cameraImg->clone() };

                showLidarImgOverlay(visImg, currBB.lidarPoints, P_rect_00, R_rect_00, RT, &visImg);

                cv::rectangle(visImg,
                        cv::Point(currBB.roi.x, currBB.roi.y),
                        cv::Point(currBB.roi.x + currBB.roi.width,
                                currBB.roi.y + currBB.roi.height),
                                cv::Scalar(0, 255, 0),
                                2);

                char str[200];

                sprintf(str, "TTC Lidar : %f s,", ttcLidar);

                putText(visImg,
                        str,
                        cv::Point2f(80, 50),
                        cv::FONT_HERSHEY_PLAIN, 2,
                        cv::Scalar(0, 0, 255));

                constexpr char windowName[]{ "Final Results : TTC" };

                cv::namedWindow(windowName, 4);
                cv::imshow(windowName, visImg);
                std::cout << "Press key to continue to next frame" << std::endl;
                cv::waitKey(0);

            }
        }
    }
}

void EgoVehicle::ClearTracks() {
    this->dataBuffer.back().tracks->clear();
}









