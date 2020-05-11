#include "FeaturePointMatcher2D.h"
#include "EgoVehicle.h"


FeaturePointMatcher2D::FeaturePointMatcher2D() {
    this->keypointsPrev_ = std::make_shared<std::vector<cv::KeyPoint>>(std::initializer_list<cv::KeyPoint>{ });
    this->keypointsNext_ = std::make_shared<std::vector<cv::KeyPoint>>(std::initializer_list<cv::KeyPoint>{ });
    this->descriptorsPrev_ = std::make_shared<cv::Mat>();
    this->descriptorsNext_ = std::make_shared<cv::Mat>();
    this->kptMatches_ = std::make_shared<std::vector<cv::DMatch>>(std::initializer_list<cv::DMatch>{ });
}


FeaturePointMatcher2D::~FeaturePointMatcher2D() {}


void FeaturePointMatcher2D::seEgoVehicleInterface(EgoVehicle *egoVehicle) {
    this->egoVehicle_ = egoVehicle;
}


void FeaturePointMatcher2D::PushKepoyints(std::shared_ptr<std::vector<cv::KeyPoint>> keypoints, std::string order,
                                          std::string detectorType) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (order.compare("prev") == 0) {
        this->keypointsMapPrev_[detectorType] = std::move(keypoints);
    }
    else {
        this->keypointsMapNext_[detectorType] = std::move(keypoints);
    }
}


std::shared_ptr<std::vector<cv::KeyPoint>>* FeaturePointMatcher2D::GetKepoyints(std::string order, std::string detectorType) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (order.compare("prev") == 0) {
        return &keypointsMapPrev_[detectorType];
    }
    else {
        return &keypointsMapNext_[detectorType];
    }
}


void FeaturePointMatcher2D::PushDescriptors(std::shared_ptr<cv::Mat> descriptors, std::string order,
                                            std::string descriptorType) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (order.compare("prev") == 0) {
        this->descriptorsMapPrev_[descriptorType] = std::move(descriptors);
    }
    else {
        this->descriptorsMapNext_[descriptorType] = std::move(descriptors);
    }

}

std::shared_ptr<cv::Mat>* FeaturePointMatcher2D::GetDescriptors(std::string order, std::string descriptorType) {

    std::lock_guard<std::mutex> lock(mutex_);

    if (order.compare("prev") == 0) {
        return &descriptorsMapPrev_[descriptorType];
    }
    else {
        return &descriptorsMapNext_[descriptorType];
    }

}

void FeaturePointMatcher2D::PushMatches(std::shared_ptr<std::vector<cv::DMatch>> kptMatches, std::string detectorType,
                                        std::string descriptorType) {

    std::lock_guard<std::mutex> lock(mutex_);
    this->matchesMap_[descriptorType].first = detectorType;
    this->matchesMap_[descriptorType].second = std::move(kptMatches);

}


void FeaturePointMatcher2D::Process(std::string detectorType, std::string descriptorType, std::string matcherType,
                                    cv::Mat* img_prev, cv::Mat* img_next, const std::string descriptorFamily,
                                    const std::string selectorType) {

    // if(this->kptMatches_->empty()) ensures that all threads will enter in this condition in first iteration
    if(this->kptMatches_->empty()) {

        DetectImageKeypoints(detectorType, img_prev, "prev");
        DetectImageKeypoints(detectorType, img_next, "next");
        DescriptorImageKeypoints(detectorType, descriptorType, img_prev, "prev");
        DescriptorImageKeypoints(detectorType, descriptorType, img_next, "next");
        matchDescriptors(descriptorFamily, matcherType, selectorType, detectorType, descriptorType);

    }
    else {

        std::unique_lock<std::mutex> lcx(mtx_);

        if(!matchesMap_.empty()) {
            this->keypointsMapPrev_ = this->keypointsMapNext_;
            this->descriptorsMapPrev_ = this->descriptorsMapNext_;
            this->keypointsMapNext_.clear();
            this->descriptorsMapNext_.clear();
            this->matchesMap_.clear();
            this->kptMatches_->clear();
        }

        lcx.unlock();

        DetectImageKeypoints(detectorType, img_next, "next");
        DescriptorImageKeypoints(detectorType, descriptorType, img_next, "next");
        matchDescriptors(descriptorFamily, matcherType, selectorType, detectorType, descriptorType);

    }
}

void FeaturePointMatcher2D::GetBestMatches() {

    for (auto const& x : matchesMap_)
    {
        if(kptMatches_->empty()) {
            descriptor_ = x.first;
            detector_ = matchesMap_[x.first].first;
            kptMatches_ = matchesMap_[x.first].second;
            keypointsPrev_ = keypointsMapPrev_[detector_];
            keypointsNext_ = keypointsMapNext_[detector_];
            descriptorsPrev_ = descriptorsMapPrev_[descriptor_];
            descriptorsNext_ = descriptorsMapNext_[descriptor_];
        }
        else {
            if(matchesMap_[x.first].second->size() > kptMatches_->size()) {
                descriptor_ = x.first;
                detector_ = matchesMap_[x.first].first;
                kptMatches_ = matchesMap_[x.first].second;
                keypointsPrev_ = keypointsMapPrev_[detector_];
                keypointsNext_ = keypointsMapNext_[detector_];
                descriptorsPrev_ = descriptorsMapPrev_[descriptor_];
                descriptorsNext_ = descriptorsMapNext_[descriptor_];
            }
        }
    }
}


void FeaturePointMatcher2D::MatchFeaturePoints(std::vector<std::string> detectorTypes, std::vector<std::string> descriptorTypes,
        std::vector<std::string> matcherTypes, cv::Mat* img_prev, cv::Mat* img_next, const std::string descriptorFamily,
                                               const std::string selectorType) {

    // sleep at every iteration to reduce CPU usage
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    threads_.reserve(descriptorTypes.size());

    // run parallel execution
    for (int i = 0; i < descriptorTypes.size(); ++i) {
        threads_.emplace_back (std::thread(&FeaturePointMatcher2D::Process, this, detectorTypes[1], descriptorTypes[i],
                                           matcherTypes[0], img_prev, img_next, descriptorFamily, selectorType));
    }

    // wait to threads finish tasks
    for(auto&& i : threads_) {
        i.join();
    }

    GetBestMatches();
    egoVehicle_->PushMatches();

    // clear vector of threads
    threads_.clear();
}


void FeaturePointMatcher2D::DetectImageKeypoints(std::string detectorType, cv::Mat* img,
                                                 std::string order) {

    // convert current image to grayscale
    std::shared_ptr<cv::Mat> imgGray = std::make_shared<cv::Mat>();
    cv::cvtColor(*img, *imgGray, cv::COLOR_BGR2GRAY);

    // extract 2D keypoints from current image
    std::shared_ptr<std::vector<cv::KeyPoint>> keypoints = std::make_shared<std::vector<cv::KeyPoint>>(std::initializer_list<cv::KeyPoint>{ });
    if (detectorType.compare("SHITOMASI") == 0) {
        detKeypointsShiTomasi(keypoints.get(), imgGray.get(), false);

    } else if (detectorType.compare("HARRIS") == 0) {
        detKeypointsHarris(keypoints.get(), imgGray.get(), false);

        if (bLimitKpts_) {
            constexpr int maxKeypoints{50};
            // there is no response info, so keep the first 50
            // as they are sorted in descending quality order
            if (detectorType.compare("SHITOMASI") == 0) {
                keypoints->erase(begin(*keypoints) + maxKeypoints, end(*keypoints));
            }
            cv::KeyPointsFilter::retainBest(*keypoints, maxKeypoints);
        }
    }
    PushKepoyints(std::move(keypoints), order, detectorType);
}


void FeaturePointMatcher2D::DescriptorImageKeypoints(std::string detectorType, std::string descriptorType, cv::Mat* img,
                                                     std::string order) {

    if (order.compare("prev") == 0) {
        descKeypoints(img, detectorType, descriptorType, order);
    }
    else {
        descKeypoints(img, detectorType, descriptorType, order);
    }

}


void FeaturePointMatcher2D::detKeypointsHarris(std::vector<cv::KeyPoint> *keypoints, cv::Mat *img, const bool bVis) {

    constexpr int blockSize{2};
    constexpr int apertureSize{3};
    constexpr int minResponse{100};
    constexpr double k{0.04};
    constexpr double overlapThreshold{0.0};
    constexpr int scaledApertureSize{apertureSize * 2};

    cv::Mat dst, dstNorm, dstNormScaled;
    dst = cv::Mat::zeros(img->size(), CV_32FC1);

    cv::cornerHarris(*img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dstNorm, dstNormScaled);

    bool foundOverlap{false};

    for (int i{0}; i < dstNorm.rows; i++) {
        for (int j{0}; j < dstNorm.cols; j++) {

            const int response{static_cast<int>(dstNorm.at<float>(i, j))};

            if (response > minResponse) {
                cv::KeyPoint point;
                point.pt = cv::Point2f(i, j);
                point.size = scaledApertureSize;
                point.response = response;
                point.class_id = 0;

                foundOverlap = false;

                for (auto iterator{std::begin(*keypoints)}; iterator != std::end(*keypoints); iterator++) {
                    if (cv::KeyPoint::overlap(point, (*iterator)) > overlapThreshold) {
                        foundOverlap = true;

                        if (point.response > (*iterator).response) {
                            *iterator = point;
                            break;
                        }
                    }
                }

                if (!foundOverlap) { keypoints->push_back(point); }
            }
        }
    }
}

void FeaturePointMatcher2D::detKeypointsShiTomasi(std::vector<cv::KeyPoint> *keypoints, cv::Mat *img, const bool bVis) {

    // compute detector parameters based on image size
    //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    constexpr int blockSize{ 4 };
    constexpr double overlapThreshold{ 0.0 }; // max. permissible overlap between two features in %
    constexpr double minDistance{ ((1.0 - overlapThreshold) * blockSize) };

    // max. num. of keypoints
    const int maxCorners{ static_cast<int>(img->rows * img->cols / std::max(1.0, minDistance)) };

    constexpr double qualityLevel{ 0.01 }; // minimal accepted quality of image corners
    constexpr double k{ 0.04 };

    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(*img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize,
              false, k);

    // add corners to result std::vector
    for (auto iterator{ std::begin(corners) }; iterator != std::end(corners); ++iterator) {
        cv::KeyPoint keyPoint;
        keyPoint.pt = cv::Point2f((*iterator).x, (*iterator).y);
        keyPoint.size = blockSize;
        keypoints->push_back(keyPoint);
    }
}


void FeaturePointMatcher2D::descKeypoints(cv::Mat* img, std::string detectorType, std::string descriptorType, std::string order) {
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;

    /// selection based on descriptorType / -> BRIEF, ORB, FREAK, AKAZE, SIFT

    if (descriptorType.compare("BRISK") == 0) {
        // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        constexpr float patternScale{ 1.0f };
        constexpr int threshold{ 30 }; // FAST/AGAST detection threshold score.
        constexpr int octaves{ 3 };    // detection octaves (use 0 to do single scale)

        extractor = cv::BRISK::create(threshold, octaves, patternScale);

    } else if (descriptorType.compare("ORB") == 0) {
        extractor = cv::ORB::create();

    }

    // perform feature description
    std::shared_ptr<cv::Mat> descriptors = std::make_shared<cv::Mat>();

    extractor->compute(*img, *(*GetKepoyints(order, detectorType)).get(), *descriptors.get());

    PushDescriptors(std::move(descriptors), order, descriptorType);
}

void FeaturePointMatcher2D::matchDescriptors(const std::string descriptorFamily, const std::string matcherType,
                                             const std::string selectorType, std::string detectorType,
                                             std::string descriptorType) {

    // configure matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::shared_ptr<std::vector<cv::DMatch>> kptMatches = std::make_shared<std::vector<cv::DMatch>>(std::initializer_list<cv::DMatch>{ });
    const std::string next = "next";
    const std::string prev = "prev";

    if (matcherType.compare("MAT_BF") == 0) {
        constexpr bool crossCheck{ false };
        const int normType{ ((descriptorFamily.compare("DES_BINARY") == 0) ? cv::NORM_HAMMING : cv::NORM_L2) };

        matcher = cv::BFMatcher::create(normType, crossCheck);

    } else if (matcherType.compare("MAT_FLANN") == 0) {
        if ((*GetDescriptors(next, descriptorType)).get()->type() != CV_32F) {
            (*GetDescriptors(next, descriptorType)).get()->convertTo(*(*GetDescriptors(next, descriptorType)).get(), CV_32F); }

        if ((*GetDescriptors(prev, descriptorType)).get()->type() != CV_32F) {
            (*GetDescriptors(prev, descriptorType)).get()->convertTo(*(*GetDescriptors(prev, descriptorType)).get(), CV_32F); }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    if (selectorType.compare("SEL_NN") == 0) {
        matcher->match(*(*GetDescriptors(prev, descriptorType)).get(),
                *(*GetDescriptors(next, descriptorType)).get(), *kptMatches.get()); // Finds the best match for each descriptor in desc1

    } else if (selectorType.compare("SEL_KNN") == 0) {
        std::vector<std::vector<cv::DMatch>> knnMatches;

        matcher->knnMatch(*(*GetDescriptors(prev, descriptorType)).get(),
                          *(*GetDescriptors(next, descriptorType)).get(), knnMatches, 2);

        constexpr float threshold{ 0.8 };

        for (auto iterator{ std::begin(knnMatches) }; iterator != std::end(knnMatches); iterator++) {
            if ((*iterator).at(0).distance < (threshold * (*iterator).at(1).distance)) {
                kptMatches->push_back((*iterator).at(0));
            }
        }
    }
    PushMatches(std::move(kptMatches), detectorType, descriptorType);
}


std::vector<cv::DMatch> FeaturePointMatcher2D::GetMatches() {
    return *kptMatches_;
}


std::vector<cv::KeyPoint> FeaturePointMatcher2D::GetKeypointsPrev() {
    return *keypointsPrev_;
}

std::vector<cv::KeyPoint> FeaturePointMatcher2D::GetKeypointsNext() {
    return *keypointsNext_;
}


cv::Mat FeaturePointMatcher2D::GetDescriptorsPrev() {
    return *descriptorsPrev_.get();
}


cv::Mat FeaturePointMatcher2D::GetDescriptorsNext() {
    return *descriptorsNext_.get();
}
