#ifndef dataStructures_h
#define dataStructures_h

// standard libraries
#include <map>
#include <vector>
#include <iostream>

// external libraries
#include <opencv2/core.hpp>


struct LidarPoint {    // single lidar point in space
    double x, y, z, r; // x,y,z in [m], r is point reflectivity
};

// BoundingBox around a classified object (contains both 2D and 3D data)
class BoundingBox {

public:
    BoundingBox() = default;
    ~BoundingBox() = default;

    int boxID;                                              // unique identifier for this bounding box
    int classID;                                            // ID based on class file provided to YOLO framework
    double confidence;                                      // classification trust

    std::vector<LidarPoint> lidarPoints;                    // Lidar 3D points which project into 2D image roi
    std::vector<cv::KeyPoint> keypoints;                    // keypoints enclosed by 2D roi
    std::vector<cv::DMatch> kptMatches;                     // keypoint matches enclosed by 2D roi
    cv::Rect roi;                                           // 2D region-of-interest in image coordinates
};


// DataFrame represents the available sensor information at the same time instance
class DataFrame {
public:
    DataFrame() {
        this->counter = 0;
        this->cameraImg = std::make_shared<cv::Mat>();
        this->keypoints = std::make_shared<std::vector<cv::KeyPoint>>(std::initializer_list<cv::KeyPoint>{ });
        this->descriptors = std::make_shared<cv::Mat>();
        this->kptMatches = std::make_shared<std::vector<cv::DMatch>>(std::initializer_list<cv::DMatch>{ });
        this->lidarPoints = std::make_shared<std::vector<LidarPoint>>(std::initializer_list<LidarPoint>{ });
        this->boundingBoxes = std::make_shared<std::vector<BoundingBox>>(std::initializer_list<BoundingBox>{ });
        this->tracks = std::make_shared<std::vector<BoundingBox>>(std::initializer_list<BoundingBox>{ });
    };

    ~DataFrame() = default;

    std::shared_ptr<cv::Mat> cameraImg;                     // camera image
    std::shared_ptr<std::vector<cv::KeyPoint>> keypoints;   // 2D keypoints within camera image
    std::shared_ptr<cv::Mat> descriptors;                   // keypoint descriptors
    std::shared_ptr<std::vector<cv::DMatch>> kptMatches;    // keypoint matches between previous and current frame
    std::shared_ptr<std::vector<LidarPoint>> lidarPoints;

    std::shared_ptr<std::vector<BoundingBox>> boundingBoxes; // ROI around detected objects in 2D image coordinates
    std::shared_ptr<std::vector<BoundingBox>> tracks;
    std::map<int, int> bbMatches;                            // bounding box matches between previous and current frame
    int counter;

    // bounding box matches from two subsequent frames
    std::map<int, int> bbBestMatches_;

    DataFrame(const DataFrame &source)
    {
        // deep copy
        this->cameraImg = std::make_shared<cv::Mat>(*source.cameraImg);
        this->keypoints = std::make_shared<std::vector<cv::KeyPoint>>(*source.keypoints);
        this->descriptors = std::make_shared<cv::Mat>(*source.descriptors);
        this->kptMatches = std::make_shared<std::vector<cv::DMatch>>(*source.kptMatches);
        this->lidarPoints = std::make_shared<std::vector<LidarPoint>>(*source.lidarPoints);
        this->boundingBoxes = std::make_shared<std::vector<BoundingBox>>(*source.boundingBoxes);
        this->counter = source.counter;
        this->bbMatches = source.bbMatches;

    }

    DataFrame(DataFrame &&source)
    {
        this->cameraImg = std::move( source.cameraImg );
        this->keypoints = std::move( source.keypoints );
        this->descriptors = std::move( source.descriptors );
        this->kptMatches = std::move( source.kptMatches );
        this->lidarPoints = std::move( source.lidarPoints );
        this->boundingBoxes = std::move( source.boundingBoxes );
        this->bbMatches = source.bbMatches;
        this->counter = source.counter;
        source.bbMatches = {};
        source.counter = 0;
    }

    DataFrame &operator=(const DataFrame &source)
    {
        if (this == &source)
        {
            return *this;
        }

        this->cameraImg = std::make_shared<cv::Mat>(*source.cameraImg);
        this->keypoints = std::make_shared<std::vector<cv::KeyPoint>>(*source.keypoints);
        this->descriptors = std::make_shared<cv::Mat>(*source.descriptors);
        this->kptMatches = std::make_shared<std::vector<cv::DMatch>>(*source.kptMatches);
        this->lidarPoints = std::make_shared<std::vector<LidarPoint>>(*source.lidarPoints);
        this->boundingBoxes = std::make_shared<std::vector<BoundingBox>>(*source.boundingBoxes);
        this->bbMatches = source.bbMatches;
        this->counter = source.counter;
        return *this;
    }

    DataFrame &operator=(DataFrame &&source)
    {

        if (this == &source)
        {
            return *this;
        }

        this->cameraImg = std::move( source.cameraImg );
        this->keypoints = std::move( source.keypoints );
        this->descriptors = std::move( source.descriptors );
        this->kptMatches = std::move( source.kptMatches );
        this->lidarPoints = std::move( source.lidarPoints );
        this->boundingBoxes = std::move( source.boundingBoxes );
        this->bbMatches = source.bbMatches;
        this->counter = source.counter;
        source.bbMatches = {};
        source.counter = 0;
        return *this;
    }

};

#endif /* dataStructures_h */
