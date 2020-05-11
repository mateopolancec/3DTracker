#include "ReceiveData.h"
#include "EgoVehicle.h"


ReceiveData::ReceiveData() {
    LoadParameters();
}

ReceiveData::~ReceiveData() {

}
void ReceiveData::VisualizeClusters(const DataFrame& frame) {

    // show results
    cv::Mat visImg = frame.cameraImg->clone();
    for (auto it{ std::begin(*frame.boundingBoxes) }; it != std::end(*frame.boundingBoxes); ++it) {

        // Draw rectangle displaying the bounding box
        int top = it->roi.y;
        int left = it->roi.x;
        int width = it->roi.width;
        int height = it->roi.height;

        cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(0, 255, 0), 2);

        std::string label = cv::format("%.2f", it->confidence);
        label = std::to_string((*frame.boundingBoxes)[(it->classID)].classID) + ":" + label;

        // Display label at the top of the bounding box
        int baseLine;

        cv::Size labelSize{ getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine) };

        top = std::max(top, labelSize.height);

        rectangle(visImg, cv::Point(left, top - round(1.5 * labelSize.height)),
                  cv::Point(left + round(1.5 * labelSize.width), top + baseLine),
                  cv::Scalar(255, 255, 255), cv::FILLED);

        cv::putText(visImg, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75, cv::Scalar(0, 0, 0), 1);
    }

    constexpr char windowName[]{ "Object classification" };
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, visImg);
    cv::waitKey(0); // wait for key to be pressed
}


void ReceiveData::LoadParameters() {

    /* INIT VARIABLES AND DATA STRUCTURES */
    std::string params_path{"../config/params.yaml"};
    egoVehicle_.SetParameters();

}

void ReceiveData::Update() {

    /* UPDATE EGO-VEHICLE TRACK MANAGEMENT */
    bool visClusters;
    double sensorFrameRate{ 10.0 / imgStepWidth };
    size_t numImages{ static_cast<size_t>(imgEndIndex - imgStartIndex) };
    int counter = 0;

    for (size_t imgIndex{ 0 }; imgIndex <= numImages; imgIndex += imgStepWidth) {
        std::ostringstream imgNumber, imgNumber_next;

        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;

        const std::string imgFullFilename{ egoVehicle_.GetImgBasePath() + imgNumber.str() + egoVehicle_.GetImgFileType() };

        // push image into data frame buffer
        DataFrame frame;
        frame.counter = 0;
        frame.cameraImg = std::make_shared<cv::Mat>(cv::imread(imgFullFilename));

        // load 3D Lidar points from file
        const std::string lidarFullFilename{ egoVehicle_.GetLidarPrefix() + imgNumber.str() + egoVehicle_.GetLidarFileType() };
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
        frame.lidarPoints = std::make_shared<std::vector<LidarPoint>>(lidarPoints);

        // push frame to data buffer
        egoVehicle_.SetDataBuffer(std::move(frame));

        /* CLUSTER LIDAR POINT CLOUD */
        // detect objects on image and crop point cloud with given ROI's

        egoVehicle_.GetDetectedObjects();
        egoVehicle_.UpdateLidarWithROI();

        // visual object detection clusters
        if (visClusters) { VisualizeClusters(egoVehicle_.GetDataBuffer().back()); }
        
        if(imgIndex >= 1) {

            /* MATCH TWO SUBSEQUENT IMAGES (FRAMES) */
            egoVehicle_.GetMatches();

            /* TRACK 3D OBJECT BOUNDING BOXES */
            egoVehicle_.TrackAssociation();

            /* COMPUTE TTC FOR EACH CLUSTER IN CURRENT FRAME */
            egoVehicle_.ComputeTTCLidar(sensorFrameRate, true);

            std::cout << "Image processing finished" << std::endl;

            egoVehicle_.EraseFirstElement();
        }
    }
}


