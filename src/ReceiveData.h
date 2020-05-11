#ifndef CAMERA_FUSION_RECEIVEDATA_H
#define CAMERA_FUSION_RECEIVEDATA_H

#pragma clang diagnostic push
#pragma ide diagnostic ignored "performance-inefficient-string-concatenation"

#include <memory>

#include "EgoVehicle.h"


class ReceiveData {

public:
    ReceiveData();
    ~ReceiveData();
    void LoadParameters();
    void Update();
    void VisualizeClusters(const DataFrame& frame);

private:
    EgoVehicle egoVehicle_;                         // ego vehicle (base object)
    const int imgStartIndex{ 0 };                   // first file index to load
    const int imgEndIndex{ 18 };                    // last file index to load
    const int imgStepWidth{ 1 };
    const int imgFillWidth{ 4 };                    // no. of digits which make up the file index (e.g. img-0001.png)
    const float minZ{ -1.5 }, maxZ{ -0.9 };
    const float minX{ 2.0 }, maxX{ 20.0 };
    const float maxY{ 2.0 }, minR{ 0.1 };           // focus on ego lane
    const float shrinkFactor{ 0.10 };
    const double sensorFrameRate{ 10.0 / imgStepWidth }; // frames per second for Lidar and camera

};



#endif //CAMERA_FUSION_RECEIVEDATA_H
