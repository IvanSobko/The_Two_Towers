#include <iostream>
#include <map>
#include <filesystem>

#include "DescriptorProcessor.h"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


int main() {

    DescriptorProcessor processor("../img/object.jpeg", "../img/");

    size_t size = processor.getSize();
    printf("Total photos to process: %zu.\n", size);
    if (!size) {
        printf("Error: couldn't load images\n");
        return 1;
    }
    processor.setHaussian(400);
    processor.process();
//    processor.displayMatches();

    std::string metricsFilename = "../results_final.csv";
    processor.saveMetricsToFile(metricsFilename);
    return 0;
}