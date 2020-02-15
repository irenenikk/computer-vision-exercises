#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <tuple>
#include "opencv2/xfeatures2d.hpp"
#include <map>

#include "standalone_image.h"
#include "gist.h"

using namespace std;
using namespace cv;
using namespace cls;

const GISTParams DEFAULT_PARAMS {true, 32, 32, 4, 3, {8, 8, 4}};

vector<String> getFilesFromFolder(String folder){
    vector<String> files;
    glob(folder, files, false);
    return files;
}

vector<tuple<Mat, String>> readImagesFromFolders(vector<String> folderNames){
    vector<tuple<Mat, String>> images;
    for (size_t i=0; i<folderNames.size(); i++) {
        String folderName = folderNames[i];
        auto files = getFilesFromFolder(folderName);
        for (auto file: files) {
            Mat image = imread(samples::findFile(file));
            images.push_back(make_tuple(image, folderName));            
        }
    }
    return images;
}

int getLabelCategorical(String label, map<String, int> &labelToCat, map<int, String> &catToLabel, int &runningLabelCat){
    if (labelToCat.count(label) > 0) {
        return labelToCat[label];
    }
    runningLabelCat++;
    labelToCat[label] = runningLabelCat;
    catToLabel[runningLabelCat] = label;
    return runningLabelCat;
}

void getData(vector<Mat> &images, vector<int> &labels, map<String, 
                int> &labelToCat, map<int, String> &catToLabel, int &runningLabelCat, String folder){
    auto folderNames = getFilesFromFolder(folder);
    auto dataTuples = readImagesFromFolders(folderNames);
    for (tuple<Mat, String> imageTuple: dataTuples) {
        Mat image = get<0>(imageTuple);
        String label = get<1>(imageTuple);
        images.push_back(image);
        int labelCategory = getLabelCategorical(label, labelToCat, catToLabel, runningLabelCat);
        labels.push_back(labelCategory);
    }
}

void getFeatures(vector<Mat> trainingImages, vector<vector<KeyPoint>> &allSiftKeypoints, 
                vector<Mat> &allSiftDescriptors, vector<vector<float>> &allGISTFeatures){
    GIST gist_ext(DEFAULT_PARAMS);
    // calculate the sift features
    for (size_t i=0; i<trainingImages.size(); i++) {
        Mat image = trainingImages[i];
        Mat grayImage;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
        vector<KeyPoint> keypoints;    
        f2d->detect(image, keypoints);
        allSiftKeypoints.push_back(keypoints);
        // draw the found keypoints
        drawKeypoints(grayImage, keypoints, image);
        //imshow("Image with keypoints", image);
        //waitKey(0);
        Mat descriptors;
        f2d->compute(image, keypoints, descriptors);
        cout << "descriptor shape " << descriptors.size() << endl;
        allSiftDescriptors.push_back(descriptors);
        vector<float> GISTFeatures;
        gist_ext.extract(image, GISTFeatures);
        allGISTFeatures.push_back(GISTFeatures);
    }

}

int main(int argc, char* argv[]){
    String trainingSetFolder = argv[1];
    String testSetFolder = argv[2];
    int runningLabelCat = 0;
    map<String, int> labelToCat;
    map<int, String> catToLabel;
    vector<Mat> trainingImages;
    vector<int> trainingLabels;
    vector<Mat> testImages;
    vector<int> testLabels;
    getData(trainingImages, trainingLabels, labelToCat, catToLabel, runningLabelCat, trainingSetFolder);
    getData(testImages, testLabels, labelToCat, catToLabel, runningLabelCat, trainingSetFolder);
    vector<vector<KeyPoint>> allSiftKeypoints;
    vector<Mat> allSiftDescriptors;
    vector<vector<float>> allGISTFeatures;
    getFeatures(trainingImages, allSiftKeypoints, allSiftDescriptors, allGISTFeatures);
    
}

/*
vector<float> result;
GIST gist_ext(DEFAULT_PARAMS);
gist_ext.extract(src, result);
for (const auto & val : result) {
    cout << fixed << setprecision(4) << val << " ";
}
cout << endl;
*/