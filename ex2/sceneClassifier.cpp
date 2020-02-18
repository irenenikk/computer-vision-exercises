#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <tuple>
#include "opencv2/xfeatures2d.hpp"
#include <map>
#include <dirent.h> 
#include <stdio.h> 
#include <cstring>
#include <algorithm>
#include <numeric>
#include <fstream>

#include "standalone_image.h"
#include "gist.h"

using namespace std;
using namespace cv;
using namespace cls;
using namespace cv::ml;

const GISTParams DEFAULT_PARAMS {true, 32, 32, 4, 3, {8, 8, 4}};

vector<String> getFilesFromFolder(String folder){
    vector<String> files;
    struct dirent *de;
    DIR *dr = opendir(folder.c_str());
    while ((de = readdir(dr)) != NULL) {
        String file = de->d_name;
        if (file.rfind(".", 0) == 0){
            continue;
        }
        files.push_back(file);
    }
    closedir(dr);
    return files;
}

vector<tuple<Mat, String>> readImagesFromFolders(String rootPath, vector<String> folderNames){
    vector<tuple<Mat, String>> images;
    for (size_t i=0; i<folderNames.size(); i++) {
        String folderName = folderNames[i];
        auto files = getFilesFromFolder(rootPath + "/" + folderName);
        for (auto file: files) {
            String filePath = rootPath + "/" + folderName + "/" + file;
            Mat image = imread(samples::findFile(filePath));
            images.push_back(make_tuple(image, folderName));            
        }
    }
    return images;
}

int getLabelCategorical(String label, map<String, int> &labelToCat, map<int, String> &catToLabel, int &runningLabelCat){
    if (labelToCat.count(label) > 0) {
        //cout << "The categorical for " << label << " is " << labelToCat[label] << endl;
        return labelToCat[label];
    }
    runningLabelCat++;
    labelToCat[label] = runningLabelCat;
    catToLabel[runningLabelCat] = label;
    return runningLabelCat;
}

void getData(vector<Mat> &images, Mat &labels, map<String, int> &labelToCat, 
                map<int, String> &catToLabel, int &runningLabelCat, String folder){
    auto folderNames = getFilesFromFolder(folder);
    auto dataTuples = readImagesFromFolders(folder, folderNames);
    for (tuple<Mat, String> imageTuple: dataTuples) {
        Mat image = get<0>(imageTuple);
        String label = get<1>(imageTuple);
        images.push_back(image);
        int labelCategory = getLabelCategorical(label, labelToCat, catToLabel, runningLabelCat);
        labels.push_back(labelCategory);
    }
}

void saveMatrixToFile(String filename, String variable, Mat content){
    FileStorage fs(filename, FileStorage::WRITE);
    fs << variable << content;
    fs.release();
}

void saveVectorofVectorsToFile(String filename, String variable, vector<vector<float>> content){
    FileStorage fs(filename, FileStorage::WRITE);
    fs << variable << content;
    fs.release();
}

Mat loadMatrix(String filename, String variable){
    FileStorage fs(filename, FileStorage::READ);
    Mat matrix; 
    fs[variable] >> matrix;
    fs.release();
    return matrix;
}

vector<vector<float>> loadVectorOfVectors(String filename, String variable){
    FileStorage fs(filename, FileStorage::READ);
    vector<vector<float>> vector; 
    fs[variable] >> vector;
    fs.release();
    return vector;
}

void getRawFeatures(vector<int> trainingIndices, vector<Mat> trainingImages, 
                        Mat &allSiftDescriptors, vector<vector<float>> &allGISTFeatures){
    GIST gist_ext(DEFAULT_PARAMS);
    cout << "Getting raw features from images" << endl;
    // calculate the sift features
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    String SIFTFile = "SIFTDescriptors.yml";
    String GISTFile = "GISTFeatures.yml";
    String siftVariableName = "allSiftDescriptors";
    String gistVariableName = "gistFeatures";
    ifstream dictionaryFile(SIFTFile);
    ifstream gistFeatureFile(GISTFile);
    if (!dictionaryFile || !gistFeatureFile){
        cout << "Extracting features for training data" << endl;
        for (size_t i=0; i<trainingIndices.size(); i++) {
            Mat image = trainingImages[trainingIndices[i]];
            Mat grayImage;
            cvtColor(image, grayImage, COLOR_BGR2GRAY);
            vector<KeyPoint> keypoints;
            f2d->detect(grayImage, keypoints);
            // draw the found keypoints
            //drawKeypoints(grayImage, keypoints, image);
            //imshow("Image with keypoints", image);
            //waitKey(0);
            Mat descriptors;
            f2d->compute(grayImage, keypoints, descriptors);
            allSiftDescriptors.push_back(descriptors);
            vector<float> GISTFeatures;
            gist_ext.extract(image, GISTFeatures);
            allGISTFeatures.push_back(GISTFeatures);
            cout << "SIFT features " << allSiftDescriptors.size() << endl;
            cout << "GIST features " << allGISTFeatures.size() << endl;
        }
        saveMatrixToFile(SIFTFile, siftVariableName, allSiftDescriptors);
        saveVectorofVectorsToFile(GISTFile, gistVariableName, allGISTFeatures);
    } else {
        Mat allSiftDescriptors = loadMatrix(SIFTFile, siftVariableName);
        vector<vector<float>> allGISTFeatures = loadVectorOfVectors(GISTFile, gistVariableName);
    }
}

void getBowSiftFeatures(vector<Mat> images, Ptr<BOWImgDescriptorExtractor> &bowDE, 
                            Mat allSiftDescriptors, Mat &BOWDescriptors){
    cout << "Starting KNN for SIFT features" << endl;
    Ptr<Feature2D> detector = xfeatures2d::SIFT::create();
    for (size_t i=0; i<images.size(); i++) {
        Mat image = images[i];
        Mat grayImage;
        cvtColor(image, grayImage, COLOR_BGR2GRAY);
        vector<KeyPoint> keypoints;        
        detector->detect(grayImage, keypoints);
        Mat bowDescriptor;
        bowDE->compute(grayImage, keypoints, bowDescriptor);
        BOWDescriptors.push_back(bowDescriptor);
    }
}

Ptr<BOWImgDescriptorExtractor> getBowDE(Mat allSiftDescriptors){
    String dictionaryName = "dictionary";
    String dictionaryFileName = dictionaryName + ".yml";
    ifstream dictionaryFile(dictionaryFileName);
    Mat dictionary;
    if (!dictionaryFile){
        cout << "Training a BOW dictionary" << endl;
        int dictionarySize = 8000;
        TermCriteria termCrit = TermCriteria();
        BOWKMeansTrainer bowTrainer(dictionarySize, termCrit, 1, KMEANS_PP_CENTERS);
        dictionary = bowTrainer.cluster(allSiftDescriptors);
        saveMatrixToFile(dictionaryFileName, dictionaryName, dictionary);
        cout << "Saved dictionary to " << dictionaryFileName << endl;
    } else {
        cout << "Loading pretrained dictionary from " << dictionaryFileName << endl;
        dictionary = loadMatrix(dictionaryFileName, dictionaryName);
    }
    Ptr<DescriptorMatcher> matcher = FlannBasedMatcher::create();
    Ptr<DescriptorExtractor> extractor = xfeatures2d::SiftDescriptorExtractor::create();
    Ptr<BOWImgDescriptorExtractor> bowDE = new BOWImgDescriptorExtractor(extractor, matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE->setVocabulary(dictionary);
    return bowDE;
}

vector<vector<float>> getTestGISTFeatures(vector<Mat> testImages){
    GIST gist_ext(DEFAULT_PARAMS);
    vector<vector<float>> testGISTFeatures;
    for (size_t i=0; i<testImages.size(); i++) {
        Mat image = testImages[i];
        vector<float> GISTFeatures;
        gist_ext.extract(image, GISTFeatures);
        testGISTFeatures.push_back(GISTFeatures);
    }
    return testGISTFeatures;
}

Mat vectorToMat(vector<vector<float>> trainingGISTFeatures){
    Mat trainingGISTMat(trainingGISTFeatures.size(), trainingGISTFeatures.at(0).size(), CV_32F);
    for(int i=0; i<trainingGISTMat.rows; ++i) {
        for(int j=0; j<trainingGISTMat.cols; ++j) {
            trainingGISTMat.at<float>(i, j) = trainingGISTFeatures.at(i).at(j); 
        }
    }
    return trainingGISTMat;
}

int main(int argc, char* argv[]){
    String trainingSetFolder = argv[1];
    String testSetFolder = argv[2];
    int runningLabelCat = 0;
    map<String, int> labelToCat;
    map<int, String> catToLabel;
    vector<Mat> trainingImages;
    Mat trainingLabels;
    vector<Mat> testImages;
    Mat testLabels;
    getData(trainingImages, trainingLabels, labelToCat, catToLabel, runningLabelCat, trainingSetFolder);
    //getData(testImages, testLabels, labelToCat, catToLabel, runningLabelCat, testSetFolder);
    // shuffle the test dataset before splitting into dev and test
    vector<int> indices(trainingImages.size());
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    vector<int> developmentIndices(indices.begin(), indices.begin() + 100);
    vector<int> trainingIndices(indices.begin() + 100, indices.end());
    // get SIFT features for training data
    Mat allSiftDescriptors;
    vector<vector<float>> trainingGISTFeatures;
    getRawFeatures(trainingIndices, trainingImages, allSiftDescriptors, trainingGISTFeatures);
    // Run KNN in order to bin descriptors
    auto bowDE = getBowDE(allSiftDescriptors);
    Mat trainingBowDescriptors;
    Mat testBowDescriptors;
    //getBowSiftFeatures(trainingImages, bowDE, allSiftDescriptors, trainingBowDescriptors);
    //getBowSiftFeatures(testImages, bowDE, allSiftDescriptors, testBowDescriptors);
    //auto testGISTFeatures = getTestGISTFeatures(testImages);
    // Train the SVM
    cout << "Training an SVM with SIFT and GIST features" << endl;
    /*
    Ptr<SVM> svmSIFT = SVM::create();
    svmSIFT->setType(SVM::C_SVC);
    svmSIFT->setKernel(SVM::LINEAR);
    svmSIFT->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svmSIFT->train(trainingBowDescriptors, ROW_SAMPLE, trainingLabels);
    */
    auto trainingGISTMat = vectorToMat(trainingGISTFeatures);
    cout << trainingGISTMat.size() << endl;
    cout << trainingLabels.size() << endl;
    Ptr<SVM> svmGIST = SVM::create();
    svmGIST->setType(SVM::C_SVC);
    svmGIST->setKernel(SVM::LINEAR);
    svmGIST->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svmGIST->train(trainingGISTMat, ROW_SAMPLE, trainingLabels);
    cout << "Testing the SVM with SIFT features" << endl;
    int siftCorrect = 0;
    int gistCorrect = 0;
    for (size_t i=0; i<developmentIndices.size(); i++) {
        //auto siftDescriptors = trainingBowDescriptors.row(developmentIndices[i]);
        //auto siftPredictedLabel = svmSIFT->predict(siftDescriptors);
        auto gistDescriptors = trainingGISTMat.row(developmentIndices[i]);
        auto gistPredictedLabel = svmGIST->predict(gistDescriptors);
        auto trueLabel = trainingLabels.row(developmentIndices[i]).ptr<int>(0)[0];
        cout << "true label " << trueLabel << endl;
        cout << "predicted label " << gistPredictedLabel << endl;
        /*
        if (siftPredictedLabel == trueLabel){
            siftCorrect++;
        }
        */
        if (gistPredictedLabel == trueLabel){
            gistCorrect++;
        }
    }
    cout << "Accuracy using SIFT is " << ((float)siftCorrect/(float)testImages.size()) << endl;
    cout << "Accuracy using GIST is " << ((float)gistCorrect/(float)testImages.size()) << endl;
 }
