#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
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

void saveVectorofVectorsToFile(String filename, String variable, vector<vector<float>> content){
    FileStorage fs(filename, FileStorage::WRITE);
    fs << variable << content;
    fs.release();
}

void saveVectorofMatricesToFile(String filename, String variable, vector<Mat> content){
    FileStorage fs(filename, FileStorage::WRITE);
    fs << variable << content;
    fs.release();
}

void saveMatrixToFile(String filename, String variable, Mat content){
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

vector<Mat> loadVectorOfMatrices(String filename, String variable){
    FileStorage fs(filename, FileStorage::READ);
    vector<Mat> matrix; 
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

void drawHistogram(Mat image, Mat b_hist, Mat g_hist, Mat r_hist, int histSize){
    // using code from this tutorial: https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double) hist_w/histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar( 0,0,0));
    for( int i = 1; i < histSize; i++ ){
        line(histImage, Point(bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)*hist_h)),
              Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)*hist_h)),
              Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)*hist_h)),
              Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)*hist_h)),
              Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)*hist_h)),
              Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)*hist_h)),
              Scalar(0, 0, 255), 2, 8, 0);
    }
    imshow("Image", image);
    imshow("Histogram", histImage);
    waitKey();
}

Mat getHSVColorHistogram(Mat image){
    // partly based on this snippet: https://stackoverflow.com/questions/20028933/calculate-hsv-histogram-of-a-coloured-image-is-it-different-from-h-s-histogram
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    int h_bins = 50; 
    int s_bins = 32;
    int v_bins = 10;
    int histSize[] = { h_bins, s_bins, v_bins };
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    float v_ranges[] = {0, 256};
    const float* ranges[] = { h_ranges, s_ranges, v_ranges };
    int channels[] = { 0, 1, 2};
    Mat histogram;
    calcHist( &hsv, 1, channels, Mat(), histogram, 3, histSize, ranges, true, false);
    normalize(histogram, histogram, 0, 1, NORM_MINMAX, -1, Mat() );
    histogram = histogram.reshape(0,1);
    return histogram;
}

Mat getRBGColorHistogram(Mat image){
    // using code from this tutorial: https://docs.opencv.org/3.4/d8/dbc/tutorial_histogram_calculation.html
    int histSize = 256;
    vector<Mat> bgr_planes;
    split(image, bgr_planes);
    float range[] = { 0, 256 };
    const float* histRange = { range };
    Mat b_hist, g_hist, r_hist;
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, true, false);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, true, false);
    normalize(b_hist, b_hist, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, 1, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, 1, NORM_MINMAX, -1, Mat());
    //drawHistogram(image, b_hist, g_hist, r_hist, histSize);
    Mat concatTemp;
    Mat concatHist;
    vconcat(b_hist, g_hist, concatTemp);
    vconcat(concatTemp, r_hist, concatHist);
    return concatHist.t();
}

void getData(vector<Mat> &images, vector<int> &labels, map<String, int> &labelToCat, 
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

void getRawFeatures(vector<Mat> allTrainingImages, vector<Mat> &allSiftDescriptors, 
                    vector<vector<float>> &allGISTFeatures, Mat &colorHistograms){
    GIST gist_ext(DEFAULT_PARAMS);
    cout << "Getting raw features from images" << endl;
    // calculate the sift features
    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
    String SIFTFile = "SIFTDescriptors.yml";
    String GISTFile = "GISTFeatures.yml";
    String colorHistogramFile = "colorHistogram.yml";
    String siftVariableName = "allSiftDescriptors";
    String gistVariableName = "gistFeatures";
    String colorHistogramVariable = "colorHistogram";
    ifstream siftFeatureFile(SIFTFile);
    ifstream gistFeatureFile(GISTFile);
    ifstream colorFeatureFile(colorHistogramFile);
    if (!siftFeatureFile || !gistFeatureFile || !colorFeatureFile) {
        cout << "Extracting features for training data" << endl;
        for (size_t i=0; i<allTrainingImages.size(); i++) {
            Mat image = allTrainingImages[i];
            Mat grayImage;
            cvtColor(image, grayImage, COLOR_BGR2GRAY);
            // SIFT
            vector<KeyPoint> keypoints;
            f2d->detect(grayImage, keypoints);
            // draw the found keypoints
            //Mat imagewithKeypoints;
            //drawKeypoints(image, keypoints, imagewithKeypoints);
            //imshow("Image with keypoints", imagewithKeypoints);
            //waitKey();
            Mat descriptors;
            f2d->compute(grayImage, keypoints, descriptors);
            allSiftDescriptors.push_back(descriptors);
            // GIST
            vector<float> GISTFeatures;
            gist_ext.extract(image, GISTFeatures);
            allGISTFeatures.push_back(GISTFeatures);
            // change to getRBGColorHistogram to use RBG colors
            Mat concatHist = getHSVColorHistogram(image);
            colorHistograms.push_back(concatHist);
        }
        saveVectorofMatricesToFile(SIFTFile, siftVariableName, allSiftDescriptors);
        saveMatrixToFile(colorHistogramFile, colorHistogramVariable, colorHistograms);
        saveVectorofVectorsToFile(GISTFile, gistVariableName, allGISTFeatures);
    } else {
        allSiftDescriptors = loadVectorOfMatrices(SIFTFile, siftVariableName);
        cout << "loaded sift descriptors" << endl;
        allGISTFeatures = loadVectorOfVectors(GISTFile, gistVariableName);
        cout << "loaded gist descriptors" << endl;
        colorHistograms = loadMatrix(colorHistogramFile, colorHistogramVariable);
        cout << "loaded color descriptors" << endl;
    }
}

void getBowSiftFeatures(vector<Mat> images, Ptr<BOWImgDescriptorExtractor> &bowDE, Mat &BOWDescriptors, String BOWDescriptorFileName){
    String BOWDescriptorVariable = "BOWDescriptors";
    ifstream BOWDescriptorFile(BOWDescriptorFileName);
    if (!BOWDescriptorFile) {
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
        saveMatrixToFile(BOWDescriptorFileName, BOWDescriptorVariable, BOWDescriptors);
    } else {
        cout << "Loading SIFT BOW descriptors from file " << BOWDescriptorFileName << endl;
        BOWDescriptors = loadMatrix(BOWDescriptorFileName, BOWDescriptorVariable);
    }
}

Ptr<BOWImgDescriptorExtractor> getBowDE(Mat trainingSiftDescriptors){
    String dictionaryName = "dictionary";
    String dictionaryFileName = dictionaryName + ".yml";
    ifstream dictionaryFile(dictionaryFileName);
    Mat dictionary;
    if (!dictionaryFile){
        cout << "Training a BOW dictionary" << endl;
        int dictionarySize = 1200;
        TermCriteria termCrit = TermCriteria();
        BOWKMeansTrainer bowTrainer(dictionarySize, termCrit, 1, KMEANS_PP_CENTERS);
        dictionary = bowTrainer.cluster(trainingSiftDescriptors);
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

void getTestFeatures(vector<Mat> testImages, vector<vector<float>> &testGISTFeatures, vector<Mat> &colorHistograms){
    GIST gist_ext(DEFAULT_PARAMS);
    for (size_t i=0; i<testImages.size(); i++) {
        Mat image = testImages[i];
        vector<float> GISTFeatures;
        gist_ext.extract(image, GISTFeatures);
        testGISTFeatures.push_back(GISTFeatures);
        // change to getRBGColorHistogram to use RBG colors
        Mat colorHistogram = getHSVColorHistogram(image);
        colorHistograms.push_back(colorHistogram);
    }
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

Ptr<SVM> createSVM(Mat trainingData, Mat trainingLabels, SVM::KernelTypes kernel){
    Ptr<SVM> svmGIST = SVM::create();
    svmGIST->setType(SVM::C_SVC);
    svmGIST->setKernel(kernel);
    svmGIST->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svmGIST->train(trainingData, ROW_SAMPLE, trainingLabels);
    return svmGIST;
}

Mat convertVectorToMat(vector<float> vector){
    Mat m = Mat(1, vector.size(), CV_32F);
    memcpy(m.data, vector.data(), vector.size()*sizeof(float));
    return m;
}

int main(int argc, char* argv[]){
    String trainingSetFolder = argv[1];
    String testSetFolder = argv[2];
    int runningLabelCat = 0;
    map<String, int> labelToCat;
    map<int, String> catToLabel;
    vector<Mat> allTrainingImages;
    vector<int> allTrainingLabels;
    vector<Mat> testImages;
    vector<int> testLabels;
    getData(allTrainingImages, allTrainingLabels, labelToCat, catToLabel, runningLabelCat, trainingSetFolder);
    getData(testImages, testLabels, labelToCat, catToLabel, runningLabelCat, testSetFolder);
    // get SIFT features for training data
    vector<Mat> allSiftDescriptors;
    vector<vector<float>> allGISTFeatures;
    Mat allColorHistograms;
    getRawFeatures(allTrainingImages, allSiftDescriptors, allGISTFeatures, allColorHistograms);
    // shuffle the test dataset before splitting into dev and test
    // the shuffled order will always be the same because the seed is always the same
    cout << "Splitting indices into training, development and test sets" << endl;
    vector<int> indices(allTrainingImages.size());
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    vector<int> developmentIndices(indices.begin(), indices.begin() + 100);
    vector<int> trainingIndices(indices.begin() + 100, indices.end());

    // separate training set
    Mat trainingSIFTDescriptors;
    Mat trainingLabels;
    vector<vector<float>> trainingGISTFeatures;
    Mat trainingColorHistograms;
    vector<Mat> trainingImages;
    cout << "Cretaing training set" << endl;
    for (size_t i=0; i<trainingIndices.size(); i++) {
        auto index = trainingIndices[i];
        trainingImages.push_back(allTrainingImages[index]);
        trainingSIFTDescriptors.push_back(allSiftDescriptors[i]);
        trainingLabels.push_back(allTrainingLabels[index]);
        trainingGISTFeatures.push_back(allGISTFeatures[index]);
        trainingColorHistograms.push_back(allColorHistograms.row(index));
    }
    cout << "Training with a training set of size " << trainingIndices.size() << endl;

    // Run KNN in order to bin descriptors
    auto bowDE = getBowDE(trainingSIFTDescriptors);
    Mat trainingBowDescriptors;
    String trainingBOWDescriptorFileName = "BOWDescriptors.yml";
    getBowSiftFeatures(trainingImages, bowDE, trainingBowDescriptors, trainingBOWDescriptorFileName);

    // Train the SVM
    cout << "Training an SVM with SIFT and GIST features" << endl;
    auto svmSIFT = createSVM(trainingBowDescriptors, trainingLabels, SVM::LINEAR);
    auto trainingGISTMat = vectorToMat(trainingGISTFeatures);
    auto svmGIST = createSVM(trainingGISTMat, trainingLabels, SVM::LINEAR);
    auto svmColor = createSVM(trainingColorHistograms, trainingLabels, SVM::LINEAR);

    // create development set of sift features
    Mat developmentSIFTDescriptors;
    vector<Mat> developmentImages;
    cout << "Creating development set of SIFT features" << endl;
    for (size_t i=0; i<developmentIndices.size(); i++) {
        auto index = developmentIndices[i];
        developmentImages.push_back(allTrainingImages[index]);
    }
    Mat developmentBowDescriptors;
    String devBOWDescriptorFileName = "devBOWDescriptors.yml";
    getBowSiftFeatures(developmentImages, bowDE, developmentBowDescriptors, devBOWDescriptorFileName);

    cout << "Testing the SVM with using the development set" << endl;
    // TODO: train an svm using color histograms
    int siftCorrect = 0;
    int gistCorrect = 0;
    int colorCorrect = 0;
    for (size_t i=0; i<developmentIndices.size(); i++) {
        auto index = developmentIndices[i];

        auto siftDescriptors = developmentBowDescriptors.row(i);
        auto siftPredictedLabel = svmSIFT->predict(siftDescriptors);

        auto gistDescriptors = allGISTFeatures[index];
        auto gistPredictedLabel = svmGIST->predict(gistDescriptors);

        auto colorFeatures = allColorHistograms.row(i);
        auto colorPredictedLabel = svmColor->predict(colorFeatures);

        auto trueLabel = allTrainingLabels[index];
        if (siftPredictedLabel == trueLabel){
            siftCorrect++;
        }
        if (gistPredictedLabel == trueLabel){
            gistCorrect++;
        }
        if (colorPredictedLabel == trueLabel){
            colorCorrect++;
        }
    }
    cout << "Accuracy in the dev set using SIFT is " << ((float)siftCorrect/(float)developmentIndices.size()) << endl;
    cout << "Accuracy in the dev set using GIST is " << ((float)gistCorrect/(float)developmentIndices.size()) << endl;
    cout << "Accuracy in the dev set using color histograms is " << ((float)colorCorrect/(float)developmentIndices.size()) << endl;
    cout << "Testing the SVM with using the test set" << endl;
    Mat testBowDescriptors;
    String testBOWDescriptorFileName = "testBOWDescriptors.yml";
    getBowSiftFeatures(testImages, bowDE, testBowDescriptors, testBOWDescriptorFileName);
    vector<vector<float>> testGISTFeatures;
    vector<Mat> testColorHistograms;
    getTestFeatures(testImages, testGISTFeatures, testColorHistograms);
    auto testGISTMat = vectorToMat(trainingGISTFeatures);
    int linearSiftCorrect = 0;
    int linearGistCorrect = 0;
    int linearColorCorrect = 0;
    int rbfSiftCorrect = 0;
    int rbfGistCorrect = 0;
    int rbfColorCorrect = 0;
    int rbfGistSiftCorrect = 0;
    int rbfGistColorCorrect = 0;
    int rbfSiftColorCorrect = 0;
    // create non-linear svms
    auto rbfSvmSIFT = createSVM(trainingBowDescriptors, trainingLabels, SVM::RBF);
    auto rbfSvmGIST = createSVM(trainingGISTMat, trainingLabels, SVM::RBF);
    auto rbfSvmColor = createSVM(trainingColorHistograms, trainingLabels, SVM::RBF);
    // create combined feature sets and respective svms
    Mat trainingSiftColor;
    hconcat(trainingBowDescriptors, trainingColorHistograms, trainingSiftColor);
    Mat trainingGistColor;
    hconcat(trainingGISTMat, trainingColorHistograms, trainingGistColor);
    Mat trainingGistSift;
    hconcat(trainingGISTMat, trainingBowDescriptors, trainingGistSift);
    auto rbfSvmSIFTColor = createSVM(trainingSiftColor, trainingLabels, SVM::RBF);
    auto rbfSvmGistColor = createSVM(trainingGistColor, trainingLabels, SVM::RBF);
    auto rbfSvmGistSift = createSVM(trainingGistSift, trainingLabels, SVM::RBF);
    for (size_t i=0; i<testImages.size(); i++) {
        auto siftDescriptors = testBowDescriptors.row(i);
        auto linearSiftPredictedLabel = svmSIFT->predict(siftDescriptors);
        auto rbfSiftPredictedLabel = rbfSvmSIFT->predict(siftDescriptors);
        Mat siftColor;
        hconcat(siftDescriptors, testColorHistograms[i], siftColor);
        auto rbfSiftColorPredictedLabel = rbfSvmSIFTColor->predict(siftColor);

        auto gistDescriptors = testGISTFeatures[i];
        auto linearGistPredictedLabel = svmGIST->predict(gistDescriptors);
        auto rbfGistPredictedLabel = rbfSvmGIST->predict(gistDescriptors);
        Mat gistColor;
        hconcat(convertVectorToMat(gistDescriptors), testColorHistograms[i], gistColor);
        auto rbfGistColorPredictedLabel = rbfSvmGistColor->predict(gistColor);

        auto colorFeatures = testColorHistograms[i];
        auto linearColorPredictedLabel = svmColor->predict(colorFeatures);
        auto rbfColorPredictedLabel = rbfSvmColor->predict(colorFeatures);

        Mat siftgist;
        hconcat(convertVectorToMat(gistDescriptors), siftDescriptors, siftgist);
        auto rbfGistSiftPredictedLabel = rbfSvmGistSift->predict(siftgist);

        auto trueLabel = testLabels[i];
        if (linearSiftPredictedLabel == trueLabel){
            linearSiftCorrect++;
        }
        if (rbfSiftPredictedLabel == trueLabel){
            rbfSiftCorrect++;
        }
        if (linearGistPredictedLabel == trueLabel){
            linearGistCorrect++;
        }
        if (rbfGistPredictedLabel == trueLabel){
            rbfGistCorrect++;
        }
        if (linearColorPredictedLabel == trueLabel){
            linearColorCorrect++;
        }
        if (rbfColorPredictedLabel == trueLabel){
            rbfColorCorrect++;
        }
        if (rbfSiftColorPredictedLabel == trueLabel){
            rbfSiftColorCorrect++;
        }
        if (rbfGistColorPredictedLabel == trueLabel){
            rbfGistColorCorrect++;
        }
        if (rbfGistSiftPredictedLabel == trueLabel){
            rbfGistSiftCorrect++;
        }        
    }
    cout << "Linear SVM accuracy in the test set using SIFT is " << ((float)linearSiftCorrect/(float)testImages.size()) << endl;
    cout << "Linear SVM accuracy in the test set using GIST is " << ((float)linearGistCorrect/(float)testImages.size()) << endl;
    cout << "Linear SVM accuracy in the test set using color histograms is " << ((float)linearColorCorrect/(float)testImages.size()) << endl;
    cout << "RBF SVM accuracy in the test set using SIFT is " << ((float)rbfSiftCorrect/(float)testImages.size()) << endl;
    cout << "RBF SVM accuracy in the test set using GIST is " << ((float)rbfGistCorrect/(float)testImages.size()) << endl;
    cout << "RBF SVM accuracy in the test set using color histograms is " << ((float)rbfColorCorrect/(float)testImages.size()) << endl;
    cout << "RBF SVM accuracy in the test set using SIFT + color histograms is " << ((float)rbfSiftColorCorrect/(float)testImages.size()) << endl;
    cout << "RBF SVM accuracy in the test set using GIST + color histograms is " << ((float)rbfGistColorCorrect/(float)testImages.size()) << endl;
    cout << "RBF SVM accuracy in the test set using SIFT + GIST is " << ((float)rbfGistSiftCorrect/(float)testImages.size()) << endl;
 }
