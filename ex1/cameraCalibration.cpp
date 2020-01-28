#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

void initialiseObjectCorners(Size boardSize, vector<Point3f> & objectCorners){
    for (int i=0; i<boardSize.height; i++) {
        for (int j=0; j<boardSize.width; j++) {
            objectCorners.push_back(Point3f(i, j, 0.0f));
        }
    }    
}

vector<String> getFilesFromFolder(String folder){
    vector<String> files;
    glob(folder, files, false);
    return files;
}

int main(int argc, char** argv ) {
    // camera calibration code based on code from opencv 3 cookbook by Laganiere
    // get folder of images as argument
    String calibrationImageFolder = argv[1];
    String undistortionImageFolder = argv[2];
    /*
    auto calibrationFiles = getFilesFromFolder(calibrationImageFolder);
    // define board size
    Size boardSize(9,6);
    // positions in original space
    vector<vector<Point3f>> objectPoints;
    // positions in image
    vector<vector<Point2f>> imagePoints;
    // set 3D points by assuming z index is zero
    // each square is assumed to be length of one unit
    vector<Point3f> objectCorners;
    initialiseObjectCorners(boardSize, objectCorners);
    Size imageSize;
    cout << "Obtaining points for camera calibration" << endl;
    // loop through images in the folder
    for (size_t i=0; i<calibrationFiles.size(); i++) {
        String imageFile = calibrationFiles[i];
        cout << "Processing image " << imageFile << endl;
        Mat image = imread(samples::findFile(imageFile), IMREAD_GRAYSCALE);
        imageSize = image.size();
        // detect chess board corners
        vector<Point2f> imageCorners;
        bool found = findChessboardCorners(image, boardSize, imageCorners);
        if (found) {
            // use subpixel accuracy in localisation
            int maxIterations = 30;
            float minAccuracy = 0.1;
            // obtain better accuracy with subpixels
            cornerSubPix(image, imageCorners, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, maxIterations, minAccuracy));
            if (imageCorners.size() == boardSize.area()) {
                // save the respective 3D points and corresponding image points
                objectPoints.push_back(objectCorners);
                imagePoints.push_back(imageCorners);
            }
        } else {
            cout << "Cannot use image " << imageFile << endl;
        }
    }
    // do final calibration
    // define variables to catch outputs
    cout << "Running calibration with " << imagePoints.size() << " images" << endl;
    vector<Mat> rvecs, tvecs;
    Mat cameraMatrix;
    Mat distCoeffs;
    calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, 0);
    // undistorting code based on Laganiere and 
    // OpenCV camera calibration tutorial: https://docs.opencv.org/3.4.9/d4/d94/tutorial_camera_calibration.html
    // undistorting stamp images
    cout << "Undistorting images" << endl;
    // the x and y mappings
    Mat mapx, mapy;
    auto undistortionFiles = getFilesFromFolder(undistortionImageFolder);
    for(int i = 0; i < undistortionFiles.size(); i++) {
        Mat undistorted;
        String imageFile = undistortionFiles[i];
        cout << "Undistorting image " << imageFile << endl;
        Mat image = imread(samples::findFile(imageFile), IMREAD_GRAYSCALE);
        // only run initialisation on the first time
        if (i == 0) {
            cout << "Calculating distortion mappings" << endl;
            Size imageSize = image.size();
            auto optimalMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), optimalMatrix, imageSize, CV_16SC2, mapx, mapy);
        }
        remap(image, undistorted, mapx, mapy, INTER_LINEAR);
        imshow("Original image", image);
        waitKey(0);
        imshow("Undistorted image", undistorted);
        waitKey(0);
    }
    */
    // corner detection code based on OpenCV tutorial:
    // https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html
    // corner detections
    RNG rng(12345);
    int maxCorners = 4;
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 700;
    int blockSize = 50, gradientSize = 5;
    bool useHarrisDetector = false;
    double k = 0.01;
    auto undistortionFiles = getFilesFromFolder(undistortionImageFolder);
    for(int i = 0; i < undistortionFiles.size(); i++) {
        String imageFile = undistortionFiles[i];
        cout << "Detecting corners for " << imageFile << endl;
        Mat image = imread(samples::findFile(imageFile), IMREAD_GRAYSCALE);
        Mat copy = image.clone();
        threshold(copy, copy, 0, 255, THRESH_BINARY + THRESH_OTSU);
        goodFeaturesToTrack(copy,
                            corners,
                            maxCorners,
                            qualityLevel,
                            minDistance,
                            Mat(),
                            blockSize,
                            gradientSize,
                            useHarrisDetector,
                            k);
        cout << "Number of corners detected: " << corners.size() << endl;
        int radius = 20;
        for( size_t i = 0; i < corners.size(); i++ )
        {
            circle(copy, corners[i], radius, Scalar(rng.uniform(0,255), rng.uniform(0, 256), rng.uniform(0, 256)), FILLED );
        }
        imshow("Corners detected", copy);
        waitKey(0);
    }
}
