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
    // get folder of images as argument
    String calibrationImageFolder = argv[1];
    String undistortionImageFolder = argv[2];
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
    // do undistorting
    cout << "Undistorting images" << endl;
    // the x and y mappings
    Mat mapx, mapy;
    auto undistortionFiles = getFilesFromFolder(undistortionImageFolder);
    for(int i = 0; i < undistortionFiles.size(); i++) {
        Mat undistorted;
        String imageFile = undistortionFiles[i];
        cout << "Undistorting image " << imageFile << endl;
        Mat image = imread(samples::findFile(imageFile), IMREAD_COLOR);
        // only run initialisation on the first time
        if (i == 0) {
            cout << "Calculating distortion mappings" << endl;
            Size imageSize = image.size();
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), Mat(), imageSize, CV_32FC1, mapx, mapy);
        }
        remap(image, undistorted, mapx, mapy, INTER_LINEAR);
        imshow("Undistorted image", undistorted);
        waitKey(1000);
    }
}
