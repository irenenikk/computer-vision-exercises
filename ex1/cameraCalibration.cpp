#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <vector>

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

vector<Point2f> defineTargetCorners(Mat target){
    vector<Point2f> target_points;
    // define corners for the target image
    target_points.push_back(Point2f(0, 0));
    target_points.push_back(Point2f(target.cols, 0));
    target_points.push_back(Point2f(target.cols, target.rows));
    target_points.push_back(Point2f(0, target.rows));
    return target_points;
}

void sortCorners(vector<Point2f>& corners){
    Point2f center(0,0);
    for (int i = 0; i < corners.size(); i++)
        center += corners[i];
    center *= (1. / corners.size());
    // using code from here: https://github.com/stereomatchingkiss/blogCodes2/blob/master/libs/warpImage/warpUtility.hpp
    vector<Point2f> top, bot;
    for (int i = 0; i < corners.size(); i++)
    {
        if (corners[i].y < center.y)
            top.push_back(corners[i]);
        else
            bot.push_back(corners[i]);
    }
    Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
    Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
    Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
    Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];
    corners.clear();
    corners.push_back(tl);
    corners.push_back(tr);
    corners.push_back(br);
    corners.push_back(bl);
}

int main(int argc, char** argv ) {
    // camera calibration code based on code from opencv 3 cookbook by Laganiere
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
    // corner detection code based on OpenCV tutorial:
    // https://docs.opencv.org/3.4/d8/dd8/tutorial_good_features_to_track.html
    // corner detections
    RNG rng(12345);
    int maxCorners = 4;
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 700;
    int blockSize = 70, gradientSize = 5;
    bool useHarrisDetector = false;
    double k = 0.01;
    auto undistortionFiles = getFilesFromFolder(undistortionImageFolder);
    Mat undistorted;
    Mat mapx, mapy;
    for(int i = 0; i < undistortionFiles.size(); i++) {
        String imageFile = undistortionFiles[i];
        cout << "Detecting corners for " << imageFile << endl;
        Mat image = imread(samples::findFile(imageFile), IMREAD_GRAYSCALE);        
        imshow("Original image", image);
        imwrite("processed_images/" + to_string(i) + ".jpg", image);
        waitKey(0);
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
            circle(copy, corners[i], radius, Scalar(100, 200, 100), FILLED );
        }
        imshow("Corners detected", copy);
        imwrite("processed_images/" + to_string(i) + "_corners.jpg", copy);
        waitKey(0);
        /*
        approxPolyDP(Mat(corners), corners, arcLength(Mat(corners), true) * 0.01, true);
        if (corners.size() != 4)
        {
            cout << "Not a quadrilateral object" << endl;
            continue;
        }    
        */    
        sortCorners(corners);
        // perspective correction
        Mat target = Mat::zeros(copy.rows, copy.cols, CV_8UC3);
        auto target_points = defineTargetCorners(target);
        Mat transformMatrix = getPerspectiveTransform(corners, target_points);
        // Apply perspective transformation
        warpPerspective(image, target, transformMatrix, target.size());
        imshow("Processed image", target);
        imwrite("processed_images/" + to_string(i) + "_processed.jpg", target);
        waitKey(0);
        // undistortion
        // undistorting code based on Laganiere and 
        // OpenCV camera calibration tutorial: https://docs.opencv.org/3.4.9/d4/d94/tutorial_camera_calibration.html        
        // only run initialisation on the first time
        if (i == 0) {
            cout << "Calculating distortion mappings" << endl;
            Size imageSize = target.size();
            auto optimalMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
            initUndistortRectifyMap(cameraMatrix, distCoeffs, Mat(), optimalMatrix, imageSize, CV_16SC2, mapx, mapy);
        }
        remap(target, undistorted, mapx, mapy, INTER_LINEAR);
        imshow("Undistorted image", undistorted);
        imwrite("processed_images/" + to_string(i) + "_undistorted.jpg", undistorted);
        waitKey(0);
        destroyAllWindows();
    }
}
