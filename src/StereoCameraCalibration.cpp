#include "StereoCameraCalibration.h"

using namespace cv;
using namespace ofxCv;

void SingleCameraCalibration::setup(string defaultFilePath, float chessBoardSize) {
    ofxCv::Calibration::reset();
    bRequestCalibrate = false;
    calibrationFrameCount = 0;
    filePath = defaultFilePath;
    setPatternSize(10, 7);
    setSquareSize(chessBoardSize);
    squareSize = chessBoardSize;
}

void SingleCameraCalibration::load() {
    if (ofFile(filePath).exists()) {
        ofxCv::Calibration::load(filePath);
    }
}

void SingleCameraCalibration::save() {
    if (size() > 3) {
        ofxCv::Calibration::save(filePath);
    }
}

// for synchronize issue
void SingleCameraCalibration::requestCalibrateNextFrame() {
    bRequestCalibrate = true;
}

void SingleCameraCalibration::update(ofPixels& pixels) {
    inputImage = pixels;
    inputImage.update();
    if (!checkerBoardImage.isAllocated()) {
        imitate(checkerBoardImage, inputImage);
        checkerBoardImage.getPixelsRef().set(0);
    }
    if (!undistortedImage.isAllocated()) {
        imitate(undistortedImage, inputImage);
        undistortedImage.getPixelsRef().set(0);
    }
    
    if (bRequestCalibrate) {
        cv::Mat img = toCv(inputImage);
        vector<Point2f> pointBuf;
        bool found = findBoard(img, pointBuf);
        Mat outImg = toCv(checkerBoardImage);
        img.copyTo(outImg);
        cv::drawChessboardCorners(outImg, cv::Size(10, 7), pointBuf, found);
        
        if (found) {
            addedImageSize = img.size();
            imagePoints.push_back(pointBuf);
            calibrate();
            bRequestCalibrate = false;
        }
        checkerBoardImage.update();
        
    }
    if (size() > 3) {
        undistort(toCv(inputImage), toCv(undistortedImage));
        undistortedImage.update();
    }
}

void SingleCameraCalibration::draw(int x, int y, int w, int h) {
    if (bRequestCalibrate) {
        checkerBoardImage.draw(x, y, w, h);
    } else {
        inputImage.draw(x, y, w, h);
    }
}

void SingleCameraCalibration::drawUndistorted(int x, int y, int w, int h) {
    undistortedImage.draw(x, y, w, h);
}

bool SingleCameraCalibration::isCalibrated() {
    return size() > 3; // TODO
}

bool SingleCameraCalibration::calibrateWithFlag(int flag) {
    if(size() < 1) {
        ofLog(OF_LOG_ERROR, "Calibration::calibrate() doesn't have any image data to calibrate from.");
        if(ready) {
            ofLog(OF_LOG_ERROR, "Calibration::calibrate() doesn't need to be called after Calibration::load().");
        }
        return ready;
    }
    
    Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
    distCoeffs = Mat::zeros(8, 1, CV_64F);
    
    updateObjectPoints();
    
    int calibFlags = flag;
    float rms = calibrateCamera(objectPoints, imagePoints, addedImageSize, cameraMatrix, distCoeffs, boardRotations, boardTranslations, calibFlags);
    ofLog(OF_LOG_VERBOSE, "calibrateCamera() reports RMS error of " + ofToString(rms));
    
    ready = checkRange(cameraMatrix) && checkRange(distCoeffs);
    
    if(!ready) {
        ofLog(OF_LOG_ERROR, "Calibration::calibrate() failed to calibrate the camera");
    }
    
    distortedIntrinsics.setup(cameraMatrix, addedImageSize);
    updateReprojectionError();
    updateUndistortion();
    
    return ready;
}

//----------------------------------------------------------------------------------

void StereoCameraCalibration::setup(string path, float chessBoardSize) {
    bRequestCalibrate = false;
    imagePointsA.clear();
    imagePointsB.clear();
    filePath = path;
    
    a.setup(path + "_a.yml", chessBoardSize);
    b.setup(path + "_b.yml", chessBoardSize);
}

void StereoCameraCalibration::setup(string path, float chessBoardSize, string pathA, string pathB) {
    bRequestCalibrate = false;
    imagePointsA.clear();
    imagePointsB.clear();
    filePath = path;
    
    a.setup(pathA, chessBoardSize);
    b.setup(pathB, chessBoardSize);
}

void StereoCameraCalibration::requestCalibrateNextFrame() {
    // valid only when both a and b are well calibrated.
    if (a.isCalibrated() && b.isCalibrated()) {
        bRequestCalibrate = true;
    } else {
        if (!a.isCalibrated()) {
            a.requestCalibrateNextFrame();
        }
        if (!b.isCalibrated()) {
            b.requestCalibrateNextFrame();
        }
    }
}

void StereoCameraCalibration::update(ofPixels& pixelsA, ofPixels& pixelsB) {
    a.update(pixelsA);
    b.update(pixelsB);
    
    if (bRequestCalibrate) {
        cv::Mat imgA = toCv(a.inputImage);
        vector<Point2f> pointsA;
        bool foundA = a.findBoard(imgA, pointsA);
        
        cv::Mat imgB = toCv(b.inputImage);
        vector<Point2f> pointsB;
        bool foundB = b.findBoard(imgB, pointsB);
        
        
        if (foundA && foundB && pointsB.size() == pointsA.size()) {
            bRequestCalibrate = false;
            imagePointsA.push_back(pointsA);
            imagePointsB.push_back(pointsB);
        }
    }
}

void StereoCameraCalibration::draw(int x, int y, int w, int h) {
    a.draw(x,       y, w/2, h/2);
    b.draw(x + w/2, y, w/2, h/2);
    a.drawUndistorted(x,       y + h/2, w/2, h/2);
    b.drawUndistorted(x + w/2, y + h/2, w/2, h/2);
}

void StereoCameraCalibration::load() {
    a.load();
    b.load();
    {
        cv::FileStorage fs(filePath, cv::FileStorage::READ);
        fs["rotation"] >> rotation;
        fs["translation"] >> translation;
        
        cv::Mat rot3x3;
        if(rotation.rows == 3 && rotation.cols == 3) {
            rot3x3 = rotation;
        } else	{
            cv::Rodrigues(rotation, rot3x3);
        }
        
        const double* rm = rot3x3.ptr<double>(0);
        const double* tm = translation.ptr<double>(0);
        
        transformAb.makeIdentityMatrix();
        transformAb.set(rm[0], rm[3], rm[6], 0,
                      rm[1], rm[4], rm[7], 0,
                      rm[2], rm[5], rm[8], 0,
                      tm[0], tm[1], tm[2], 1);
        
        // convert coordinate system opencv to opengl
        transformAb.postMultScale(1, -1, -1);
    }
}

void StereoCameraCalibration::save() {
    a.save();
    b.save();
    
    if (imagePointsA.size() == imagePointsB.size() && imagePointsA.size() > 3) {
        write(filePath, a.squareSize, a, b, imagePointsA, imagePointsB);
    }
}

void StereoCameraCalibration::write(const string& filepath, float squareSize, ofxCv::Calibration& src, ofxCv::Calibration& dst,
           vector<vector<Point2f> >& imagePointsSrc, vector<vector<Point2f> >& imagePointsDst) {
    cv::Mat essentialMatrix;
    cv::Mat fundamentalMatrix;
    vector<vector<Point3f> > objectPoints;
    
    vector<Point3f> points = ofxCv::Calibration::createObjectPoints(cv::Size(10,7), squareSize, ofxCv::CHESSBOARD);
    objectPoints.resize(imagePointsSrc.size(), points);
    
    
    Mat cameraMatrixSrc = src.getDistortedIntrinsics().getCameraMatrix();
    Mat cameraMatrixDst = dst.getDistortedIntrinsics().getCameraMatrix();
    Mat distCoeffsSrc = src.getDistCoeffs();
    Mat distCoeffsDst = dst.getDistCoeffs();
    
    cerr << imagePointsDst.size() << endl;
    
    cv::stereoCalibrate(objectPoints,
                        imagePointsSrc, imagePointsDst,
                        cameraMatrixSrc, distCoeffsSrc,
                        cameraMatrixDst, distCoeffsDst,
                        src.getDistortedIntrinsics().getImageSize(), rotation, translation,
                        essentialMatrix, fundamentalMatrix);
    
    FileStorage fs(filepath, FileStorage::WRITE);
    fs << "rotation" << rotation;
    fs << "translation" << translation;
    
}
