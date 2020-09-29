#include "StereoCameraCalibration.h"

using namespace cv;
using namespace ofxCv;

void SingleCameraCalibration::setup(string defaultFilePath, float chessBoardSize, bool absolute) {
    ofxCv::Calibration::reset();
    bRequestCalibrate = false;
    bAbsolute = absolute;
    calibrationFrameCount = 0;
    notFoundFrameCount = 0;
    filePath = defaultFilePath;
    setPatternSize(10, 7);
    setSquareSize(chessBoardSize);
    scale = 1.0;
    scale_min = 1.0;
}

void SingleCameraCalibration::load() {
    if (ofFile(ofToDataPath(filePath, bAbsolute)).exists()) {
        ofxCv::Calibration::load(filePath, bAbsolute);
    }
}

void SingleCameraCalibration::save() {
    if (size() > 3) {
        ofxCv::Calibration::save(filePath, bAbsolute);
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
        checkerBoardImage.update();
    }
    if (!undistortedImage.isAllocated()) {
        imitate(undistortedImage, inputImage);
        undistortedImage.getPixelsRef().set(0);
        undistortedImage.update();
    }
    
    if (bRequestCalibrate) {
        cv::Mat img_orig = toCv(inputImage);
        cv::Mat img;
        if (scale == 1.0) {
            img = img_orig;
        } else {
            cv::resize(img_orig, img, cv::Size(), scale, scale);
        }
        vector<Point2f> pointBuf;
        bool found = findBoard(img, pointBuf);
        for (auto& p : pointBuf) {
            p.x /= scale;
            p.y /= scale;
        }
        Mat outImg = toCv(checkerBoardImage);
        img.copyTo(outImg);
        cv::drawChessboardCorners(outImg, getPatternSize(), pointBuf, found);
        
        if (found) {
            addedImageSize = img.size();
            imagePoints.push_back(pointBuf);
            calibrate();
            bRequestCalibrate = false;
            cerr << "found at scale " << scale << endl;
            scale = 1.0;
        } else {
            notFoundFrameCount++;
            if (notFoundFrameCount > 3) {
                notFoundFrameCount = 0;
                if (scale <= scale_min) {
                    scale = 1.0;
                    bRequestCalibrate = false;
                } else {
                    scale *= 0.5;
                }
            }
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
        ofPushStyle();
        ofNoFill();
        glLineWidth(10);
        ofSetColor(ofColor::red);
        ofRect(x, y, w, h);
        ofPopStyle();
    } else {
        inputImage.draw(x, y, w, h);
    }
    
    {
        stringstream ss;
        ss << "Size   : " << size() << endl;
        ss << "RepErr : " << getReprojectionError() << endl;
        ss << "Focal  : " << getDistortedIntrinsics().getFocalLength() << endl;
        ss << "Principal Point : " << getDistortedIntrinsics().getPrincipalPoint() << endl;
        ofDrawBitmapStringHighlight(ss.str(), x + 10, y + 20);
    }
}

void SingleCameraCalibration::drawUndistorted(int x, int y, int w, int h) {
    if (undistortedImage.isAllocated()) {
        undistortedImage.draw(x, y, w, h);
    }
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
StereoCameraCalibration::StereoCameraCalibration() {
    bRequestCalibrate = false;
    bAbsolute = false;
	notFoundFrameCount = 0;
}


void StereoCameraCalibration::setup(string path, float chessBoardSize, bool absolute, string pathA, string pathB) {
    bRequestCalibrate = false;
    bAbsolute = absolute;
    imagePointsA.clear();
    imagePointsB.clear();
    filePath = path;
    
    if (pathA.empty()) {
        pathA = path + "_a.yml";
    }
    if (pathB.empty()) {
        pathB = path + "_b.yml";
    }
    
    a.setup(pathA, chessBoardSize, absolute);
    b.setup(pathB, chessBoardSize, absolute);
}

void StereoCameraCalibration::requestCalibrateNextFrame() {
    // valid only when both a and b are well calibrated.
    if (a.isCalibrated() && b.isCalibrated()) {
        bRequestCalibrate = true;
		notFoundFrameCount = 0;
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
        
        
		notFoundFrameCount++;
        if (foundA && foundB && pointsB.size() == pointsA.size()) {
			bRequestCalibrate = false;
            imagePointsA.push_back(pointsA);
            imagePointsB.push_back(pointsB);
        }
		if (notFoundFrameCount > 3) {
			bRequestCalibrate = false;
		}
    }
}

void StereoCameraCalibration::draw(int x, int y, int w, int h) {
    float aw = w/2;
    float ah = aw * a.inputImage.getHeight() / a.inputImage.getWidth();
    float bw = w/2;
    float bh = bw * b.inputImage.getHeight() / b.inputImage.getWidth();
    
    a.draw(x,       y, aw, ah);
    b.draw(x + w/2, y, bw, bh);
    a.drawUndistorted(x,       y + h/2, aw, ah);
    b.drawUndistorted(x + w/2, y + h/2, bw, bh);
    if (bRequestCalibrate) {
        ofPushStyle();
        ofNoFill();
        glLineWidth(5);
        ofSetColor(ofColor::red);
        ofRect(x, y, w, h);
        ofPopStyle();
    }
    
    {
        stringstream ss;
        ss << "Stereo Calib Info" << endl;
        ss << "Size   : " << size() << endl;
        ss << "Translate : " << translation << endl;
        ss << "Rotate  : " << rotation << endl;
        ss << "SquareSize : " << a.getSquareSize();
        ofDrawBitmapStringHighlight(ss.str(), x + 10, y + h - 100, ofColor::red);
    }
}

void StereoCameraCalibration::load() {
    a.load();
    b.load();
    if (ofFile(ofToDataPath(filePath, bAbsolute)).exists()) {
        cv::FileStorage fs(ofToDataPath(filePath, bAbsolute), cv::FileStorage::READ);
        fs["rotation"] >> rotation;
        fs["translation"] >> translation;
        updateTransformAb();
    }
}

void StereoCameraCalibration::save() {
    a.save();
    b.save();
    
    if (imagePointsA.size() == imagePointsB.size() && imagePointsA.size() > 3) {
        write(a.squareSize, a, b, imagePointsA, imagePointsB);
    }
}

void StereoCameraCalibration::updateTransformAb() {
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
    
    transformAb.preMultScale(ofVec3f(1, -1, -1));
    transformAb.postMultScale(ofVec3f(1, -1, -1));
}

void StereoCameraCalibration::write(float squareSize, ofxCv::Calibration& src, ofxCv::Calibration& dst,
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
    
    FileStorage fs(ofToDataPath(filePath, bAbsolute), FileStorage::WRITE);
    fs << "rotation" << rotation;
    fs << "translation" << translation;
    updateTransformAb();
}
