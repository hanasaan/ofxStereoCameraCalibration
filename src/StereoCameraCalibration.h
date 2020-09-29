#pragma once

#include "ofMain.h"
#include "ofxCv.h"

class StereoCameraCalibration;

class SingleCameraCalibration : public ofxCv::Calibration {
protected:
    bool bRequestCalibrate;
    int  calibrationFrameCount;
    int notFoundFrameCount;
    string filePath;
    bool bAbsolute;
    
    ofImage inputImage;
    ofImage checkerBoardImage;
    ofImage undistortedImage;
    
    friend class StereoCameraCalibration;
    
    float scale;
    float scale_min;
public:
    void setup(string defaultFilePath, float chessBoardSize, bool absolute = false);
    void load();
    void save();
    void setMinScale(float s) {
        scale_min = s;
    }
    
    void requestCalibrateNextFrame();
    
    void update(ofPixels& pixels);
    void draw(int x, int y, int w, int h);
    void drawUndistorted(int x, int y, int w, int h);
    bool isCalibrated();
    
    bool calibrateWithFlag(int flag);
};

class StereoCameraCalibration {
protected:
    SingleCameraCalibration a;
    SingleCameraCalibration b;
    bool bRequestCalibrate;
    bool bAbsolute;
	int notFoundFrameCount;
	
    vector<vector<cv::Point2f> > imagePointsA;
    vector<vector<cv::Point2f> > imagePointsB;
    
    string filePath;
    
    cv::Mat translation;
    cv::Mat rotation;
    ofMatrix4x4 transformAb;
public:
    StereoCameraCalibration();
    void setup(string path, float chessBoardSize, bool absolute = false, string pathA = "", string pathB = "");
    void update(ofPixels& pixelsA, ofPixels& pixelsB);
    void draw(int x, int y, int w, int h);
    
    void load();
    void save();
    
    void requestCalibrateNextFrame();
    
    SingleCameraCalibration& getARef() {return a;}
    SingleCameraCalibration& getBRef() {return b;}
    const ofMatrix4x4& getTransformAbRef() {return transformAb;}
    ofMatrix4x4 getTransformAb() {return transformAb;}
    bool isRequested() {return bRequestCalibrate;}
    int size() {return imagePointsA.size();}
    const cv::Mat& getTranslation() const {return translation;}
    const cv::Mat& getRotation() const {return rotation;}
    
private:
    void write(float squareSize, ofxCv::Calibration& src, ofxCv::Calibration& dst,
               vector<vector<cv::Point2f> >& imagePointsSrc, vector<vector<cv::Point2f> >& imagePointsDst);
    void updateTransformAb();
};

