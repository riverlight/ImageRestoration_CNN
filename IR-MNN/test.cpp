#include <iostream>
#include <math.h>
#include <fstream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>


#include <MNN/ImageProcess.hpp>
#include <MNN/Interpreter.hpp>


#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
using namespace MNN;


#define OUTPUT_STRIDE 16


using namespace std;

#define LClip(x, lmin ,lmax) ((x)<lmin ? lmin : ( (x)>lmax ? lmax : (x) ))

void MNNTensor_2_RGB(MNN::Tensor* pTensor, unsigned char* rgb)
{
    int width = pTensor->width();
    int height = pTensor->height();
    float* pSrc = pTensor->host<float>();
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float* s = pSrc + (i * width + j) * 4;
            unsigned char* d = rgb + (i * width + j) * 3;
            d[0] = LClip(s[0]*255, 0, 255);
            d[1] = LClip(s[1] * 255, 0, 255);
            d[2] = LClip(s[2] * 255, 0, 255);
        }
    }
}

void Mat_2_MNNTensor(cv::Mat& m, MNN::Tensor& t)
{
    int width = m.cols;
    int height = m.rows;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char* s = (uchar *)&m.at<cv::Vec3b>(i, j)[0];
            float* d = t.host<float>() + (i * width + j) * 4;
            d[0] = float(s[0]) / 255.0;
            d[1] = float(s[1]) / 255.0;
            d[2] = float(s[2]) / 255.0;
        }
    }
}

void MNNTensor_2_Mat(MNN::Tensor& t, cv::Mat& m)
{
    int width = m.cols;
    int height = m.rows;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char* d = (uchar*)&m.at<cv::Vec3b>(i, j)[0];
            float* s = t.host<float>() + (i * width + j) * 4;
            d[0] = LClip(s[0] * 255, 0, 255);
            d[1] = LClip(s[1] * 255, 0, 255);
            d[2] = LClip(s[2] * 255, 0, 255);
        }
    }
}


int main(int argc, char* argv[])
{
	cout << "Hi, this is MNN test program!" << endl;

    const auto poseModel = "../model/best-resnet_305.mnn";
    const auto inputImageFileName = "d:/workroom/testroom/v0.png";
    int width, height;
    cv::Mat img;
    img = cv::imread(inputImageFileName);
    width = img.cols;
    height = img.rows;

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type = MNN_FORWARD_CPU;
    netConfig.numThread = 3;
    auto session = mnnNet->createSession(netConfig);

    auto input = mnnNet->getSessionInput(session, nullptr);
    mnnNet->resizeTensor(input, { 1, 3, height, width });
    mnnNet->resizeSession(session);

    auto output = mnnNet->getSessionOutput(session, nullptr);
    mnnNet->resizeTensor(output, { 1, 3, height, width });
    mnnNet->resizeSession(session);

    Mat_2_MNNTensor(img, *input);

    // run...
    cout << "start run..." << endl;

    {
        AUTOTIME;
        mnnNet->runSession(session);
    }
    MNNTensor_2_Mat(*output, img);
    
    cv::imwrite("d:/1.jpg", img);
    cout << "done" << endl;

	return 0;
}
