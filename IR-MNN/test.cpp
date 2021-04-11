#include <iostream>
#include <math.h>
#include <fstream>
#include <memory>
#include <opencv2/highgui/highgui.hpp>    
#include <opencv2/imgproc/imgproc.hpp>
#include <MNN/Interpreter.hpp>

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN;
using namespace std;
using namespace cv;

#define LClip(x, lmin ,lmax) ((x)<lmin ? lmin : ( (x)>lmax ? lmax : (x) ))

//const char* modelFile = "../model/best-resnet_305-q.mnn";
const char* modelFile = "d:/nir9_best.mnn";

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
    int factor = t.stride(0) / (width * height);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char* s = (uchar *)&m.at<cv::Vec3b>(i, j)[0];
            float* d = t.host<float>() + (i * width + j) * factor;
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
    int factor = t.stride(0) / (width*height);
    float* r = t.host<float>() + t.stride(1) * 0;
    float* g = t.host<float>() + t.stride(1) * 1;
    float* b = t.host<float>() + t.stride(1) * 2;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            unsigned char* d = (uchar*)&m.at<cv::Vec3b>(i, j)[0];
            int offset = i * width + j;
            d[0] = LClip(r[offset] * 255, 0, 255);
            d[1] = LClip(g[offset] * 255, 0, 255);
            d[2] = LClip(b[offset] * 255, 0, 255);
        }
    }
}

static void IR_Image()
{
    const auto poseModel = modelFile;
    const auto inputImageFileName = "d:/workroom/testroom/v0.png";
    int width, height;
    cv::Mat img;
    img = cv::imread(inputImageFileName);
    width = img.cols;
    height = img.rows;
    cout << "img chn : " << img.channels() << endl;

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type = MNN_FORWARD_CPU;
    netConfig.numThread = 3;
    auto session = mnnNet->createSession(netConfig);

    auto input = mnnNet->getSessionInput(session, nullptr);
    input->printShape();
    //input->print();
    mnnNet->resizeTensor(input, { 1, 3, height, width });
    //mnnNet->resizeSession(session);

    auto output = mnnNet->getSessionOutput(session, nullptr);
    output->printShape();
    cout << "stride : " << input->stride(0) << " " << input->stride(1) << " " << input->stride(2) << endl;
    cout << "stride : " << output->stride(0) << " " << output->stride(1) << " " << output->stride(2) << endl;
    //output->print();

    mnnNet->resizeTensor(output, { 1, 3, height, width });
    mnnNet->resizeSession(session);
    input->printShape();
    output->printShape();
    cout << "stride : " << input->stride(0) << " " << input->stride(1) << " " << input->stride(2) << endl;
    cout << "stride : " << output->stride(0) << " " << output->stride(1) << " " << output->stride(2) << endl;

    Mat_2_MNNTensor(img, *input);

    // run...
    cout << "start run..." << endl;

    {
        AUTOTIME;
        mnnNet->runSession(session);
    }
    output = mnnNet->getSessionOutput(session, nullptr);
    MNNTensor_2_Mat(*output, img);

    cv::imwrite("d:/1.jpg", img);
    cout << "done" << endl;
}

void IR_Video()
{
    VideoCapture inputV("d:/workroom/testroom/fei-light.mp4");
    VideoWriter outputV;
    if (!outputV.open("d:/workroom/testroom/fei-light-mnn.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), inputV.get(CAP_PROP_FPS), \
            Size(inputV.get(CAP_PROP_FRAME_WIDTH), inputV.get(CAP_PROP_FRAME_HEIGHT))))
        return;

    int width, height;
    width = inputV.get(CAP_PROP_FRAME_WIDTH);
    height = inputV.get(CAP_PROP_FRAME_HEIGHT);

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(modelFile));
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

    Mat frame;
    int count = 0;
    while (1) {
        std::cout << "frame id : " << count << endl;

        inputV >> frame;
        if (frame.empty())
            break;
        
        Mat_2_MNNTensor(frame, *input);
        mnnNet->runSession(session);
        MNNTensor_2_Mat(*output, frame);

        outputV << frame;
        count++;
        if (count > 25 * 3)
            break;
    }
    mnnNet->releaseSession(session);

    outputV.release();
    inputV.release();
}


int main(int argc, char* argv[])
{
	cout << "Hi, this is MNN test program!" << endl;

    //IR_Video();
    IR_Image();

	return 0;
}
