#include <iostream>
#include <math.h>
#include <fstream>
#include <memory>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

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


int main(int argc, char* argv[])
{
	cout << "Hi, this is MNN test program!" << endl;

#if 1
    //const auto poseModel = "../model/best-resnet_305.mnn";
    const auto poseModel = "../model/best-resnet_305.mnn";
    const auto inputImageFileName = "d:/workroom/testroom/v0.png";
    int originalWidth;
    int originalHeight;
    int originChannel;
    auto inputImage = stbi_load(inputImageFileName, &originalWidth, &originalHeight, &originChannel, 3);
    if (nullptr == inputImage) {
        MNN_ERROR("Invalid path: %s\n", inputImageFileName);
        return 0;
    }
    cout << originalWidth << " " << originalHeight << " " << originChannel << endl;
    cout << int(inputImage[0]) << " " << int(inputImage[3]) << " "  << int(inputImage[4]) << endl;

    // create net and session
    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(poseModel));
    MNN::ScheduleConfig netConfig;
    netConfig.type = MNN_FORWARD_CPU;
    netConfig.numThread = 3;
    auto session = mnnNet->createSession(netConfig);

    int targetWidth = originalWidth;
    int targetHeight = originalHeight;
    //const int targetWidth = static_cast<int>((float)originalWidth / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
    //const int targetHeight = static_cast<int>((float)originalHeight / (float)OUTPUT_STRIDE) * OUTPUT_STRIDE + 1;
    cout << "targetWidth : " << targetWidth << endl;
    cout << "targetHeight : " << targetHeight << endl;

    auto input = mnnNet->getSessionInput(session, nullptr);
    cout << "input : " << input->batch() << " " << input->size() << " " << input->elementSize() << " " << input->channel() << " " << input->width() << " " << input->height() << endl;
    //if (input->elementSize() <= 4) 
    {
        mnnNet->resizeTensor(input, { 1, 3, targetHeight, targetWidth });
        mnnNet->resizeSession(session);
    }
    cout << "input : " << input->batch() << " " << input->elementSize() << " " << input->channel() << " " << input->width() << " " << input->height() << endl;

    auto output = mnnNet->getSessionOutput(session, nullptr);
    cout << "output : " << output->batch() << " " << output->elementSize() << " " << output->channel() << " " << output->width() << " " << output->height() << endl;
//   if (output->elementSize() <= 4) 
    {
        mnnNet->resizeTensor(output, { 1, 3, targetHeight, targetWidth });
        mnnNet->resizeSession(session);
    }
    cout << "output : " << output->batch() << " " << output->elementSize() << " " << output->channel() << " " << output->width() << " " << output->height() << endl;

    // preprocess input image
    {
        const float means[3] = { 0.0f, 0.0f, 0.0f };
        const float norms[3] = { 1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f };
        CV::ImageProcess::Config preProcessConfig;
        ::memcpy(preProcessConfig.mean, means, sizeof(means));
        ::memcpy(preProcessConfig.normal, norms, sizeof(norms));
        preProcessConfig.sourceFormat = CV::RGB;
        preProcessConfig.destFormat = CV::RGB;
        preProcessConfig.filterType = CV::BILINEAR;

        auto pretreat = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(preProcessConfig));
        CV::Matrix trans;

        // Dst -> [0, 1]
        trans.postScale(1.0 / targetWidth, 1.0 / targetHeight);
        //[0, 1] -> Src
        trans.postScale(originalWidth, originalHeight);

        pretreat->setMatrix(trans);
        const auto rgbaPtr = reinterpret_cast<uint8_t*>(inputImage);
        pretreat->convert(rgbaPtr, originalWidth, originalHeight, 0, input);
    }

    // run...
    cout << "start run..." << endl;
    cout << input->elementSize() << " " << input->channel() << " " << input->width() << " " << input->height() << endl;

    {
        AUTOTIME;
        cout << "..." << endl;
        mnnNet->runSession(session);
    }
    {
        AUTOTIME;
        MNNTensor_2_RGB(output, inputImage);
    }
    
    //stbi_write_png("d:/1.png", originalWidth, originalHeight, 3, inputImage, 3 * originalWidth);
    stbi_write_jpg("d:/1.jpg", originalWidth, originalHeight, 3, inputImage, 3 * originalWidth);
    stbi_image_free(inputImage);

    cout << "done" << endl;
#endif

	return 0;
}
