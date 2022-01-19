#ifndef _HEAD_POSE_
#define _HEAD_POSE_

#include <opencv2/opencv.hpp>
#include <vector>

#include <glog/logging.h>
#include <cnrt.h>

struct Pose {
	float roll;
	float yaw;
	float pitch;
};


class HEADPose {

private:
	cnrtDev_t dev;
	cnrtModel_t model;
	cnrtFunction_t function;
	cnrtRuntimeContext_t rt_ctx_;
	int inputNum, outputNum;
	int64_t* inputSizeS = nullptr;
    int64_t* outputSizeS = nullptr;
	cnrtDataType_t* input_data_type = nullptr;
    cnrtDataType_t* output_data_type = nullptr;
	
	void** inputCpuPtrS = nullptr;
	void** outputCpuPtrS = nullptr;
	void** outTransCpuPtrS = nullptr;
	void** param = nullptr;
    std::vector<int> in_count;
    std::vector<int> out_count;
	
	void** inputMluPtrS = nullptr;
	void** outputMluPtrS = nullptr;
	
	cnrtQueue_t cnrt_queue;
	
	const int input_size_w = 48;
	const int input_size_h = 48;

	std::vector<float> means = { 104, 117, 124 };
	std::vector<float> stdvs = {0.0078125f, 0.0078125f, 0.0078125f};

public:
	HEADPose(std::string models_path);
	void init();
	
	Pose Detect(cv::Mat& img);
	
	~HEADPose();
	
};
#endif 