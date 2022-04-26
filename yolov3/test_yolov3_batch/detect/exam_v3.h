#ifndef _EXAM_DETECT__H_
#define _EXAM_DETECT__H_

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cnrt.h>

struct Box {
	float xmin;
	float ymin;
	float xmax;
	float ymax;
	float score;
	int classes;
};

struct ExamModel
{
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
	int dev_id = 0;
	int dev_channel = -1;
	cnrtInvokeParam_t 		invokeParam;				//invoke参数
	
};


class EXAMDetect {

private:
	/*cnrtDev_t dev;
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
	int dev_id = 0;
	int dev_channel = -1;
	cnrtInvokeParam_t 		invokeParam;				//invoke参数*/
	
	
	ExamModel exam_model;
	ExamModel exam_model_batch;
	
	const float threhold = 0.25f;
	const float iou_threhold = 0.5f;
	
	const int input_size_w = 512;
	const int input_size_h = 512;
	std::vector<float> stdvs = {1.0f/255, 1.0f/255, 1.0f/255};

public:
	int batch_size = 4;
	const int classify_number = 24;
	std::string OBJ_LABELS[24] = {"idle","write","vacant","lpeep","rpeep","bpeep","signal","handsdown","handspeep","handpeep",
                  "getitem","changeitem","opitem","sleep","standup","handsup","drinkwater","destroypaper","turnpaper","stretch",
                  "teacher_normal","teacher_smoke","teacher_book","teacher_dozing"};
	EXAMDetect(std::string models_path, std::string models_batch_path);

	void Detect(cv::Mat& img, std::vector<Box>&exams);
	
	void Detect(std::vector<cv::Mat>&imgs, std::vector<std::vector<Box>>& exams);
	
	~EXAMDetect();
	
};
#endif 