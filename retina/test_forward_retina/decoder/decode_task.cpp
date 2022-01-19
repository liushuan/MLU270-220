#include "decode_task.h"
#include <unistd.h>
#include <dirent.h>
#include <istream>

DecodeTask::DecodeTask(std::string url, TrackTask* tk, int gpu_id) {
	if (url == "" || tk == nullptr) {
		std::cout << "url error.......  url is:" << url << ":end" << std::endl;
		return;
	}
	file_url = url;
	trackTask = tk;
}

void DecodeTask::decoder_thread2() {
	long frame_count = 0;
	std::cout << "start ai_generate_data_thread" << std::endl;
	//opencv 读取视频文件
	cv::VideoCapture capture(file_url);
	if (!capture.isOpened())
	{
		std::cout << "Unable to connect to :" << file_url << std::endl;
		getchar();
		return;
	}
	std::cout << "start" << std::endl;
	double start = cv::getTickCount();

	while (!stop_decoder)
	{

		cv::Mat frame;
		//视频文件中的每帧图像读到 frame中
		capture >> frame;
		
		if (frame.empty()) {
			std::cout << "data is null." << std::endl;
			break;
		}

		frame_count++;

		if (frame_count % radio_fps == 0 && radio_fps < 30) {
			continue;
		}
		//cv::resize(frame, frame, cv::Size(trackTask->getHeadDetect()->INPUT_W, trackTask->getHeadDetect()->INPUT_H));
		trackTask->add_data_queue(frame, true);

	}
	double end = cv::getTickCount();
	std::cout << "all frame is:" << frame_count << std::endl;
	std::cout << "cost time is:" << (end - start) / cv::getTickFrequency() * 1000 << std::endl;

	std::cout << "decoder thread end." << std::endl;
	if (trackTask != nullptr) {
		trackTask->auto_stop();
	}

}


template <typename Type >
static Type stringToNum(const std::string& str)
{
		std::istringstream iss(str);
		Type num;
		iss >> num;
		return num;
}
static void getFiles(std::string cate_dir, std::vector<std::string> &filess, std::vector<std::string> &file_names, const std::string & tail)
	{
		DIR *dir;
		struct dirent *ptr;
		char base[1000];

		if ((dir = opendir(cate_dir.c_str())) == NULL)
		{
			perror("Open dir error...");
			return;
		}
		while ((ptr = readdir(dir)) != NULL)
		{
			if (strstr(ptr->d_name, "video") != NULL){
				std::string file_name = ptr->d_name;
				file_names.push_back(file_name);
				filess.push_back(cate_dir + "/" + file_name);
				
			}
		}
		closedir(dir);
	}
static int get_camera_id(std::string path){
	std::vector<std::string>files, file_names;
	getFiles(path, files, file_names, "video");
	int min_index = 0xFFFFFF;
	for (int i = 0; i < files.size(); i++){
		if (file_names[i].length() > 6){
			int index = stringToNum<int>(file_names[i].replace(file_names[i].find("video"), 5, ""));
			
			if (index < min_index){
				min_index = index;
			}
		}
	}
	return min_index;
}

void DecodeTask::decoder_thread3() {
	std::cout << "start ai_generate_data_thread" << std::endl;
	start:
	std::cout << "start connect camera." << std::endl;
	long frame_count = 0;
	//opencv 读取视频文件
	try{
		camera_id = get_camera_id("/dev");

		cv::VideoCapture capture(camera_id);
		capture.set(cv::CAP_PROP_FRAME_WIDTH, 1920);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);
		capture.set(cv::CAP_PROP_FPS, 30.0);
		if (!capture.isOpened())
		{
			std::cout << "unable to connect to camera_id:" << camera_id <<" , please check your camera."<< std::endl;
			sleep(3);
			goto start;
		}
		
		int camera_offine_count = 0;
		while (!stop_decoder)
		{
			cv::Mat frame;
			//视频文件中的每帧图像读到 frame中
			capture >> frame;

			if (frame.empty()) {
				camera_offine_count++;
				if (camera_offine_count % 30 == 0){
					std::cout << "camera offine." << std::endl;
				}
				if (camera_offine_count > 100){
					capture.release();
					goto start;
				}
				continue;
			}

			frame_count++;

			if (frame_count % radio_fps == 0 && radio_fps < 30) {
				continue;
			}
			//cv::resize(frame, frame, cv::Size(trackTask->getCarDetect()->INPUT_W, trackTask->getCarDetect()->INPUT_H));
			trackTask->add_data_queue(frame);
			
			if (frame_count > 1000000){
				frame_count = 0;
			}
		}
		capture.release();
	}catch(...){
		std::cout<<"camera open failed, please check your camera."<<std::endl;
		sleep(2);
		goto start;
	}
	
	std::cout << "decoder thread end." << std::endl;
	if (trackTask != nullptr) {
		trackTask->auto_stop();
	}
	
}


void DecodeTask::stop() {
	stop_decoder = true;
	if (DecoderThread->joinable()) {
		DecoderThread->join();
	}
}

void DecodeTask::start() {
	stop_decoder = false;
	if (file_url.find(".") != std::string::npos) {
		DecoderThread = std::make_shared<std::thread>(&DecodeTask::decoder_thread2, this);
	}
	else {
		camera_id = atoi(file_url.c_str());
		DecoderThread = std::make_shared<std::thread>(&DecodeTask::decoder_thread3, this);
	}
}


DecodeTask::~DecodeTask() {
}
