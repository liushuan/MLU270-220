#pragma once

#include "../task/dataqueue.h"
#include "../task/trackTask.h"
#include <thread>
#include <mutex>

class DecodeTask {

private:
	const int MaxSize = 64;
	bool stop_decoder = false;
	std::shared_ptr<std::thread> DecoderThread;
	std::string file_url;
	TrackTask *trackTask;
	int radio_fps = 30;
	std::mutex mtx;
	int camera_id = 0;
public:
	DecodeTask(std::string url, TrackTask* tk, int gpu_id = 0);
	void decoder_thread2();
	void decoder_thread3();
	void stop();
	void start();
	void set_fps(int fsp) {
		radio_fps = fsp;
	}
	~DecodeTask();

};
