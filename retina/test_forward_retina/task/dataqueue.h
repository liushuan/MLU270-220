#ifndef _DATA_QUE__H
#define _DATA_QUE__H
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>

template <class type>
class DataQue {


private:
	std::queue<type> data;
	int MaxSize = 64;

	std::mutex mtx;
	std::condition_variable cnd_push;
	std::condition_variable cnd_pop;
	bool stop_queue = false;
	int drop_frame_count = 0;
public:

	void add2queue(const type & img, void(*clear_method)(type), bool wait = false) {
		std::unique_lock <std::mutex> lck(mtx);
		//rtsp 必须立刻插入   MP4 或者 dir 可以等待处理完成再插入
		if (!wait) {
			if (data.size() >= MaxSize) {
				if (clear_method != NULL) {
					type img = data.front();
					clear_method(img);
				}
				data.pop();
				drop_frame_count++;
				if (drop_frame_count > 24) {
					//std::cout << "....................drop frame :" << drop_frame_count << std::endl;
					drop_frame_count = 0;
				}
			}
		}
		else {
			while (data.size() >= MaxSize && !stop_queue) {
				
				cnd_push.wait(lck);
			}
		}
		data.push(img);
		cnd_pop.notify_all();
	}

	void resume_queue() {
		std::unique_lock <std::mutex> lck(mtx);
		stop_queue = false;
	}

	void just_stop_queue() {
		std::unique_lock <std::mutex> lck(mtx);
		stop_queue = true;
		cnd_pop.notify_all();
		cnd_push.notify_all();
	}

	void stop_clear_queue(void(*clear_method)(type)) {
		std::unique_lock <std::mutex> lck(mtx);
		stop_queue = true;
		while (!data.empty()) {
			if (clear_method != NULL) {
				type img = data.front();
				clear_method(img);
			}
			data.pop();
		}
		cnd_pop.notify_all();
		cnd_push.notify_all();
	}

	int getsize() {
		std::unique_lock <std::mutex> lck(mtx);
		return data.size();
	}
	void setMaxSize(int max_size) {
		MaxSize = max_size;
	}
	int getMaxSize() {
		return MaxSize;
	}

	std::vector<type> getfromqueue(const int maxbatch) {
		std::unique_lock <std::mutex> lck(mtx);
		std::vector<type> result;
		while (data.empty() && !stop_queue)
		{
			cnd_pop.wait(lck);
		}
		for (size_t i = 0; i < maxbatch && !data.empty(); i++)
		{
			type img = data.front();
			result.push_back(img);
			data.pop();
		}
		cnd_push.notify_all();
		return result;
	}
};

#endif