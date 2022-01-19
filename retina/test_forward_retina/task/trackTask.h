#ifndef _TRACK_TASK_H
#define _TRACK_TASK_H

#include "dataqueue.h"
#include "../detect/retina_face.h"
#include "../detect/headpose.h"
#include "../track/KalmanFilter/tracker.h"
#include "../track/DeepAppearanceDescriptor/FeatureTensor.h"

#include <thread>



struct YourStruct1 {
	cv::Mat img;
	cv::Rect rect;
	int head_id;
	int direct;
	std::string time;
};
struct YourStruct2 {
	int head_id;
	int direct;
	cv::Mat face;
	std::string time;
};

enum {
	DIRECT_IN,
	DIRECT_OUT
};

struct OBJTRACK {
	cv::Mat img;
	std::vector<FaceA> detects;
};

typedef void(*TrackStart)(void * pUser, std::vector<YourStruct1> &track_starts);
typedef void(*TrackEnd)(void * pUser, std::vector<YourStruct2> &track_ends);



class TrackTask {

private:
	const int MaxSize = 64;
	DataQue<cv::Mat> dataque;
	DataQue<OBJTRACK> obj_queue;
	
	bool stop_ai_magic = false;
	RetinaFace * faceDetect;
	HEADPose * headPose;
	std::shared_ptr<std::thread> detectThread;
	std::shared_ptr<std::thread> trackThread;
	TrackStart track_start_cb = nullptr;
	TrackEnd track_end_cb = nullptr;

	float padding_float_w = 0.04f;
	float padding_float_h = 0.1f;
	float padding_upper = 2.5f;
	bool show = false;
	int min_head = 8;
	float dir_x = 0;
	float dir_y = 0;

	std::shared_ptr<tracker> mytracker;
	std::vector<cv::Point> points;
	std::vector<int> directs;
	
	bool ai_magic_thread_finished = false;

	int screen_width = 1280;
	int screen_height = 720;
	
	float max_angle_pitch = 25;
	float max_angle_yaw = 25;
	float max_angle_roll = 25;
	float best_face_zoom = 1.2f;
	

public:
	TrackTask(RetinaFace * detect, HEADPose * pose);

	void ai_magic_thread();
	void ai_track_thread();

	void stop();
	void start();
	void auto_stop();
	
	void add_data_queue(uchar * bgr_data, int width, int height, bool wait= false);
	void add_data_queue(cv::Mat img, bool wait = false);

	std::vector<Track> get_track_queue();

	RetinaFace * getFaceDetect() {
		return faceDetect;
	}

	void set_start_cb(TrackStart ts) {
		track_start_cb = ts;
	}
	void set_end_cb(TrackEnd tn) {
		track_end_cb = tn;
	}

	void setTrackPoints(std::vector<cv::Point> &points_, std::vector<int> &directs_) {
		points.clear();
		if (points_.size() != directs_.size() || points_.size() < 3) {
			std::cout << "set Track Points failed." << std::endl;
		}
		else {
			points = points_;
			directs = directs_;
			std::cout << "set Track Points success." << std::endl;
		}
	}

	int rect_distance_points(cv::Rect rect);
	bool rect_in_points(cv::Rect rect);

	void set_show(bool show_) {
		show = show_;
	}

	void set_direct(float dirx, float diry) {
		dir_x = dirx;
		dir_y = diry;
	}

	void set_screen(int w, int h) {
		screen_width = w;
		screen_height = h;
	}

	void set_minhead(int min_head_) {
		min_head = min_head_;
	}
	
};


#endif