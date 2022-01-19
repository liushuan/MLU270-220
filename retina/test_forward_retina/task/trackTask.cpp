#include "trackTask.h"

#include <time.h>
#include <unistd.h>

static bool rect_in_padding(cv::Rect rect, int padding_x1, int padding_y1, int padding_x2, int padding_y2) {
	int center_x = rect.x + rect.width / 2;
	int center_y = rect.y + rect.height / 2;
	if (center_x > padding_x1 && center_x < padding_x2 && center_y > padding_y1 && center_y < padding_y2) {
		return true;
	}
	return false;
}


TrackTask::TrackTask(RetinaFace * detect, HEADPose * pose) {
	faceDetect = detect;
	headPose = pose;
	if (detect == nullptr) {
		std::cout << "detect model is nullptr. init failed." << std::endl;
		return;
	}
	const int nn_budget = 50;
	const float max_cosine_distance = 0.8f;
	const float max_iou_threhold = 0.8f;
	const int max_age = 30;
	const int ninit = 3;
	//deep SORT
	mytracker = std::make_shared<tracker>(max_cosine_distance, nn_budget, max_iou_threhold, max_age, ninit);
}

static std::string get_time(std::string &dt) {
	time_t t;
	struct tm *tmp;
	char buf2[64];
	/* 获取时间 */
	time(&t);
	tmp = localtime(&t);
	/* 转化时间 */
	if (strftime(buf2, 64, "time : %r, %a %b %d, %Y", tmp) == 0) {
		printf("buffer length 64 is too small\n");
	}
	dt = std::to_string(tmp->tm_year) + std::to_string(tmp->tm_mon) + std::to_string(tmp->tm_mday) + std::to_string(tmp->tm_hour) + std::to_string(tmp->tm_min) + std::to_string(tmp->tm_sec);
	return buf2;
}


void TrackTask::ai_magic_thread() {
	std::cout << "ai_magic_thread thread start." << std::endl;
	dataque.setMaxSize(30);
	ai_magic_thread_finished = false;
	while (!stop_ai_magic || dataque.getsize() > 0) {
		std::vector<cv::Mat> datas = dataque.getfromqueue(faceDetect->batch_size);
		if (datas.size() > 0) {
			//auto start = std::chrono::system_clock::now();
			std::vector<FaceA> faces;
			faceDetect->Detect(datas[0], faces);
			//auto end = std::chrono::system_clock::now(); 
			//std::cout << "detect time is:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
			OBJTRACK objt;
			objt.img = datas[0];
			objt.detects = faces;
			obj_queue.add2queue(objt, nullptr, true);
			datas.clear();
		}
	}
	ai_magic_thread_finished = true;
	std::cout << "ai_magic_thread task thread end!" << std::endl;
}

static cv::Mat get_zoom_rect(cv::Mat img, cv::Rect rect, float zoom = 1.5f) {
		cv::Rect n_rect;
		n_rect.height = rect.height * zoom;
		n_rect.width = rect.width*zoom;
		n_rect.x = rect.x - rect.width*(zoom - 1) / 2;
		n_rect.y = rect.y - rect.height*(zoom - 1) / 2;
	
		n_rect.x = std::max(n_rect.x, 0);
		n_rect.y = std::max(n_rect.y, 0);
		n_rect.width = std::min(n_rect.width, img.cols - n_rect.x - 1);
		n_rect.height = std::min(n_rect.height, img.rows - n_rect.y - 1);
		return img(n_rect);
}

static cv::Mat square_rect_Mat(cv::Mat img, cv::Rect rect) {
	cv::Point2f center(rect.x + rect.width/2, rect.y + rect.height/ 2);
	
	int max = std::max(rect.width, rect.height);
	rect.width = rect.height = max;
	
	
	rect.x = center.x - max / 2;
	rect.y = center.y - max / 2;

	rect.x = std::max(rect.x, 0);
	rect.y = std::max(rect.y, 0);
	rect.width = std::min(rect.width, img.cols - rect.x);
	rect.height = std::min(rect.height, img.rows - rect.y);
	cv::Mat roi = img(rect);
	cv::resize(roi, roi, cv::Size(48, 48));
	return roi;
}


void TrackTask::ai_track_thread() {
	std::cout << "start track thread" << std::endl;
	while (!stop_ai_magic || obj_queue.getsize() > 0 || !ai_magic_thread_finished) {
		std::vector<OBJTRACK> datas = obj_queue.getfromqueue(1);
		if (datas.size() > 0) {
			//auto start = std::chrono::system_clock::now();
			if (points.size() == 0) {
				int padding_x1 = datas[0].img.cols * padding_float_w;
				int padding_y1 = datas[0].img.rows * padding_float_h * padding_upper;
				int padding_x2 = datas[0].img.cols *(1 - padding_float_w);
				int padding_y2 = datas[0].img.rows*(1 - padding_float_h);
				points.push_back(cv::Point(padding_x1, padding_y1));
				points.push_back(cv::Point(padding_x2, padding_y1));
				points.push_back(cv::Point(padding_x2, padding_y2));
				points.push_back(cv::Point(padding_x1, padding_y2));
				directs.push_back(DIRECT_OUT);
				directs.push_back(DIRECT_IN); 
				directs.push_back(DIRECT_IN);
				directs.push_back(DIRECT_IN);
			}
			DETECTIONS detections;
			for (size_t j = 0; j < datas[0].detects.size(); j++)
			{
				cv::Rect rect(datas[0].detects[j].rect.x, datas[0].detects[j].rect.y, datas[0].detects[j].rect.width, datas[0].detects[j].rect.height);
				if (rect_in_points(rect) && rect.width > min_head && rect.height > min_head) {

					DETECTION_ROW detect_row;
					detect_row.confidence = 1.0f;
					rect.x = std::max(rect.x, 0);
					rect.y = std::max(rect.y, 0);
					rect.width = std::min(rect.width, datas[0].img.cols-rect.x-1);
					rect.height = std::min(rect.height, datas[0].img.rows - rect.y - 1);
					detect_row.tlwh = DETECTBOX(rect.x, rect.y, rect.width, rect.height);
					detect_row.classify = 0;
					detect_row.confidence = datas[0].detects[j].score;
					detections.push_back(detect_row);
				}
			}
			//std::cout << "size data:"<<datas.size() <<" result:"<< results.size() <<"  detect:"<<detections.size() << std::endl;
			if (FeatureTensor::getInstance()->getRectsFeature(datas[0].img, detections))
			{
				std::vector<Track> delete_track;
				mytracker->predict();
				mytracker->update(detections, delete_track);
				
				std::string save_dt;
				std::string time = get_time(save_dt);
				
				std::vector<YourStruct1> new_tracks;
				for (Track& track : mytracker->tracks) {
					if (track.is_new_track() && track.time_since_update <= 1) {
						DETECTBOX tmp = track.current_box;
						cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
						YourStruct1 youS;
						//youS.head = get_zoom_rect(datas[0].img, rect, 1.5f);
						youS.img = datas[0].img;
						youS.rect = rect;
						youS.direct = rect_distance_points(rect);
						youS.head_id = track.track_id;
						youS.time = time;
						new_tracks.push_back(youS);

					}
					//std::cout << "track time_since_update:" << track.time_since_update <<" _max_age:"<<track._max_age<<" age:"<<track.age<<" hint:"<<track.hits<< std::endl;
				}
				
				//更新face
				for (Track& track : mytracker->tracks) {
					if (!track.is_confirmed() || track.time_since_update > 1) continue;
					DETECTBOX tmp = track.current_box;
					cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
					cv::Mat face = square_rect_Mat(datas[0].img, rect);
					Pose pose = headPose->Detect(face);
					//std::cout<<"pitch: "<<std::abs(pose.pitch)<<":"<<std::abs(pose.yaw)<<":"<<std::abs(pose.roll)<<std::endl;
					if (std::abs(pose.pitch) < max_angle_pitch && std::abs(pose.yaw) < max_angle_yaw && std::abs(pose.roll) < max_angle_roll){
						if (track.bestFace.empty()){
							track.bestFace = get_zoom_rect(datas[0].img, rect, best_face_zoom);
							track.pose_roll = pose.roll;
							track.pose_yaw = pose.yaw;
							track.pose_pitch = pose.pitch;
						}else{
							cv::Mat n_face = get_zoom_rect(datas[0].img, rect, best_face_zoom);
							float old_pose = std::abs(track.pose_pitch)+std::abs(track.pose_roll)+std::abs(track.pose_yaw);
							float new_pose = std::abs(pose.pitch)+std::abs(pose.yaw)+std::abs(pose.roll);
							if (new_pose < old_pose){
								if ((n_face.cols*n_face.rows) > (track.bestFace.cols*track.bestFace.rows / 2))
								{
									track.bestFace = n_face;
								}else if (new_pose < (old_pose/2)){
									track.bestFace = n_face;
								}
							}
						}
					}
				}
				
				if (new_tracks.size() > 0) {
					//std::cout << "new_track track face :" << new_track.size() << std::endl;
					if (track_start_cb != nullptr) {
						track_start_cb(this, new_tracks);
					}
					new_tracks.clear();
				}
				if (delete_track.size() > 0) {
					if (track_end_cb != nullptr) {
						std::vector<YourStruct2> end_tracks;
						for (size_t j = 0; j < delete_track.size(); j++)
						{
							YourStruct2 youS2;
							DETECTBOX tmp = delete_track[j].current_box;
							cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
							youS2.direct = rect_distance_points(rect);
							youS2.head_id = delete_track[j].track_id;
							//youS2.time = time;
							youS2.time = save_dt;
							youS2.face = delete_track[j].bestFace;
							end_tracks.push_back(youS2);
						}
						track_end_cb(this, end_tracks);
					}
				}
				//auto end = std::chrono::system_clock::now();
				//std::cout << "track time is:" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
				if (show) {
					for (size_t j = 0; j < points.size(); j++)
					{
						if (directs[j] == DIRECT_IN) {
							cv::line(datas[0].img, points[j], points[(j + 1) % points.size()], cv::Scalar(255, 0, 0), 2);
						}
						else if (directs[j] == DIRECT_OUT) {
							cv::line(datas[0].img, points[j], points[(j + 1) % points.size()], cv::Scalar(0, 0, 255), 2);
						}
						
					}
					std::vector<RESULT_DATA> result;
					
					for (Track& track : mytracker->tracks) {
						if (!track.is_confirmed() || track.time_since_update > 1) continue;
						result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
					}
					for (unsigned int kk = 0; kk < result.size(); kk++)
					{
						DETECTBOX tmp = result[kk].second;
						cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));

						rectangle(datas[0].img, rect, cv::Scalar(255, 255, 0), 2);
						std::string label = cv::format("%d", result[kk].first);
						cv::putText(datas[0].img, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
					}
				}
			}
			else
			{
				std::cout << "get feature failed!" << std::endl;;
			}
			if (show) {
				cv::resize(datas[0].img, datas[0].img, cv::Size(screen_width, screen_height));
				cv::putText(datas[0].img, "in:" + std::to_string(mytracker->track_count_in[0]), cv::Point(100, 100), 1, 1.5, cv::Scalar(0, 0, 255));
				cv::putText(datas[0].img, "out:" + std::to_string(mytracker->track_count_out[0]), cv::Point(100, 200), 1, 1.5, cv::Scalar(0, 0, 255));
				cv::putText(datas[0].img, "in-out:" + std::to_string(std::abs(mytracker->track_count_in[0] - mytracker->track_count_out[0])), cv::Point(100, 300), 1, 1.5, cv::Scalar(0, 0, 255));
				cv::imshow("video", datas[0].img);
				cv::waitKey(1);
			}
			datas.clear();
		}
	}
	std::cout << "end track thread" << std::endl;
}

std::vector<Track> TrackTask::get_track_queue() {
	std::vector<Track> result;
	for (Track& track : mytracker->tracks) {
		if (!track.is_confirmed() || track.time_since_update > 1) continue;
		result.push_back(track);
	}
	return result;
}

static float segmentSDF(float x, float y, float ax, float ay, float bx, float by) {
	float vx = x - ax, vy = y - ay, ux = bx - ax, uy = by - ay;
	float t = fmaxf(fminf((vx * ux + vy * uy) / (ux * ux + uy * uy), 1.0f), 0.0f);
	float dx = vx - ux * t, dy = vy - uy * t;
	return (dx * dx + dy * dy);
}
int TrackTask::rect_distance_points(cv::Rect rect) {
	int center_x = rect.x + rect.width / 2;
	int cneter_y = rect.y + rect.height / 2;
	float min_value = INFINITY;
	int min_index = 0;
	int length = points.size();
	if (length < 3) {
		return -1;
	}
	for (size_t i = 0; i < length; i++)
	{
		float dis = segmentSDF(center_x, cneter_y, points[i].x, points[i].y, points[(i + 1) % length].x, points[(i + 1) % length].y);
		if (dis < min_value) {
			min_value = dis;
			min_index = i;
		}
	}
	return directs[min_index];
}

bool TrackTask::rect_in_points(cv::Rect rect) {
	int center_x = rect.x + rect.width / 2;
	int center_y = rect.y + rect.height / 2;
	if (cv::pointPolygonTest(points, cv::Point(center_x, center_y), false) < 0) {
		return false;
	}
	return true;
}

void TrackTask::add_data_queue(uchar * bgr_data, int width, int height, bool wait) {
	cv::Mat bgr_img;
	bgr_img.create(height, width, CV_8UC3);
	memcpy(bgr_img.data, bgr_data, sizeof(uchar)*width*height * 3);
	dataque.add2queue(bgr_img, nullptr, wait);
}
void TrackTask::add_data_queue(cv::Mat img, bool wait) {
	cv::Mat data = img.clone();
	dataque.add2queue(data, nullptr, wait);
}

void TrackTask::start() {
	stop_ai_magic = false;
	detectThread = std::make_shared<std::thread>(&TrackTask::ai_magic_thread, this);
	trackThread = std::make_shared<std::thread>(&TrackTask::ai_track_thread, this);
}

void TrackTask::stop() {
	stop_ai_magic = true;
	dataque.stop_clear_queue(nullptr);
	obj_queue.stop_clear_queue(nullptr);
	if (detectThread->joinable()) {
		detectThread->join();
	}
	if (trackThread->joinable()) {
		trackThread->join();
	}
}

void TrackTask::auto_stop() {
	stop_ai_magic = true;
	dataque.just_stop_queue();
	obj_queue.just_stop_queue();
	if (detectThread->joinable()) {
		detectThread->join();
	}
	if (trackThread->joinable()) {
		trackThread->join();
	}
}
