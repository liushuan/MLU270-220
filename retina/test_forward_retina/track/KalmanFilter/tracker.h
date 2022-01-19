#ifndef TRACKER_H
#define TRACKER_H
#include <vector>


#include "kalmanfilter.h"
#include "track.h"
#include "../DeepAppearanceDescriptor/model.h"

#define TRACK_COUNT 1


class NearNeighborDisMetric;

class tracker
{
public:
	NearNeighborDisMetric* metric;
	float max_iou_distance;
	int max_age;
	int n_init;

	KalmanFilter* kf;

	int _next_idx;
	int max_size = 100000;

	long track_count_in[TRACK_COUNT] = { 0 };
	long track_count_out[TRACK_COUNT] = { 0 };
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
	std::vector<Track, Eigen::aligned_allocator<Track>> tracks;
	tracker(/*NearNeighborDisMetric* metric,*/
		float max_cosine_distance, int nn_budget,
		float max_iou_distance = 0.7,
		int max_age = 30, int n_init = 3);
	void set_max_size_id(int size_) {
		max_size = size_;
	}
	void predict();
	void update(const DETECTIONS& detections, std::vector<Track>&delete_t);
	typedef DYNAMICM(tracker::* GATED_METRIC_FUNC)(
		std::vector<Track, Eigen::aligned_allocator<Track>>& tracks,
		const DETECTIONS& dets,
		const std::vector<int>& track_indices,
		const std::vector<int>& detection_indices);
	bool delete_track(int track_id);
private:
	void _match(const DETECTIONS& detections, TRACHER_MATCHD& res);
	void _initiate_track(const DETECTION_ROW& detection);
public:
	DYNAMICM gated_matric(
		std::vector<Track, Eigen::aligned_allocator<Track>>& tracks,
		const DETECTIONS& dets,
		const std::vector<int>& track_indices,
		const std::vector<int>& detection_indices);
	DYNAMICM iou_cost(
		std::vector<Track, Eigen::aligned_allocator<Track>>& tracks,
		const DETECTIONS& dets,
		const std::vector<int>& track_indices,
		const std::vector<int>& detection_indices);
	Eigen::VectorXf iou(DETECTBOX& bbox,
		DETECTBOXSS &candidates);

	bool direct_in(Track &t);
	float beta = 0.9f;
	int count = 0;
	float dir_x = 0;
	float dir_y = 0;

	void set_direct(float dirx, float diry) {
		dir_x = dirx;
		dir_y = diry;
	}

};

#endif // TRACKER_H
