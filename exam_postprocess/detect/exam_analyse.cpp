#include "exam_analyse.h"


bool ExamAnalyse::found_continue_classify(const std::vector<int> &classifyes) {
	std::vector<ContinueT> continueT(class_number);
	int last_index = -1;
	for (size_t i = 0; i < classifyes.size(); i++)
	{
		if (classifyes[i] < 0 || classifyes[i] > stretch) {
			continue;
		}
		int current_index = classifyes[i];
		if (current_index != last_index) {
			if (last_index != -1) {
				continueT[last_index].ct = 0;
			}
			continueT[current_index].ct = 1;
			last_index = current_index;
		}
		else {
			continueT[current_index].ct++;
		}

		if (continueT[current_index].ct > continueT[current_index].bt) {
			continueT[current_index].bt = continueT[current_index].ct;
		}
	}
	for (size_t i = 0; i < continueT.size(); i++)
	{
		if (abnormal[i] && continueT[i].bt > continue_time) {
			return true;
		}
	}
	return false;
}

static int found_max_index(float * data, int length) {
	int max_index = 0;
	float max_value = -1;
	for (size_t i = 0; i < length; i++)
	{
		if (max_value < data[i]) {
			max_value = data[i];
			max_index = i;
		}
	}
	return max_index;
}

void ExamAnalyse::filter(const std::vector<int> &classifyes, std::vector<int> &r_classifyes, int windows_size) {
	int padding = windows_size / 2;
	for (size_t i = 0; i < classifyes.size(); i++)
	{
		if (i < padding || i >= (classifyes.size()-padding)) {
			r_classifyes.push_back(classifyes[i]);
		}
		else {
			float number[stretch + 1] = {0};
			for (size_t j = 0; j < windows_size; j++)
			{
				if (classifyes[i - padding + j] < 0 || classifyes[i - padding + j] > stretch) {
					continue;
				}
				if (j == padding) {
					number[classifyes[i - padding + j]] += 1.5f;
				}
				else {
					number[classifyes[i - padding + j]] += 1.0f;
				}
			}
			int max_index = found_max_index(number, class_number);
			r_classifyes.push_back(max_index);
		}
	}
}


bool ExamAnalyse::analyse_spend(const std::vector<int> &classifyes) {
	int detect_class_num[stretch+1] = {0};
	for (size_t i = 0; i < classifyes.size(); i++)
	{
		if (classifyes[i] < 0 || classifyes[i] > stretch) {
			continue;
		}
		detect_class_num[classifyes[i]]++;
	}
	//策略 1 .某一行为持续一定数量
	for (size_t i = 0; i < class_number; i++)
	{
		if (abnormal[i] && spend_time < detect_class_num[i]) {
			return true;
		}
	}
	//策略 2 左右看+向后看持续一定数量
	if ((detect_class_num[lpeep] + detect_class_num[rpeep] + detect_class_num[bpeep]) > (1.5*spend_time)) {
		return true;
	}
	//策略 3 向下偷看+手放下持续一定数量
	if ((detect_class_num[handpeep] > spend_time / 2 || detect_class_num[handspeep] > spend_time / 2) && ((detect_class_num[handpeep] + detect_class_num[handspeep] + detect_class_num[handsdown]) > 1.5*spend_time) ) {
		return true;
	}
	return false;
}


bool ExamAnalyse::analyse_continue(const std::vector<int> &classifyes) {
	bool status1 = found_continue_classify(classifyes);
	if (status1) {
		return status1;
	}
	std::vector<int> r_classifyes;
	filter(classifyes, r_classifyes, filter_windows);
	bool status2 = found_continue_classify(r_classifyes);
	if (status2) {
		return status2;
	}
	std::vector<int> rr_classifyes;
	filter(r_classifyes, rr_classifyes, filter_windows);
	bool status3 = found_continue_classify(rr_classifyes);
	return status3;
}

void ExamAnalyse::set_continue_time(int continue_t) {
	continue_time = continue_t;
}
void ExamAnalyse::set_spend_time(int spend_t) {
	spend_time = spend_t;
}



bool ExamAnalyse::analyse(const std::vector<int> &classifyes) {

	bool status1 = analyse_spend(classifyes);
	if (status1) {
		return status1;
	}
	bool status2 = analyse_continue(classifyes);
	return status2;
}