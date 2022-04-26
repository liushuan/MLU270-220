#ifndef _EXAM_ANALYSE_H_
#define _EXAM_ANALYSE_H_

#include <vector>

struct ContinueT
{
	int ct = 0;
	int bt = 0;
};

class ExamAnalyse
{
	
private:
	
	enum {
		idle = 0, write, vacant, lpeep, rpeep, bpeep, signal, handsdown, handspeep, handpeep,
		getitem, changeitem, opitem, sleep, standup, handsup, drinkwater, destroypaper, turnpaper, stretch
	};
	int class_number = stretch+1;
	bool abnormal[stretch+1] = {false, false, false, true, true, true, true, false, true, true,
	true, true, true, false, false, false, false, true, true, false};

	int filter_windows = 3;
	int spend_time = 6;
	int continue_time = 4;
	void filter(const std::vector<int> &classifyes, std::vector<int> &r_classifyes, int windows_size);
	bool found_continue_classify(const std::vector<int> &classifyes);
	

	bool analyse_spend(const std::vector<int> &classifyes);
	bool analyse_continue(const std::vector<int> &classifyes);

public:

	void set_continue_time(int continue_t);
	void set_spend_time(int spend_t);
	bool analyse(const std::vector<int> &classifyes);

};






#endif