#ifndef _TEST_PR_H
#define _TEST_PR_H

#include <iostream>
#include <vector>
#include <string>

#ifdef _WIN32
#include <io.h>
#include <direct.h>
#else
#include <unistd.h>
#include <dirent.h>
#endif

#include <fstream>
#include <istream>
#include <stdlib.h>



#include<opencv2/opencv.hpp>
using namespace std;


#define CLASSNUM 24
static std::vector<std::string> OBJ_LABELS = {"idle","write","vacant","lpeep","rpeep","bpeep","signal","handsdown","handspeep","handpeep",
                  "getitem","changeitem","opitem","sleep","standup","handsup","drinkwater","destroypaper","turnpaper","stretch",
                  "teacher_normal","teacher_smoke","teacher_book","teacher_dozing"};
float PR[200][CLASSNUM][3] = { 0.0f };
float RRR[200][CLASSNUM][3] = { 0.0f };

namespace util {

	enum POS {
		FRONT,
		BACK
	};
	struct Obj {
		float xmin;
		float ymin;
		float xmax;
		float ymax;
		int classify;
		float score;
		int position = FRONT;
		bool match = false;
	};
	struct Target {
		std::vector<Obj> objs;
		std::string img_name;
		std::string txt_name;
	};
	struct LineParam
	{
		float k;
		float b;
	};

	template <typename Type >
	static Type stringToNum(const string& str)
	{
		istringstream iss(str);
		Type num;
		iss >> num;
		return num;
	}
	static vector<string> split_string(const string& s, const string& c)
	{
		vector<string> v;
		string::size_type pos1, pos2;
		pos2 = s.find(c);
		pos1 = 0;
		while (string::npos != pos2)
		{
			v.push_back(s.substr(pos1, pos2 - pos1));
			pos1 = pos2 + c.size();
			pos2 = s.find(c, pos1);
		}
		if (pos1 != s.length()) {
			v.push_back(s.substr(pos1));
		}
		return v;
	}



	static std::vector<string> readStringFromFileData(string filePath)
	{
		std::vector<string> data;
		ifstream fileA(filePath);
		if (!fileA)
		{
			cout << "没有找到需要读取的  " << filePath << " 请将文件放到指定位置再次运行本程序。" << endl << "  按任意键以退出";
			return data;
		}
		for (int i = 0; !fileA.eof(); i++)
		{
			string buf;
			getline(fileA, buf, '\n');

			/*if (buf == "")
			{
			cout << "buf is empty." << endl;
			continue;
			}*/
			data.push_back(buf);
		}
		fileA.close();
		return data;
	}
#ifdef _WIN32
	static void getFiles(string path, vector<string>& files, vector<string>& filenames, const string & tail)
	{
		//文件句柄  
		intptr_t hFile = 0;
		//文件信息  
		struct _finddata_t fileinfo;
		string p;
		if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		{
			do
			{
				//如果是目录,迭代之  
				//如果不是,加入列表  
				if ((fileinfo.attrib &  _A_SUBDIR))
				{
					if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					{
						getFiles(p.assign(path).append("\\").append(fileinfo.name), files, filenames, tail);
					}

				}
				else
				{
					string filename = fileinfo.name;
					size_t found = filename.find(tail);
					if (found != string::npos)
					{
						files.push_back(p.assign(path).append("\\").append(filename));
						filenames.push_back(filename);
					}
				}
			} while (_findnext(hFile, &fileinfo) == 0);

			_findclose(hFile);
		}
	}
#else
	
	static void getFiles(std::string cate_dir, std::vector<std::string> &filess, std::vector<std::string> &file_names, const string & tail)
	{
		DIR *dir;
		struct dirent *ptr;
		if ((dir = opendir(cate_dir.c_str())) == NULL)
		{
			perror("Open dir error...");
			return;
		}

		while ((ptr = readdir(dir)) != NULL)
		{
			if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) {    ///current dir OR parrent dir
				continue;
			}
			else if (ptr->d_type == 8) {    ///file
											//printf("d_name:%s/%s\n",basePath,ptr->d_name);
				std::string file_name = ptr->d_name;
				size_t found = file_name.find(tail);
				if (found != string::npos)
				{
					file_names.push_back(file_name);
					filess.push_back(cate_dir + "/" + file_name);
				}
			}
			else if (ptr->d_type == 10) {    ///link file
											 //printf("d_name:%s/%s\n",basePath,ptr->d_name);
				continue;
			}
			else if (ptr->d_type == 4)    ///dir
			{
				std::string file_name = ptr->d_name;
				size_t found = file_name.find(tail);
				if (found != string::npos)
				{
					file_names.push_back(file_name);
					filess.push_back(cate_dir + "/" + file_name);
				}
			}
		}
		closedir(dir);
	}
#endif

	void getLinePara(float x1, float y1, float x2, float y2, LineParam & LP)
	{
		double m = 0;
		// 计算分子  
		m = x2 - x1;
		if (0 == m)
		{
			LP.k = 100000000.0f;
			LP.b = y1 - LP.k * x1;
		}
		else
		{
			LP.k = (y2 - y1) / (x2 - x1);
			LP.b = y1 - LP.k * x1;
		}
	}

	void split_points(const std::vector<cv::Point2f>&points, const LineParam &lp, std::vector<int> &sp_upper, std::vector<int> &sp_lower) {
		for (size_t i = 0; i < points.size(); i++)
		{
			if (points[i].y >= (lp.k*points[i].x + lp.b)) {
				sp_lower.push_back(i);
			}
			else {
				sp_upper.push_back(i);
			}
		}
	}

	void set_position(Target &t) {
		std::vector<cv::Point2f> cps;
		for (size_t i = 0; i < t.objs.size(); i++)
		{
			float cx = (t.objs[i].xmin + t.objs[i].xmax) / 2.0f;
			float cy = (t.objs[i].ymin + t.objs[i].ymax) / 2.0f;
			cps.push_back(cv::Point2f(cx, cy));
		}
		cv::RotatedRect rr = cv::minAreaRect(cps);
		//绘制拟合椭圆
		cv::Point2f vertices[4];
		rr.points(vertices);
		std::vector<cv::Point2f> cornerp = { vertices[0],vertices[1], vertices[2], vertices[3] };

		std::sort(cornerp.begin(), cornerp.end(),
			[&](const cv::Point2f& a, const cv::Point2f& b) {
			return (a.y > b.y);
		}
		);
		//std::cout << "y:" << cornerp[0].y << " " << cornerp[1].y << " " << cornerp[2].y << " " << cornerp[3].y << std::endl;
		float x1 = cornerp[0].x + (cornerp[2].x - cornerp[0].x)*2.0 / 3.0f;
		float y1 = cornerp[0].y + (cornerp[2].y - cornerp[0].y)*2.0 / 3.0f;
		float x2 = cornerp[1].x + (cornerp[3].x - cornerp[1].x)*2.0 / 3.0f;
		float y2 = cornerp[1].y + (cornerp[3].y - cornerp[1].y)*2.0 / 3.0f;
		LineParam line;
		getLinePara(x1, y1, x2, y2, line);
		std::vector<int> back_index, front_index;
		split_points(cps, line, back_index, front_index);
		for (size_t i = 0; i < back_index.size(); i++)
		{
			t.objs[back_index[i]].position = BACK;
		}
		for (size_t i = 0; i < front_index.size(); i++)
		{
			t.objs[front_index[i]].position = FRONT;
		}
		bool show = false;
		if (show) {
			cv::Mat img = cv::imread(t.img_name);
			cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 4);

			for (size_t i = 0; i < back_index.size(); i++)
			{
				cv::circle(img, cv::Point(t.objs[back_index[i]].xmin, t.objs[back_index[i]].ymin), 5, cv::Scalar(255, 0, 255), -1);
			}

			for (size_t i = 0; i < front_index.size(); i++)
			{
				cv::circle(img, cv::Point(t.objs[front_index[i]].xmin, t.objs[front_index[i]].ymin), 5, cv::Scalar(0, 0, 255), -1);
			}
			cv::imshow("img", img);
			cv::waitKey(0);
		}
	}

	std::vector<Target> get_targets(std::string txt_path, std::string img_path) {
		std::vector<Target>targets;
		std::vector<std::string> files, file_names;
		getFiles(txt_path, files, file_names, ".txt");
		for (size_t i = 0; i < files.size(); i++)
		{
			Target t;
			t.txt_name = files[i];
			t.img_name = img_path + "/" + file_names[i].replace(file_names[i].find(".txt"), 4, ".jpg");
			std::vector<std::string> lines = readStringFromFileData(t.txt_name);
			cv::Mat img = cv::imread(t.img_name);

			for (size_t i = 0; i < lines.size(); i++)
			{
				std::vector<std::string> strs = split_string(lines[i], " ");
				if (strs.size() == 5) {
					Obj obj;
					obj.classify = stringToNum<int>(strs[0]);
					obj.xmin = (stringToNum<float>(strs[1]) - stringToNum<float>(strs[3]) / 2.0f) *img.cols;
					obj.ymin = (stringToNum<float>(strs[2]) - stringToNum<float>(strs[4]) / 2.0f) *img.rows;
					obj.xmax = (stringToNum<float>(strs[1]) + stringToNum<float>(strs[3]) / 2.0f) *img.cols;
					obj.ymax = (stringToNum<float>(strs[2]) + stringToNum<float>(strs[4]) / 2.0f) *img.rows;
					obj.score = 1.1f;
					obj.match = false;
					t.objs.push_back(obj);
				}
				else if (strs.size() > 1)
				{
					std::cout << "txt style error. file is:" << t.txt_name << std::endl;
				}
			}
			targets.push_back(t);

			if (i % 50 == 0) {
				std::cout << "get file number is:" << i << std::endl;
			}
		}
		//获得前后
		for (size_t i = 0; i < targets.size(); i++)
		{
			set_position(targets[i]);
		}
		return targets;
	}

	float euclidean(float x1, float y1, float x2, float y2) {
		float d1 = x1 - x2;
		float d2 = y1 - y2;
		return std::sqrt(d1 * d1 + d2 * d2);
	}
	void get_preds_position(std::vector<Target> & preds, std::vector<Target> &targets) {
		for (size_t i = 0; i < preds.size(); i++)
		{
			for (size_t j = 0; j < preds[i].objs.size(); j++)
			{
				Obj &b1 = preds[i].objs[j];
				float min_distance = INFINITY;
				int min_index_k = -1;
				for (size_t k = 0; k < targets[i].objs.size(); k++)
				{
					Obj &b2 = targets[i].objs[k];
					float dis = euclidean((b1.xmin + b1.xmax) / 2.0f, (b1.ymin + b1.ymax) / 2.0f, (b2.xmin + b2.xmax) / 2.0f, (b2.ymin + b2.ymax) / 2.0f);
					if (dis < min_distance) {
						min_distance = dis;
						min_index_k = k;
					}
				}
				if (min_index_k > -1) {
					preds[i].objs[j].position = targets[i].objs[min_index_k].position;
				}
				else {
					preds[i].objs[j].position = FRONT;
				}
			}
		}
	}


	float iou(float x1, float y1, float x2, float y2, float tx1, float ty1, float tx2, float ty2) {

		double area1 = (x2 - x1)*(y2 - y1);
		double area2 = (tx2 - tx1)*(ty2 - ty1);

		double xx1 = x1 > tx1 ? x1 : tx1;
		double yy1 = y1 > ty1 ? y1 : ty1;

		double xx2 = (x2) < (tx2) ? (x2) : (tx2);
		double yy2 = (y2) < (ty2) ? (y2) : (ty2);

		double w = (xx2 - xx1 + 0.00001) > 0.0 ? (xx2 - xx1 + 0.00001) : 0.0;
		double h = (yy2 - yy1 + 0.00001) > 0.0 ? (yy2 - yy1 + 0.00001) : 0.0;

		double inter = w * h;
		double ovr = inter / (area1 + area2 - inter);
		return ovr;
	}


	struct PRAP {
		int fronts_pr[CLASSNUM][3] = { 0 };
		int backs_pr[CLASSNUM][3] = { 0 };
		float map_fronts[CLASSNUM] = { 0 };
		float map_backs[CLASSNUM] = { 0 };
	};


	static void writeStringtoFile(std::string filePath, vector<string> & data)
	{
		if (filePath == "" || data.size() == 0)
		{
			cout << " filePath == "" || data.size() == 0 " << endl;
			return;
		}
		ofstream in;
		in.open(filePath, ios::app);
		int length = data.size();
		for (int i = 0; i < length; i++)
		{
			in << data[i] << "\n";
		}
		in.close();
	}


	std::vector<std::vector<float>> get_best_threhold(std::vector<Target> & preds, std::vector<Target> &targets, float iou_t, const std::vector<std::string>& OBJ_LABELS) {
		std::vector<std::vector<float>>maps;
		//1.p  2.r  3.ap  4.前后
		float opt_threhold[CLASSNUM][3] = { 0.0f };
		float max_f1score[CLASSNUM][3] = { 0.0f };

		float fpr[CLASSNUM] = { 0.0f };
		float bpr[CLASSNUM] = { 0.0f };
		float apr[CLASSNUM] = { 0.0f };

		for (size_t n = 1; n < 201; n++)
		{
			float score_threhold = n / 200.0f;

			PRAP prap;
			for (size_t i = 0; i < targets.size(); i++)
			{
				for (size_t j = 0; j < targets[i].objs.size(); j++)
				{
					Obj &b1 = targets[i].objs[j];
					float max_iou = 0.0f;
					int max_index_k = -1;
					for (size_t k = 0; k < preds[i].objs.size(); k++)
					{
						Obj &b2 = preds[i].objs[k];
						if (b1.classify == b2.classify && !b2.match && (b2.score >= score_threhold)) {
							float viou = iou(b1.xmin, b1.ymin, b1.xmax, b1.ymax, b2.xmin, b2.ymin, b2.xmax, b2.ymax);
							if (viou >= iou_t) {
								if (max_iou < viou) {
									max_iou = viou;
									max_index_k = k;
								}
							}
						}
					}
					if (max_index_k != -1) {
						preds[i].objs[max_index_k].match = true;
						if (b1.position == FRONT) {
							prap.fronts_pr[b1.classify][0]++;
						}
						else {
							prap.backs_pr[b1.classify][0]++;
						}

					}
					if (b1.position == FRONT) {
						prap.fronts_pr[b1.classify][2]++;
					}
					else {
						prap.backs_pr[b1.classify][2]++;
					}
				}

				for (size_t k = 0; k < preds[i].objs.size(); k++)
				{
					Obj &b2 = preds[i].objs[k];
					if (!b2.match && (b2.score >= score_threhold)) {
						if (b2.position == FRONT) {
							prap.fronts_pr[b2.classify][1]++;
						}
						else {
							prap.backs_pr[b2.classify][1]++;
						}
					}

					//reset all match.
					b2.match = false;
				}
			}
			//计算PR
			for (size_t i = 0; i < CLASSNUM; i++)
			{
				float front_p = 0;
				if (((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1]) != 0) {
					front_p = (float)prap.fronts_pr[i][0] / ((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1]);
				}

				float back_p = 0;
				if (((float)prap.backs_pr[i][0] + prap.backs_pr[i][1]) != 0) {
					back_p = (float)prap.backs_pr[i][0] / ((float)prap.backs_pr[i][0] + prap.backs_pr[i][1]);
				}

				float all_p = 0;
				if (((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1] + (float)prap.backs_pr[i][0] + prap.backs_pr[i][1]) != 0) {
					all_p = ((float)prap.fronts_pr[i][0] + (float)prap.backs_pr[i][0]) / ((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1] + (float)prap.backs_pr[i][0] + prap.backs_pr[i][1]);
				}



				float front_r = 0;
				if ((float)prap.fronts_pr[i][2] != 0) {
					front_r = (float)prap.fronts_pr[i][0] / (float)prap.fronts_pr[i][2];

				}
				float back_r = 0;
				if ((float)prap.backs_pr[i][2] != 0) {
					back_r = (float)prap.backs_pr[i][0] / (float)prap.backs_pr[i][2];

				}
				float all_r = 0;
				if (((float)prap.fronts_pr[i][2] + (float)prap.backs_pr[i][2]) != 0) {
					all_r = ((float)prap.fronts_pr[i][0] + (float)prap.backs_pr[i][0]) / ((float)prap.fronts_pr[i][2] + (float)prap.backs_pr[i][2]);

				}
				PR[n-1][i][0] = front_p;
				PR[n-1][i][1] = back_p;
				PR[n-1][i][2] = all_p;
				RRR[n-1][i][0] = front_r;
				RRR[n-1][i][1] = back_r;
				RRR[n-1][i][2] = all_r;
				float f1_score_front = 0.0f;
				if ((front_p + front_r) != 0) {
					f1_score_front = 2 * front_p*front_r / (front_p + front_r);
				}
				float f1_score_back = 0.0f;
				if ((back_p + back_r) != 0) {
					f1_score_back = 2 * back_p * back_r / (back_p + back_r);
				}
				float f1_score_all = 0.0f;
				if ((all_p + all_r) != 0) {
					f1_score_all = 2 * all_p *all_r / (all_p + all_r);
				}

				if (max_f1score[i][0] < f1_score_front) {
					max_f1score[i][0] = f1_score_front;
					opt_threhold[i][0] = score_threhold;
				}
				if (max_f1score[i][1] < f1_score_back) {
					max_f1score[i][1] = f1_score_back;
					opt_threhold[i][1] = score_threhold;
				}
				if (max_f1score[i][2] < f1_score_all) {
					max_f1score[i][2] = f1_score_all;
					opt_threhold[i][2] = score_threhold;
				}
			}
		}
		for (size_t n = 0; n < 200; n++)
		{
			for (size_t i = 0; i < CLASSNUM; i++)
			{
				if (n > 0) {
					fpr[i] += (RRR[n - 1][i][0] - RRR[n][i][0])*((PR[n - 1][i][0] + PR[n][i][0]) / 2);
					bpr[i] += (RRR[n - 1][i][1] - RRR[n][i][1])*((PR[n - 1][i][1] + PR[n][i][1]) / 2);
					apr[i] += (RRR[n - 1][i][2] - RRR[n][i][2])*((PR[n - 1][i][2] + PR[n][i][2]) / 2);
					if ((RRR[n - 1][i][0] - RRR[n][i][0]) < 0 || (RRR[n - 1][i][1] - RRR[n][i][1]) < 0 || (RRR[n - 1][i][2] - RRR[n][i][2]) < 0) {
						std::cout << "no value :" << (RRR[n - 1][i][0] - RRR[n][i][0]) << " " << (RRR[n - 1][i][1] - RRR[n][i][1]) << " " << (RRR[n - 1][i][2] - RRR[n][i][2]) << std::endl;
					}
				}
			}
		}

		std::vector<float> a1, a2, a3;
		for (size_t i = 0; i < CLASSNUM; i++)
		{
			//std::cout << "f:" << fpr[i] << " " << bpr[i] << " " << apr[i] << std::endl;
			float fp = fpr[i];
			float bp = bpr[i];
			float ap = apr[i];
			a1.push_back(fp);
			a2.push_back(bp);
			a3.push_back(ap);
		}
		maps.push_back(a1);
		maps.push_back(a2);
		maps.push_back(a3);

		std::vector<std::string> best_threholds;
		best_threholds.push_back("\n");
		best_threholds.push_back("index,classify,front_threhold,maxfront_f1_score,back_threhold,maxback_f1_score, all_t, maxall_f1_score");
		for (size_t i = 0; i < CLASSNUM; i++)
		{
			std::string line = std::to_string(i) + "," + OBJ_LABELS[i] + "," + std::to_string(opt_threhold[i][0]) + "," + std::to_string(max_f1score[i][0])
				+ "," + std::to_string(opt_threhold[i][1]) + "," + std::to_string(max_f1score[i][1])
				+ "," + std::to_string(opt_threhold[i][2]) + "," + std::to_string(max_f1score[i][2]);
			best_threholds.push_back(line);
		}
		writeStringtoFile("best_threhold.csv", best_threholds);
		std::cout << "get best threhold finished." << std::endl;
		return maps;
	}



	void test_PR(std::vector<Target> & preds, std::vector<Target> &targets) {

		get_preds_position(preds, targets);
		float iou_t = 0.5f;

		std::vector<std::vector<float>>maps = get_best_threhold(preds, targets, iou_t, OBJ_LABELS);


		//正常保存 //统计TP FP, GT
		float score_threhold = 0.3f;
		PRAP prap;
		for (size_t i = 0; i < targets.size(); i++)
		{
			for (size_t j = 0; j < targets[i].objs.size(); j++)
			{
				Obj &b1 = targets[i].objs[j];
				float max_iou = 0.0f;
				int max_index_k = -1;
				for (size_t k = 0; k < preds[i].objs.size(); k++)
				{
					Obj &b2 = preds[i].objs[k];
					if (b1.classify == b2.classify && !b2.match && (b2.score >= score_threhold)) {
						float viou = iou(b1.xmin, b1.ymin, b1.xmax, b1.ymax, b2.xmin, b2.ymin, b2.xmax, b2.ymax);
						if (viou >= iou_t) {
							if (max_iou < viou) {
								max_iou = viou;
								max_index_k = k;
							}
						}
					}
				}
				if (max_index_k != -1) {
					preds[i].objs[max_index_k].match = true;
					if (b1.position == FRONT) {
						prap.fronts_pr[b1.classify][0]++;
					}
					else {
						prap.backs_pr[b1.classify][0]++;
					}

				}
				if (b1.position == FRONT) {
					prap.fronts_pr[b1.classify][2]++;
				}
				else {
					prap.backs_pr[b1.classify][2]++;
				}
			}

			for (size_t k = 0; k < preds[i].objs.size(); k++)
			{
				Obj &b2 = preds[i].objs[k];
				if (!b2.match && (b2.score >= score_threhold)) {
					if (b2.position == FRONT) {
						prap.fronts_pr[b2.classify][1]++;
					}
					else {
						prap.backs_pr[b2.classify][1]++;
					}
				}
				//reset all match.
				b2.match = false;
			}
		}
		std::cout << "start count PR" << std::endl;
		//计算 P R
		std::vector<std::string> pr_lines;
		pr_lines.push_back("\n");
		pr_lines.push_back("index,classify,numberfront,frontP,frontR,frontMap,frontF1Score,numberback, backP,backR,backMap,backF1Score,numberobj,allP,allR,allMap,allF1Score");
		float all_front[3] = { 0.0f };
		float all_back[3] = { 0.0f };
		float all_all[3] = { 0.0f };
		for (size_t i = 0; i < CLASSNUM; i++)
		{
			float front_p = 0;
			if (((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1]) != 0) {
				front_p = (float)prap.fronts_pr[i][0] / ((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1]);
			}
			float back_p = 0;
			if (((float)prap.backs_pr[i][0] + prap.backs_pr[i][1]) != 0) {
				back_p = (float)prap.backs_pr[i][0] / ((float)prap.backs_pr[i][0] + prap.backs_pr[i][1]);
			}
			float all_p = 0;
			if (((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1] + (float)prap.backs_pr[i][0] + prap.backs_pr[i][1]) != 0) {
				all_p = ((float)prap.fronts_pr[i][0] + (float)prap.backs_pr[i][0]) / ((float)prap.fronts_pr[i][0] + prap.fronts_pr[i][1] + (float)prap.backs_pr[i][0] + prap.backs_pr[i][1]);

			}

			float front_r = 0;
			if ((float)prap.fronts_pr[i][2] != 0) {
				front_r = (float)prap.fronts_pr[i][0] / (float)prap.fronts_pr[i][2];
			}
			float back_r = 0;
			if ((float)prap.backs_pr[i][2] != 0) {
				back_r = (float)prap.backs_pr[i][0] / (float)prap.backs_pr[i][2];
			}
			float all_r = 0;
			if (((float)prap.fronts_pr[i][2] + (float)prap.backs_pr[i][2]) != 0) {
				all_r = ((float)prap.fronts_pr[i][0] + (float)prap.backs_pr[i][0]) / ((float)prap.fronts_pr[i][2] + (float)prap.backs_pr[i][2]);
			}

			float front_f1_score = 0;
			if ((front_p + front_r) != 0) {
				front_f1_score = 2 * front_p*front_r / (front_p + front_r);
			}
			float back_f1_score = 0;
			if ((back_p + back_r) != 0) {
				back_f1_score = 2 * back_p*back_r / (back_p + back_r);
			}
			float all_f1_score = 0;
			if ((all_p + all_r) != 0) {
				all_f1_score = 2 * all_p*all_r / (all_p + all_r);
			}

			std::string line = std::to_string(i + 1) + "," + OBJ_LABELS[i]
				+ "," + std::to_string(prap.fronts_pr[i][2]) + "," + std::to_string(front_p) + "," + std::to_string(front_r) + "," + std::to_string(maps[0][i]) + "," + std::to_string(front_f1_score)
				+ "," + std::to_string(prap.backs_pr[i][2]) + "," + std::to_string(back_p) + "," + std::to_string(back_r) + "," + std::to_string(maps[1][i]) + "," + std::to_string(back_f1_score)
				+ "," + std::to_string(prap.fronts_pr[i][2] + prap.backs_pr[i][2]) + "," + std::to_string(all_p) + "," + std::to_string(all_r) + "," + std::to_string(maps[2][i]) + "," + std::to_string(all_f1_score);
			pr_lines.push_back(line);

			all_front[0] += (float)prap.fronts_pr[i][0];
			all_front[1] += (float)prap.fronts_pr[i][1];
			all_front[2] += (float)prap.fronts_pr[i][2];
			all_back[0] += (float)prap.backs_pr[i][0];
			all_back[1] += (float)prap.backs_pr[i][1];
			all_back[2] += (float)prap.backs_pr[i][2];
		}
		float all_front_p = 0.0f;
		if ((all_front[0] + all_front[1]) != 0) {
			all_front_p = all_front[0] / (all_front[0] + all_front[1]);
		}
		float all_front_r = 0.0f;
		if (all_front[2] != 0) {
			all_front_r = all_front[0] / all_front[2];
		}

		float all_back_p = 0.0f;
		if ((all_back[0] + all_back[1]) != 0) {
			all_back_p = all_back[0] / (all_back[0] + all_back[1]);
		}
		float all_back_r = 0.0f;
		if (all_back[2] != 0) {
			all_back_r = all_back[0] / all_back[2];
		}
		float all_all_p = 0;
		if ((all_front[0] + all_front[1] + all_back[0] + all_back[1]) != 0) {
			all_all_p = (all_front[0] + all_back[0]) / (all_front[0] + all_front[1] + all_back[0] + all_back[1]);
		}
		float all_all_r = 0;
		if ((all_front[2] + all_back[2]) != 0) {
			all_all_r = (all_front[0] + all_back[0]) / (all_front[2] + all_back[2]);
		}

		float all_front_f1_score = 0;
		if ((all_front_p + all_front_r) != 0) {
			all_front_f1_score = 2 * all_front_p*all_front_r / (all_front_p + all_front_r);
		}
		float all_back_f1_score = 0;
		if ((all_back_p + all_back_r) != 0) {
			all_back_f1_score = 2 * all_back_p*all_back_r / (all_back_p + all_back_r);
		}
		float all_all_f1_score = 0;
		if ((all_all_p + all_all_r) != 0) {
			all_all_f1_score = 2 * all_all_p*all_all_r / (all_all_p + all_all_r);
		}
		float all_map_front = 0.0f;
		float all_map_back = 0.0f;
		float all_all_map = 0.0f;
		std::cout << "maps size:" << maps.size() << std::endl;
		std::cout << "mpas :" << maps[0].size() << " " << maps[1].size() << " " << maps[2].size() << std::endl;
		for (size_t i = 0; i < CLASSNUM; i++)
		{

			all_map_front = all_map_front + maps[0][i];
			all_map_back = all_map_back + maps[1][i];
			all_all_map = all_all_map + maps[2][i];
		}
		all_map_front = all_map_front / CLASSNUM;
		all_map_back = all_map_back / CLASSNUM;
		all_all_map = all_all_map / CLASSNUM;

		std::string line = std::to_string(0) + ",all,"
			+ std::to_string(all_front[2]) + "," + std::to_string(all_front_p) + "," + std::to_string(all_front_r) + ',' + std::to_string(all_map_front) + ',' + std::to_string(all_front_f1_score)
			+ "," + std::to_string(all_back[2]) + "," + std::to_string(all_back_p) + "," + std::to_string(all_back_r) + ',' + std::to_string(all_map_back) + ',' + std::to_string(all_back_f1_score)
			+ "," + std::to_string(all_front[2] + all_back[2]) + "," + std::to_string(all_all_p) + "," + std::to_string(all_all_r) + ',' + std::to_string(all_all_map) + ',' + std::to_string(all_all_f1_score);
		pr_lines.insert(pr_lines.begin() + 2, line);

		writeStringtoFile("./log.csv", pr_lines);
		std::cout << "write log finished." << std::endl;
	}

}

#endif