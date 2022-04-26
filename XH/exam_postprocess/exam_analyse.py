

class ACT():
    idle = 0
    write = 1
    vacant = 2
    lpeep = 3
    rpeep = 4
    bpeep = 5
    signal = 6
    handsdown = 7
    handspeep = 8
    handpeep = 9
    getitem = 10
    changeitem = 11
    opitem = 12
    sleep = 13
    standup = 14
    handsup = 15
    drinkwater = 16
    destroypaper = 17
    turnpaper = 18
    stretch = 19

class ExamAnalyse():
    def __init__(self):
        super(ExamAnalyse, self).__init__()
        self.filter_windows = 3
        self.spend_time = 8
        self.continue_time = 4
        self.class_number = 20
        self.abnormal = [False, False, False, True, True, True, True, False, True, True,
                                 True, True, True, False, False, False, False, True, True, False]



    def filter(self, classifyes, windows_size):
        r_classifyes = []

        padding = windows_size // 2
        for i in range(len(classifyes)):
            if (i < padding or i >= len(classifyes)-padding):
                r_classifyes.append(classifyes[i])
            else:
                number = [0]*self.class_number
                for j in range(windows_size):
                    if (classifyes[i-padding+j] < 0) or (classifyes[i - padding + j] > (self.class_number-1)):
                        continue
                    if (j == padding):
                        number[classifyes[i-padding+j]] += 1.5
                    else:
                        number[classifyes[i - padding + j]] += 1.0
                max_index = number.index(max(number))
                r_classifyes.append(max_index)
        return r_classifyes

    def found_continue_classify(self, classifyes):
        ct_list = [0]*self.class_number
        bt_list = [0] *self.class_number
        last_index = -1
        for i in range(len(classifyes)):
            if classifyes[i] < 0 or classifyes[i] > self.class_number-1:
                continue
            current_index = classifyes[i]
            if (current_index != last_index):
                if last_index != -1:
                    ct_list[last_index] = 0
                ct_list[last_index] = 1
                last_index = current_index
            else:
                ct_list[current_index] += 1

            if ct_list[current_index] > bt_list[current_index]:
                bt_list[current_index] = ct_list[current_index]

        for i in range(self.class_number):
            if self.abnormal[i] and bt_list[i] > self.continue_time:
                return True
        return False

    def analyse_spend(self, classifyes):
        detect_class_num = [0]*self.class_number
        for i in range(len(classifyes)):
            if classifyes[i] < 0 or classifyes[i] > self.class_number-1:
                continue
            detect_class_num[classifyes[i]] += 1

        #策略 1 某一行为持续一定数量
        for i in range(self.class_number):
            if self.abnormal[i] and self.spend_time < detect_class_num[i]:
                return True
        #策略 2 左右看+向后看持续一定数量
        if (detect_class_num[ACT.lpeep] + detect_class_num[ACT.rpeep] + detect_class_num[ACT.bpeep]) > (1.5*self.spend_time):
            return True

        #策略 3 向下偷看 + 手放下持续一定数量
        if (detect_class_num[ACT.handpeep] > self.spend_time/2 or detect_class_num[ACT.handspeep] > self.spend_time / 2)  and (detect_class_num[ACT.handpeep] + detect_class_num[ACT.handspeep] + detect_class_num[ACT.handsdown]) > (1.5 *self.spend_time):
            return True

        return False

    def analyse_continue(self, classifyes):
        status1 = self.found_continue_classify(classifyes)
        if status1:
            return status1
        r_classifyes = self.filter(classifyes, self.filter_windows)
        status2 = self.found_continue_classify(r_classifyes)
        if status2:
            return status2

        rr_classifyes = self.filter(r_classifyes, self.filter_windows)
        status3 = self.found_continue_classify(rr_classifyes)
        return status3

    def set_continue_time(self, continue_t):
        self.continue_time = continue_t
    def set_spend_time(self, spend_t):
        self.spend_time = spend_t

    def analyse(self, classifyes):
        status1 = self.analyse_spend(classifyes)
        if status1:
            return status1
        statu2 = self.analyse_continue(classifyes)
        return statu2