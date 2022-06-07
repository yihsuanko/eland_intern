import csv
import re
import pandas as pd
import logging
from match import longestSubstring
import random
import json
from sklearn.model_selection import train_test_split

_date = "0324"
_open_path = "./data/ner_all_2022_v2.tsv"
_save_path = "./train_data/data_{}.json".format(_date)

textfile = open("/Users/mac/Desktop/eland-intern/org_file/data.txt", "r")
_title = []
for element in textfile:
    element = element.strip("\n")
    _title.append(element)
textfile.close()

def search_gov(aim_title):
    location = ["臺灣省", "台灣省", "福建省", "臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市", "高雄縣", "基隆市", "新竹市", "嘉義市",
                "新竹縣", "苗栗縣", "彰化縣", "南投縣", "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣", "臺東縣", "澎湖縣"]
    location_1 = ["中央", "中國", "臺灣", "台灣", "臺北", "新北", "桃園", "臺中", "臺南", "高雄", "高雄", "基隆", "新竹", '台中', '台北', '台南', '台東',
                  "嘉義", "新竹", "苗栗", "彰化", "南投", "雲林", "嘉義", "屏東", "宜蘭", "花蓮", "臺東", "澎湖", "福建", "連江", "金門"]

    for j in location:
        if j in aim_title:
            aim_title = aim_title.replace(j, "")

    for j in location_1:
        if j in aim_title:
            aim_title = aim_title.replace(j, "")

    for pattern in _title:
        if re.search(pattern, aim_title):
            logger.debug("1, pattern: {}".format(pattern))
            return True

        if len(aim_title) >= 3 and longestSubstring(aim_title, pattern):
            logger.debug("2, pattern: {}".format(pattern))
            return True

    return False


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


def save_tev(data, path):
    with open(path, 'w', newline='') as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for i in data:
            i[1] = ','.join(i[1])
            tsv_output.writerow(i)

def find_gorg(data):
    for i in data:
        # 找到每段文字中B-ORG I-ORG 的 index
        temp = []
        code = i[1]
        temp += (get_index_positions(code, 'B-ORG'))
        temp += (get_index_positions(code, 'I-ORG'))
        temp.sort()

        # 去做代換
        count = []
        for j in range(len(temp)):
            if j == 0:
                title = i[0][temp[j]]
                count.append(temp[j])
            elif temp[j] == temp[j-1] + 1:
                title += i[0][temp[j]]
                count.append(temp[j])
            else:
                if search_gov(title):
                    logger.debug("title: {}, True".format(title))
                    for z in count:
                        i[1][z] = 'I-GORG'
                    i[1][count[0]] = 'B-GORG'
                else:
                    logger.debug("title: {}".format(title))
                    pass
                count = []
                count.append(temp[j])
                title = i[0][temp[j]]

            if j == len(temp)-1:
                if search_gov(title):
                    logger.debug("title: {}, True".format(title))
                    for z in count:
                        i[1][z] = 'I-GORG'
                    i[1][count[0]] = 'B-GORG'
                else:
                    logger.debug("title: {}".format(title))
                    pass
    return data

def myDict(data):
    myDict = []

    for i in range(len(data)):
        temp = []
        word = list(data[i][0])
        code = data[i][1]
        temp = {"id": i, "tokens": word, "ner_tags": code}
        myDict.append(temp)

    return myDict

def save_json(data, file_name):
    with open(file_name, "w", encoding='utf-8') as f:
        for row in data:
            json.dump(row, f, ensure_ascii=False)
            f.write('\n')   

# 一個func 會在一個畫面
def main():

    tsv_file = open(_open_path)
    read_tsv = csv.reader(tsv_file, delimiter="\t")

    data = []
    for row in read_tsv:
        data.append(row)
    tsv_file.close()

    for i in data:
        i[1] = i[1].split(',')

    data = find_gorg(data)

    # save_tev(data, './output/output_0311.tsv')

    data = myDict(data)
    save_json(data, _save_path)

    s = pd.Series(data)

    training_data, s_rem = train_test_split(s, train_size=0.8, random_state=13)
    test_data, validation_data = train_test_split(
        s_rem, test_size=0.5, random_state=13)

    save_json(list(test_data), "./train_data/test_file_{}.json".format(_date))
    save_json(list(training_data), "./train_data/train_file_{}.json".format(_date))
    save_json(list(validation_data), "./train_data/validation_file_{}.json".format(_date))

if __name__ == '__main__':
    # 設定logging
    logger = logging.getLogger()  # 設置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # 使用FileHandler輸出到file
    fh = logging.FileHandler('./log/log_{}.txt'.format(_date))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # # 使用StreamHandler輸出到console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 使用兩個Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

    main()
