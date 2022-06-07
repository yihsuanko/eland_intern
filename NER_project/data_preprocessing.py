from numpy import NaN
import pandas as pd
from lcs import longestSubstring
from parameter import *

_org_path = "./org_list_utf8.csv"
_school_path = './school_list.csv'
_save_path = "./data.txt"
_count = 3


def clean_text(text_list, clean_element):

    text_list = [x for x in text_list if clean_element not in x]
    return text_list


def data_extraction(path, title):

    data = pd.read_csv(path, encoding='utf8', header=1)
    data.rename(columns=title, inplace=True)
    data = set(data["title"])

    return data


def data_combine(*args):
    data = set()
    for title in args:
        data = data | title
    return data


def data_minus(data, *args):
    for title in args:
        data = data - title
    return data


def save_txt(path, data):
    textfile = open(path, "w")
    for element in data:
        textfile.write(element + "\n")
    textfile.close()


def replace_location(raw_data, new, *args):
    data = pd.DataFrame(raw_data, columns=["title"])
    for location in args:
        for place in location:
            data["title"] = data["title"].str.replace(place, new)
    return data


def long_same(start, end, data):
    long_word = []
    count = _count
    for j in range(start + 1, end):
        str1 = data[start]
        str2 = data[j]
        title = longestSubstring(str1, str2)
        if title not in long_word and len(title) > count:
            long_word.append(title)
            count = len(title)
        elif title in long_word or len(title) < count:
            continue

    return long_word


def add_org(data):
    temp = []
    for i in range(len(data)):
        count = 3
        if i > len(data) - 20:
            temp += (long_same(i+1, len(data), data))
        else:
            temp += (long_same(i+1, i+10, data))
    data += temp
    return data


def strict_del(data, *args):
    for text in args:
        for i in range(len(data)):
            element = len(data)-1-i
            if data[element] == text:
                del data[element]
    return data


def text_del(data, *args):
    for text in args:
        for i in range(len(data)):
            element = len(data)-1-i
            if data[element][0] == text:
                data[element] = data[element][1:]
    return data


def main():
    org = data_extraction(_org_path, {'機關名稱': 'title'})
    au_org = data_extraction(_org_path, {'主管機關名稱': 'title'})
    school = data_extraction(_school_path, {'01-MAR-22': 'title'})

    all_org_title = data_combine(org, au_org)
    all_org_title = data_minus(all_org_title, school)
    all_org_title = list(all_org_title)
    all_org_title.remove(NaN)

    for i in DEL_ORG:
        all_org_title = clean_text(all_org_title, i)

    clean = [x for x in all_org_title if "(" in x]
    all_org_title = [x for x in all_org_title if "(" not in x]

    for i in clean:
        i = i.split("(")
        all_org_title.append(i[0])

    all_org_title.sort()

    all_org_title = replace_location(
        all_org_title, "", LOCATION, LOCATION_1, LOCATION_2)
    all_org_title = replace_location(all_org_title, "駐.+", LOCATION_3)
    all_org_title = all_org_title["title"].tolist()
    all_org_title.sort()
    all_org_title = add_org(all_org_title)

    all_org_title = replace_location(
        all_org_title, "", LOCATION, LOCATION_1, DEL_COMPANY)
    all_org_title = all_org_title["title"].tolist()
    all_org_title = list(set(all_org_title))

    strict_del(all_org_title, "", "社會福利", "中華民國")
    text_del(all_org_title, "立", "區")

    for i in range(len(all_org_title)):
        title = len(all_org_title)-1-i

        if " " in all_org_title[title]:
            del all_org_title[title]
        if ".*" in all_org_title[title]:
            temp = all_org_title[title].replace(".*", "")
        if len(all_org_title[title]) < 3:
            del all_org_title[title]

    all_org_title += ADD_ORG
    all_org_title = list(set(all_org_title))
    all_org_title.sort()

    save_txt(_save_path, all_org_title)


if __name__ == "__main__":
    main()
