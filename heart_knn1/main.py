import csv
import numpy as np
import math

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def loadData(path):
    f = open(path, "r")
    data = csv.reader(f)
    data = np.array(list(data))
    data = np.delete(data, 0, 0)  # delete header
    np.random.shuffle(data)
    f.close()

    trainSet = data[:270]
    testSet = data[200:]

    return trainSet, testSet

def calcDistancs(pointA, pointB, numOfFeature=13):
    tmp = 0.0
    for i in range(numOfFeature):
        tmp += (float(pointA[i]) - float(pointB[i])) ** 2
    return math.sqrt(tmp)


def kNearestNeighbor(trainSet, point, k):
    distances = []
    for item in trainSet:
        distances.append({
            "label": item[-1],
            "value": calcDistancs(item, point)
        })
    distances.sort(key=lambda x: x["value"])
    labels = [item["label"] for item in distances]
    return labels[:k]


def findMostOccur(arr):
    labels = set(arr) # set label
    ans = ""
    maxOccur = 0.0
    for label in labels:
        num = arr.count(label)
        if num > maxOccur:
            maxOccur = num
            ans = label
    return ans

if __name__ == "__main__":
    trainSet, testSet = loadData("heart.csv")
    # Chạy phân tích data
    df = pd.read_csv("heart.csv")
    sns.countplot(x="result", data=df, palette="bwr")
    plt.show()

    # Chạy phân tích giới tính xem đến bv nam hay nữ nhiều (không tính vs result)
    sns.countplot(x='sex', data=df, palette="mako_r")
    plt.xlabel("Sex (0 = Nữ, 1= Nam)")
    plt.show()
    # Chạy phân tích Tần suất bệnh tim theo FBS
    pd.crosstab(df.age, df.result).plot(kind="bar", figsize=(20, 6))
    plt.title('Tần suất bệnh tim ở lứa tuổi')
    plt.xlabel('Tuổi')
    plt.ylabel('Tần suất')
    plt.show()
    # Phân tích lượng đường khi đói theo sex dựa theo kq bệnh tim
    pd.crosstab(df.fbs, df.result).plot(kind="bar", figsize=(15, 6), color=['#FFC300', '#581845'])
    plt.title('Tần suất bệnh tim theo FBS')
    plt.xlabel('FBS - (Đường huyết lúc đói> 120 mg / dl) (1 = true; 0 = false)')
    plt.xticks(rotation=0)
    plt.legend(["Không có Bệnh", "Có Bệnh"])
    plt.ylabel('Tần suất Bệnh tật hay Không')
    plt.show()
    # Phân tích Tần suất bệnh tim theo loại đau ngực dựa theo kq (4 loại)
    pd.crosstab(df.cp, df.result).plot(kind="bar", figsize=(15, 6), color=['#11A5AA', '#AA1190'])
    plt.title('Tần suất bệnh tim theo loại đau ngực')
    plt.xlabel('Loại đau ngực')
    plt.xticks(rotation=0)
    plt.ylabel('Tần suất bệnh tật hay không')
    plt.show()
    # kết thúc phân tích

    #KNN dự đoán
    numOfRightAnwser = 0
    for item in testSet:
        knn = kNearestNeighbor(trainSet, item, 1) # lay 1 diem gan nhat
        answer = findMostOccur(knn)
        numOfRightAnwser += item[-1] == answer
        print("label: {} -> predicted: {}".format(item, answer))

    print("Accuracy:", (numOfRightAnwser / len(testSet))*100, "%")
