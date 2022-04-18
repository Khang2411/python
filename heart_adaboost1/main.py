import numpy as np
import pandas as pd
import self as self
#train_test_split chia các mảng hoặc ma trận thành các tập con ngẫu nhiên và kiểm tra.
# Điều đó có nghĩa là mỗi khi bạn chạy nó mà không chỉ định random_state, bạn sẽ nhận được một kết quả khác, đây là hành vi dự kiến.
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv("heart.csv")

# Chạy phân tích data
df = pd.read_csv("heart.csv")
sns.countplot(x="result", data=data, palette="bwr")
plt.show()

# Chạy phân tích giới tính xem đến bv nam hay nữ nhiều (không tính vs result)
sns.countplot(x='sex', data=data, palette="mako_r")
plt.xlabel("Sex (0 = Nữ, 1= Nam)")
plt.show()
# Chạy phân tích Tần suất bệnh tim theo FBS
pd.crosstab(data.age, data.result).plot(kind="bar", figsize=(20, 6))
plt.title('Tần suất bệnh tim ở lứa tuổi')
plt.xlabel('Tuổi')
plt.ylabel('Tần suất')
plt.show()
# Phân tích lượng đường khi đói theo sex dựa theo kq bệnh tim
pd.crosstab(data.fbs, data.result).plot(kind="bar", figsize=(15, 6), color=['#FFC300', '#581845'])
plt.title('Tần suất bệnh tim theo FBS')
plt.xlabel('FBS - (Đường huyết lúc đói> 120 mg / dl) (1 = true; 0 = false)')
plt.xticks(rotation=0)
plt.legend(["Không có Bệnh", "Có Bệnh"])
plt.ylabel('Tần suất Bệnh tật hay Không')
plt.show()
# Phân tích Tần suất bệnh tim theo loại đau ngực dựa theo kq (4 loại)
pd.crosstab(data.cp, data.result).plot(kind="bar", figsize=(15, 6), color=['#11A5AA', '#AA1190'])
plt.title('Tần suất bệnh tim theo loại đau ngực')
plt.xlabel('Loại đau ngực')
plt.xticks(rotation=0)
plt.ylabel('Tần suất bệnh tật hay không')
plt.show()
# kết thúc phân tích

from sklearn.model_selection import train_test_split
training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
# lấy dữ liệu để training là 20% = 0 = đề còn lại là test

X_train = training_set.iloc[:, 0:12].values
Y_train = training_set.iloc[:, 13].values
#print(Y_train)

X_test = test_set.iloc[:, 0:12].values # lấy coulmn từ 1->16
Y_test = test_set.iloc[:, 13].values # lấy label Class
from sklearn.ensemble import AdaBoostClassifier
adaboost = AdaBoostClassifier(n_estimators=100, base_estimator=None, learning_rate=1, random_state = 1)
# base_estimator là thuật toán học tập được sử dụng để đào tạo các mô hình yếu.
# Điều này hầu như luôn luôn không cần phải thay đổi bởi vì cho đến nay người học phổ biến nhất sử dụng AdaBoost
# là một cây quyết định - đối số mặc định của tham số này.
# n_estimator là số lượng mô hình để đào tạo lặp đi lặp lại.
# learning-rate các giá trị nhỏ hơn 50% learning_rate thì học yếu -> thúc đầy để học tiếp bằng cách tăng max_dept
adaboost.fit(X_train, Y_train)
Y_pred = adaboost.predict(X_test)
test_set["Predictions"] = Y_pred
print(test_set)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
accuracy = float(cm.diagonal().sum()) / len(Y_test)
print("\nĐộ chính xác của AdaBoost cho tập dữ liệu đã cho :" '', accuracy)
