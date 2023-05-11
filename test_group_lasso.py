import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel
from group_lasso import GroupLasso
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 1. 收集資料
# 假設我們已經從醫院或實驗室獲取了病患的生物標記資料，並已標記他們是否已被診斷為RA

# 2. 數據預處理
# 將生物標記資料轉換為數字形式，並將資料分為訓練集和測試集

# 讓使用者輸入檔案名稱
print("請記得把運行的資料和執行檔放在同個資料夾")
file_name = input("請輸入檔案名稱：")
file_name = file_name + ".csv"

# 使用pandas讀取CSV檔案
data = pd.read_csv(file_name)
X = data.drop(['姓名', '病歷號', '收件日期'], axis=1)
X.dropna(inplace=True)
y = X['RA_diagnosis']
X = X.drop(['RA_diagnosis'], axis=1)

#checkpoint
#將數據集分成訓練集、驗證集和測試集
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# 檢查每個集合的大小
print("訓練集大小：", len(X_train))
print("驗證集大小：", len(X_val))
print("測試集大小：", len(X_test))

# 3. 數據標準化
# 使用StandardScaler進行標準化
#利用平均值和標準差
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 設定group lasso的超參數
groups = np.array([0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10])
alpha = 0.01

# 創建並擬合group lasso模型
model = GroupLasso(groups=groups)
model.fit(X_train, y_train)

# 預測驗證集
y_pred = model.predict(X_val)

# 計算準確度、召回率、精確率和F1-score
accuracy = accuracy_score(y_val, y_pred)
conf_matrix = confusion_matrix(y_val, y_pred)
class_report = classification_report(y_val, y_pred)

print("Validation Accuracy: {:.2f}%".format(accuracy*100))
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# 顯示特徵重要性
coef = model.coef_
coef_dict = {X.columns[i]: coef[i] for i in range(len(X.columns))}
coef_dict = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
print("Feature importance:")
for feature, importance in coef_dict:
    print("{}: {:.4f}".format(feature, importance))
