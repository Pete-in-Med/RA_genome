import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import SelectFromModel


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
X_test = scaler.transform(X_test)

# 4. 模型選擇
# 在這個示例中，我們選擇使用Logistic Regression作為我們的分類器
#比較因素和結果是否有關
classifier = LogisticRegression()

# 5. 模型訓練
classifier.fit(X_train, y_train)

# 6. 訓練模型並使用驗證集進行驗證
classifier.fit(X_train, y_train)
val_predictions = classifier.predict(X_val)
conf_mat = confusion_matrix(y_val, val_predictions)
accuracy = accuracy_score(y_val, val_predictions)
precision = precision_score(y_val, val_predictions)
recall = recall_score(y_val, val_predictions)
f1 = f1_score(y_val, val_predictions)

print("驗證集準確率：", accuracy)
print("Confusion Matrix：\n", conf_mat)
print("Precision：", precision)
print("Recall：", recall)
print("F1 Score：\n", f1)

# 7. 模型評估
#混淆矩陣
#準確度
#recall = tp/(tp+fn)
#precision_score = tp/(tp+fn)，最好為1
#F1 = 2 * (precision * recall) / (precision + recall)，最好為1
y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("測試集準確率：", accuracy)
print('Confusion Matrix:\n', confusion_matrix)
print('Recall:', recall)
print('Precision:', precision)
print('F1 Score:', f1)

# 8. 找到重要的預測指標
# 創建Logistic Regression模型，並設置penalty為"L1"
lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)

# 使用SelectFromModel選擇重要特徵
sfm = SelectFromModel(lr, threshold=0.1)
X_train_selected = sfm.fit_transform(X_train, y_train)

# 打印被選擇的特徵
selected_features = [feature_names[i] for i in sfm.get_support(indices=True)]
print("選擇的特徵：", selected_features)

# 9. confusion matrix 的視覺化呈現
# 繪製熱圖
# 設定畫布大小
plt.figure(figsize=(10, 5))

# 設定第一個子圖
plt.subplot(121)
plt.title("Confusion Matrix 1")
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)

# 設定第二個子圖
plt.subplot(122)
plt.title("Confusion Matrix 2")
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d", cbar=False)

# 顯示圖片
plt.show()

# 10. 視覺化呈現準確度、recall值、precision_score和F1
# 建立一個圖形
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# 準確度
sns.barplot(x=["Accuracy"], y=[accuracy], ax=ax[0, 0])
ax[0, 0].set_ylim([0.9, 1])
ax[0, 0].set_title("Accuracy")

# Precision
sns.barplot(x=["Precision"], y=[precision], ax=ax[0, 1])
ax[0, 1].set_ylim([0.9, 1])
ax[0, 1].set_title("Precision")

# Recall
sns.barplot(x=["Recall"], y=[recall], ax=ax[1, 0])
ax[1, 0].set_ylim([0.9, 1])
ax[1, 0].set_title("Recall")

# F1 score
sns.barplot(x=["F1 Score"], y=[f1], ax=ax[1, 1])
ax[1, 1].set_ylim([0.9, 1])
ax[1, 1].set_title("F1 Score")

# 設置圖形標題
fig.suptitle("Model Evaluation")

# 顯示圖形
plt.show()
# 11. 模型部署
# 收集新病患的生物標記資料並進行預測
#new_X = new_data.drop(['姓名', '病歷號', '收件日期', '備註'], axis=1)
#new_X = scaler.transform(new_X)
#new_pred = classifier.predict(new_X)

#print('New Patients Prediction:', new_pred)
