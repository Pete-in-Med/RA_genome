import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# 1. 收集資料
# 假設我們已經從醫院或實驗室獲取了病患的生物標記資料，並已標記他們是否已被診斷為RA

# 2. 數據預處理
# 將生物標記資料轉換為數字形式，並將資料分為訓練集和測試集
# 清理資料(自己加的)
data = pd.read_csv('tri_service_hospital_Treg_subset_20220508.csv')
X = data.drop('RA_diagnosis', axis=0)
y = data['RA_diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
data.dropna(inplace=True)

# 3. 模型選擇
# 在這個示例中，我們選擇使用Logistic Regression作為我們的分類器
classifier = LogisticRegression()

# 4. 模型訓練
classifier.fit(X_train, y_train)

# 5. 模型評估
#混淆矩陣
#準確度
#recall = tp/(tp+fn)
#precision_score = tp/(tp+fn)，最好為1
#F1 = 2 * (precision * recall) / (precision + recall)，最好為1
y_pred = classifier.predict(X_test)
print(y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print('Confusion Matrix:\n', confusion_matrix)
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F1 Score:', f1)

# 6. 模型部署
# 收集新病患的生物標記資料並進行預處理
# 清理資料
new_data = pd.read_csv('new_biomarker_data.csv')
new_data.dropna(inplace=True)
new_X = new_data.drop('RA_diagnosis', axis=1)

# 使用已經訓練好的模型對新病患進行預測
new_pred = classifier.predict(new_X)
print('New patient RA diagnosis prediction:', new_pred)
