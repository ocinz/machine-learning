# import pandas as pd

# # Load Dataset
# df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/HousingData.csv')

# # Tampilkan jumlah data duplikat
# print(df.duplicated().sum())

# # Tampilkan jumlah data yang missing
# print(df.isna().sum().sum())

# # Drop baris yang mempunyai baris kosong
# df = df.dropna()

# # Tampilkan berapa baris sisa dari data yang sudah didrop
# print(df.shape)



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load Dataset
# df = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/HousingData.csv')
# df = df.dropna()

# # Tampilkan histogram plot pada kolom AGE menggunakan library seaborn
# sns.histplot(df['AGE'])
# plt.show()

# #Tampilkan juga histogram plot pada kolom AGE menggunakan library matplotlib, kemudian tambahkan xlabel (Umur) dan ylabel (Jumlah) dengan title (Sebaran Umur Rumah)
# plt.hist(df['AGE'])
# plt.title('Sebaran Umur Rumah')
# plt.xlabel('Umur')
# plt.ylabel('Jumlah')
# plt.show()

# #Tampilkan pie chart pada kolom "CHAS" menggunakan library matplotlib dengan melakukan perhitungan jumlahnya terlebih dahulu
# count_chas = df.groupby('CHAS').agg(chas_count=('CHAS', 'count')).reset_index()
# print(count_chas['CHAS'])
# plt.pie(count_chas['chas_count'], labels=count_chas['CHAS'], autopct='%1.1f%%')
# plt.show()



# import pandas as pd
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# # Load Dataset
# dataset = 'https://storage.googleapis.com/dqlab-dataset/HousingData.csv'
# df = pd.read_csv(dataset)

# # drop missing values
# df = df.dropna()

# # pisahkan antara kolom variable dan target ("MEDV" menjadi kolom target)
# X = df.drop('MEDV', axis=1)
# y = df['MEDV']

# # train test split dengan ratio 0,75 dan 0,25 dan set random_state=0
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# # train dataset menggunakan Linear Regression
# model = LinearRegression()
# model.fit(X_train, y_train)

# y_train_predict = model.predict(X_train)
# y_test_predict = model.predict(X_test)

# #hitung rmse dan r2 untuk dataset train
# rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
# r2_train = r2_score(y_train, y_train_predict)

# #hitung rmse dan r2 untuk dataset test
# rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
# r2_test = r2_score(y_test, y_test_predict)

# print(rmse_train)
# print(rmse_test)
# print(r2_train)
# print(r2_test)



# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# # Load Dataset
# dataset = 'https://storage.googleapis.com/dqlab-dataset/HousingData.csv'
# df = pd.read_csv(dataset)

# # drop missing values
# df = df.dropna()

# # tampilkan boxplot
# sns.boxplot(data=df)
# plt.show()

# # lakukan log transformation untuk kolom "B" dan "CRIM"
# df['B'] = np.log(df['B'])
# df['CRIM'] = np.log(df['CRIM'])

# # pisahkan antara kolom variable dan target ("MEDV" menjadi kolom target)
# X = df.drop('MEDV', axis=1)
# y = df['MEDV']


# # train test split dengan ratio 0,75 dan 0,25 dan set random_state=0
# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)

# # train dataset menggunakan Ridge Regression dengan random_state=0
# ridge = Ridge(random_state=0)
# ridge.fit(X_train, y_train)

# y_train_predict = ridge.predict(X_train)
# y_test_predict = ridge.predict(X_test)

# #hitung rmse dan r2 untuk model Ridge Regression
# rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
# r2_train = r2_score(y_train, y_train_predict)

# rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
# r2_test = r2_score(y_test, y_test_predict)

# print(rmse_train)
# print(rmse_test)
# print(r2_train)
# print(r2_test)

# # train dataset menggunakan Gradaient Boosting dengan random_state=0
# grad_boost = GradientBoostingRegressor(random_state=0)
# grad_boost.fit(X_train, y_train)

# y_train_predict = grad_boost.predict(X_train)
# y_test_predict = grad_boost.predict(X_test)

# #hitung rmse dan r2 untuk model Gradient Boosting
# rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
# r2_train = r2_score(y_train, y_train_predict)

# rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
# r2_test = r2_score(y_test, y_test_predict)

# print(rmse_train)
# print(rmse_test)
# print(r2_train)
# print(r2_test)



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load Dataset
dataset = 'https://storage.googleapis.com/dqlab-dataset/data_misi_classification.csv'
data_df = pd.read_csv(dataset)

# tampilkan 5 data teratas dari data_df
print(data_df.head())

# encoding kolom-kolom binary / boolean termasuk kolom `sex` menjadi nilai 0 dan 1
# f => 0
# t => 1
data_df.replace('f', 0, inplace=True)
data_df.replace('t', 1, inplace=True)

# male (M) => 0
# female (F) => 1
data_df['sex'].replace('M', 0, inplace=True) # male mapped to 0
data_df['sex'].replace('F', 1, inplace=True) # female mapped to 1

# impute missing variable dengan nilai 0
data_df.replace(np.nan, 0, inplace=True)

# labeling target variable dengan ketentuan sebagai berikut {'negative': 0, 'hypothyroid': 1, 'hyperthyroid': 2}
diagnoses = {'negative': 0,
             'hypothyroid': 1,
             'hyperthyroid': 2}
data_df['target'] = data_df['target'].map(diagnoses)

# Kemudian pisahkan antara kolom variable dan target
X = data_df.drop('target', axis=1)
y = data_df['target']

# Kemudian lakukan train test split dengan ratio 0,75 dan 0,25 dan set random_state=0
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
print('X training:',X_train.shape, 'y training:', y_train.shape)
print('X testing:',X_test.shape, 'y testing:', y_test.shape)

# Setelah displit, train dataset menggunakan algoritma SVM, DT dan Random Forest dengan random_state = 0
svm_mod = SVC(random_state=0)
svm_mod.fit(X_train, y_train)

dt_mod = DecisionTreeClassifier(random_state=0)
dt_mod.fit(X_train, y_train)

rf_mod = RandomForestClassifier(random_state=0)
rf_mod.fit(X_train, y_train)

# Analisis performa tiap algorithma pada data train dengan melihat classification report dan metric performa
print('------------- performa data train --------------')
y_pred = svm_mod.predict(X_train)
print('Report SVM:')
print(classification_report(y_train, y_pred))

y_pred = dt_mod.predict(X_train)
print('Report DT:')
print(classification_report(y_train, y_pred))

y_pred = rf_mod.predict(X_train)
print('Report RF:')
print(classification_report(y_train, y_pred))


# Analisis performa tiap algorithma pada data test dengan melihat classification report dan metric performa
print('------------- performa data test --------------')
y_pred = svm_mod.predict(X_test)
print('Report SVM:')
print(classification_report(y_test, y_pred))

y_pred = dt_mod.predict(X_test)
print('Report DT:')
print(classification_report(y_test, y_pred))

y_pred = rf_mod.predict(X_test)
print('Report RF:')
print(classification_report(y_test, y_pred))