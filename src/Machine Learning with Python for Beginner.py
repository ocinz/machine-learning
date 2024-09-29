# Terminologi Machine Learning

# Dalam pembuatan model machine learning tentunya dibutuhkan data. Sekumpulan data yang digunakan dalam machine learning disebut DATASET, yang kemudian dibagi/di-split menjadi training dataset dan test dataset.

# TRAINING DATASET digunakan untuk membuat/melatih model machine learning, sedangkan TEST DATASET digunakan untuk menguji performa/akurasi dari model yang telah dilatih/di-training.

# Teknik atau pendekatan yang digunakan untuk membangun model disebut ALGORITHM seperti Decision Tree, K-NN, Linear Regression, Random Forest, dsb. dan output atau hasil dari proses melatih algorithm dengan suatu dataset disebut MODEL.

# Umumnya dataset disajikan dalam bentuk tabel yang terdiri dari baris dan kolom. Bagian Kolom adalah FEATURE atau VARIABEL data yang dianalisa, sedangkan bagian baris adalah DATA POINT/OBSERVATION/EXAMPLE.

# Hal yang menjadi target prediksi atau hal yang akan diprediksi dalam machine learning disebut LABEL/CLASS/TARGET. Dalam statistika/matematika, LABEL/CLASS/TARGET ini dinamakan dengan Dependent Variabel, dan FEATURE adalah Independent Variabel.

# Machine Learning itu terbagi menjadi 2 tipe yaitu supervised dan unsupervised Learning. Jika LABEL/CLASS dari dataset sudah diketahui maka dikategorikan sebagai supervised learning, dan jika Label belum diketahui maka dikategorikan sebagai unsupervised learning

# import pandas as pd
# dataset = pd.read_csv("https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv")
# print('Shape dataset:', dataset.shape)
# print('\nLima data teratas:\n', dataset.head())
# print('\nInformasi dataset:')
# print(dataset.info())
# print('\nStatistik deskriptif:\n', dataset.describe())

# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

# # Mengubah kolom 'Revenue' dan 'Weekend' menjadi numerik (True/False -> 1/0)
# # dataset['Revenue'] = dataset['Revenue'].apply(lambda x: 1 if x == True else 0)
# dataset['Revenue'] = dataset['Revenue'].astype(int)
# dataset['Weekend'] = dataset['Weekend'].astype(int)

# # Memilih hanya kolom numerik untuk menghitung korelasi
# dataset_numeric = dataset.select_dtypes(include=['float64', 'int64'])

# dataset_corr = dataset_numeric.corr()
# print("Korelasi dataset:\n", dataset_numeric.corr())
# print("Distribusi Label (Revenue):\n", dataset_numeric['Revenue'].value_counts())
# print("Korelasi BounceRates-ExitRates:\n", dataset_corr.loc['BounceRates','ExitRates'])
# print("Korelasi Revenue-PageValues:\n", dataset_corr.loc['Revenue','PageValues'])
# print("Korelasi TrafficType-Weekend:\n", dataset_corr.loc['TrafficType','Weekend'])

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

# # checking the Distribution of customers on Revenue
# plt.rcParams['figure.figsize'] = (12, 5)
# plt.subplot(1, 2, 1)
# sns.countplot(x='Revenue', data=dataset, palette='pastel', hue='Revenue', legend=False) #kode baru
# # sns.countplot(dataset['Revenue'], palette='pastel')
# plt.title('Buy or Not', fontsize=20)
# plt.xlabel('Revenue or Not', fontsize=14)
# plt.ylabel('count', fontsize=14)
# # checking the Distribution of customers on Weekend
# plt.subplot(1, 2, 2)
# sns.countplot(x='Weekend', data=dataset, palette='inferno', hue='Weekend', legend=False) #kode baru
# # sns.countplot(dataset['Weekend'], palette='inferno')
# plt.title('Purchase on Weekends', fontsize=20)
# plt.xlabel('Weekend or not', fontsize=14)
# plt.ylabel('count', fontsize=14)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

# # visualizing the distribution of customers around the Region
# plt.hist(dataset['Region'], color = 'lightblue')
# plt.title('Distribution of Customers', fontsize = 20)
# plt.xlabel('Region Codes', fontsize = 14)
# plt.ylabel('Count Users', fontsize = 14)
# plt.show()


# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

# #checking missing value for each feature  
# print('Checking missing value for each feature:')
# print(dataset.isnull().sum())
# #Counting total missing value
# print('\nCounting total missing value:')
# print(dataset.isnull().sum().sum())


# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

# #Drop rows with missing value   
# dataset_clean = dataset.dropna() 
# print('Ukuran dataset_clean:', dataset_clean.shape)



# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

# print("Before imputation:")
# # Checking missing value for each feature  
# print(dataset.isnull().sum())
# # Counting total missing value  
# print(dataset.isnull().sum().sum())

# print("\nAfter imputation:")
# # Fill missing value with mean of feature value  
# dataset.fillna(dataset.mean(numeric_only=True), inplace = True)
# # Checking missing value for each feature  
# print(dataset.isnull().sum())
# # Counting total missing value  
# print(dataset.isnull().sum().sum()) 



# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')

# print("Before imputation:")
# # Checking missing value for each feature  
# print(dataset.isnull().sum())
# # Counting total missing value  
# print(dataset.isnull().sum().sum())

# print("\nAfter imputation:")
# # Fill missing value with mean of feature value  
# dataset.fillna(dataset.median(numeric_only=True), inplace = True)
# # Checking missing value for each feature  
# print(dataset.isnull().sum())
# # Counting total missing value  
# print(dataset.isnull().sum().sum()) 



# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
# dataset.fillna(dataset.mean(numeric_only=True), inplace = True)

# from sklearn.preprocessing import MinMaxScaler  
# #Define MinMaxScaler as scaler  
# scaler = MinMaxScaler()  
# #list all the feature that need to be scaled  
# scaling_column = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues']
# #Apply fit_transfrom to scale selected feature  
# dataset[scaling_column] = scaler.fit_transform(dataset[scaling_column])
# #Cheking min and max value of the scaling_column
# print(dataset[scaling_column].describe().T[['min','max']])



import pandas as pd
dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
dataset.fillna(dataset.mean(numeric_only=True), inplace = True)

# import numpy as np
# from sklearn.preprocessing import LabelEncoder
# # Convert feature/column 'Month'
# LE = LabelEncoder()
# dataset['Month'] = LE.fit_transform(dataset['Month'])
# print(LE.classes_)
# print(np.sort(dataset['Month'].unique()))
# print(' ')

# # Convert feature/column 'VisitorType'
# LE = LabelEncoder()
# dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])
# print(LE.classes_)
# print(np.sort(dataset['VisitorType'].unique()))


# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
# dataset.fillna(dataset.mean(numeric_only=True), inplace = True)

# from sklearn.preprocessing import LabelEncoder
# LE = LabelEncoder()
# dataset['Month'] = LE.fit_transform(dataset['Month'])
# LE = LabelEncoder()
# dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])

# # removing the target column Revenue from dataset and assigning to X
# X = dataset.drop('Revenue', axis=1)
# # assigning the target column Revenue to y
# y = dataset['Revenue']
# # checking the shapes
# print('Shape of X:', X.shape)
# print('Shape of y:', y.shape)


# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
# dataset.fillna(dataset.mean(numeric_only=True), inplace = True)

# from sklearn.preprocessing import LabelEncoder
# LE = LabelEncoder()
# dataset['Month'] = LE.fit_transform(dataset['Month'])
# LE = LabelEncoder()
# dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])
# X = dataset.drop(['Revenue'], axis = 1)
# y = dataset['Revenue']

# from sklearn.model_selection import train_test_split
# # splitting the X, and y
# X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=0)
# # checking the shapes
# print('Shape of X_train :', X_train.shape)
# print('Shape of y_train :', y_train.shape)
# print('Shape of X_test :', X_test.shape)
# print('Shape of y_test :', y_test.shape)


# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
# dataset.fillna(dataset.mean(numeric_only=True), inplace = True)

# from sklearn.preprocessing import LabelEncoder
# LE = LabelEncoder()
# dataset['Month'] = LE.fit_transform(dataset['Month'])
# LE = LabelEncoder()
# dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])
# X = dataset.drop(['Revenue'], axis = 1)
# y = dataset['Revenue']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model = model.fit(X_train,y_train)
# y_pred = model.predict(X_test)

# from sklearn.metrics import confusion_matrix, classification_report

# # evaluating the model
# print('Training Accuracy :', model.score(X_train, y_train))
# print('Testing Accuracy :', model.score(X_test, y_test))

# # confusion matrix
# print('\nConfusion matrix:')
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# # classification report
# print('\nClassification report:')
# cr = classification_report(y_test, y_pred)
# print(cr)



# import pandas as pd
# dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
# dataset.fillna(dataset.mean(numeric_only=True), inplace = True)

# from sklearn.preprocessing import LabelEncoder
# LE = LabelEncoder()
# dataset['Month'] = LE.fit_transform(dataset['Month'])
# LE = LabelEncoder()
# dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])
# X = dataset.drop(['Revenue'], axis = 1)
# y = dataset['Revenue']

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix, classification_report

# # Call the classifier
# # logreg = LogisticRegression()
# logreg = LogisticRegression(max_iter=1000, solver='liblinear')
# # Fit the classifier to the training data  
# logreg = logreg.fit(X_train, y_train)
# #Training Model: Predict 
# y_pred = logreg.predict(X_test)

# #Evaluate Model Performance
# print('Training Accuracy :', logreg.score(X_train, y_train))  
# print('Testing Accuracy :', logreg.score(X_test, y_test))  

# # confusion matrix
# print('\nConfusion matrix')  
# cm = confusion_matrix(y_test, y_pred)  
# print(cm)

# # classification report  
# print('\nClassification report')  
# cr = classification_report(y_test, y_pred)  
# print(cr)



import pandas as pd
dataset = pd.read_csv('https://storage.googleapis.com/dqlab-dataset/pythonTutorial/online_raw.csv')
dataset.fillna(dataset.mean(numeric_only=True), inplace = True)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
dataset['Month'] = LE.fit_transform(dataset['Month'])
LE = LabelEncoder()
dataset['VisitorType'] = LE.fit_transform(dataset['VisitorType'])
X = dataset.drop(['Revenue'], axis = 1)
y = dataset['Revenue']

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Call the classifier
decision_tree = DecisionTreeClassifier()
# Fit the classifier to the training data
decision_tree = decision_tree.fit(X_train, y_train)

# evaluating the decision_tree performance
print('Training Accuracy :', decision_tree.score(X_train, y_train))
print('Testing Accuracy :', decision_tree.score(X_test, y_test))