# #meng-import library pandas, library ini dapat kita gunakan untuk membaca data dalam format xlsx ataupun csv
# import pandas as pd
# pd.set_option('display.max_column', 20)

# #men-load file churn_analysis_train.xlsx sebagai pandas data frame untuk mempermudah proses pengolahan data
# df = pd.read_excel('https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx')

# #perintah untuk menampilkan 5 data pertama 
# print(df.head(5))



# #Kode program sebelumnya
# import pandas as pd
# pd.set_option('display.max_column', 20)

# df = pd.read_excel('https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx')

# #menghilangkan kolom 'ID_Customer' dari data frame dikarenakan kolom ini tidak relevan untuk dijadikan input dalam tugas klasifikasi (ID customer tidak mempengaruhi apakah customer akan lanjut berlangganan atau tidak
# df.drop(['ID_Customer'], axis=1, inplace=True)
# print(df['churn'].value_counts())



# #Kode program sebelumnya
# import pandas as pd
# pd.set_option('display.max_column', 20)

# df = pd.read_excel('https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx')
# df.drop('ID_Customer', axis=1, inplace=True)

# #menyimpan kolom 'churn' sebagai list ke dalam variabel y
# y = df.pop('churn').to_list()

# #mengubah nilai 'Yes' menjadi 1 dan nilai 'No' menjadi 0 agar sesuai dengan format yang sebelumnya telah kita bahas
# y = [1 if label == 'Yes' else 0 for label in y]
# print(df.head())



# #Kode program sebelumnya
# import pandas as pd
# pd.set_option('display.max_column', 20)

# df = pd.read_excel('https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx')
# df.drop('ID_Customer', axis=1, inplace=True)

# y = df.pop('churn').to_list()
# y = [1 if label == 'Yes' else 0 for label in y]

# #memeriksa tipe data dari setiap kolom
# print('Tipe data setiap kolom:')
# print('-----------------------')
# df.info()

# #lakukan pengecekan untuk kolom dengan tipe data 'object' (kategorikal)
# print('\nKolom dengan tipe data object (kategorikal):')
# print('--------------------------------------------')
# for col in df.select_dtypes(include=['object']):
# 	print(df[col].value_counts())
# 	print("===============================")

# #statistik deskriptif dari setiap kolom
# print('\nStatistik deskriptif dari setiap kolom:')
# print('---------------------------------------')
# print(df.describe())



# #Kode program sebelumnya
# import pandas as pd
# pd.set_option('display.max_column', 20)

# df = pd.read_excel('https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx')
# df.drop('ID_Customer', axis=1, inplace=True)

# y = df.pop('churn').to_list()
# y = [1 if label == 'Yes' else 0 for label in y]

# #membuang kolom 'harga_per_bulan'
# df.drop(['harga_per_bulan'], axis=1, inplace=True)

# #membuang kolom 'jumlah_harga_langganan'
# df.drop(['jumlah_harga_langganan'], axis=1, inplace=True)
# df.info()



#Kode program sebelumnya
import pandas as pd
pd.set_option('display.max_column', 20)

df = pd.read_excel('https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx')
df.drop('ID_Customer', axis=1, inplace=True)
df.drop('harga_per_bulan', axis=1, inplace=True)
df.drop('jumlah_harga_langganan', axis=1, inplace=True)

y = df.pop('churn').to_list()
y = [1 if label == 'Yes' else 0 for label in y]

#mengimport class LabelEncoder untuk mengubah atribut dengan dua kemungkinan nilai (binary)
from sklearn.preprocessing import LabelEncoder
 
#menyiapkan dictionary untuk menyimpan seluruh LabelEncoder untuk setiap atribut kategorikal yang bersifat biner
labelers = {}
 
#untuk setiap kolom dengan tipe data 'object' (kategorikal)
column_categorical_non_binary = []
for col in df.select_dtypes(include=['object']):
	#saat jumlah nilai unik dari suatu kolom sama dengan dua
	#atau dengan kata lain kolom bersifat biner
	if len(df[col].unique()) == 2:
		#buat objek LabelEncoder baru untuk kolom dan tampung dalam
		#dictionary labelers
		labelers[___] = LabelEncoder()
		#meminta objek LabelEncoder untuk mempelajari dan
		#mentransformasikan kolom
		df[___] = labelers[col].fit_transform(___)
	#untuk kolom bersifat non-biner
	else:
		#tambahkan nama kolom ke dalam array yang telah disiapkan
		column_categorical_non_binary.append(col)		
print(___)

df = ___
print(___)