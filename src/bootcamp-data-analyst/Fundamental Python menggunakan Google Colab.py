# import pandas as pd

# # load dataset
# dataset = 'https://storage.googleapis.com/dqlab-dataset/SuperStore.csv'
# df = pd.read_csv(dataset)

# # Pisahkan Customer Name menjadi dua komponen yaitu First_Name dan Last_Name
# name = df['Customer_Name'].str.split(" ", n=1, expand=True)
# df['First_Name'] = name[0]
# df['Last_Name'] = name[1]

# # tampilkan 5 baris pertama
# print(df.head())


# konsumen = {
#     'nama' : 'Claire',
#     'voucher' : [10, 15, 20, 25, 30, 50]
# }

# print(konsumen['voucher'][:3])


# def kalkulator(num1,operator,num2):
#     if operator == "+":
#         result = num1 + num2
#     elif operator == "-":
#         result = num1 - num2
#     elif operator == "*":
#         result = num1 * num2
#     elif operator == "/":
#         if num2 != 0:
#             result = num1 / num2
#         else:
#             print("Error: Tidak bisa dibagi dengan nol!")
#             return
    
#     print("Hasil: ", result)
	
# kalkulator(1,'+',2)
# kalkulator(3,'-',2)
# kalkulator(4,'*',2)
# kalkulator(6,'/',2)


# import pandas as pd
# dataset = 'https://storage.googleapis.com/dqlab-dataset/SuperStore.csv'
# df = pd.read_csv(dataset)
# print(df.keys())
# print(pd.pivot_table(
#     data=df,
#     index=['Region','Segment'],
#     columns='Category',
#     values='Sales',
#     aggfunc='sum'
# ))


# import pandas as pd

# dataset = 'https://storage.googleapis.com/dqlab-dataset/SuperStore_missing.csv'
# df = pd.read_csv(dataset)

# # Isi missing values pada kolom Sales dengan menggunakan nilai median
# df.fillna(df.median(numeric_only=True), inplace=True)

# total_na_sales = df['Sales'].isna().sum()

# print(total_na_sales)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = 'https://storage.googleapis.com/dqlab-dataset/SuperStore.csv'
df = pd.read_csv(dataset)

sales = pd.pivot_table(
    data=df,
    index=['Region','Category'],
    values='Sales',
    aggfunc='sum'
).reset_index()

print(sales)

sns.barplot(
    data=sales,
    x='Sales',
    y='Region',
    hue='Category'
)
plt.show()