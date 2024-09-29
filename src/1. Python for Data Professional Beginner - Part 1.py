# start learning data science
print("Hello World!")

# print command
print("Halo Dunia")
print("Riset Bahasa Python")

# Statement | Instruksi yang diberikan secara baris per baris untuk dijalankan oleh mesin
print("Belajar Python menyenangkan") 
print("Halo Dunia")
print("Hello World!")
# Variables | Lokasi penyimpanan yang dapat digunakan untuk menampung sebuah data atau informasi
# Literals | Simbol-simbol yang dapat kita gunakan untuk mengisi suatu variabel
bilangan1 = 5
bilangan2 = 10
kalimat1 = "Belajar Bahasa Python"
# Operators | Simbol-simbol yang dapat digunakan untuk mengubah nilai dari satu variabel dengan melibatkan satu atau lebih variabel dan literal
print(bilangan1 + bilangan2)

# tugas praktek 
bilangan1 = 20
bilangan2 = 10 
print(bilangan1-bilangan2)

# tugas praktek
harga_asli = 20000
potongan = 2000
harga_setelah_potongan = harga_asli - potongan 
harga_final = harga_setelah_potongan * 1.1 
print(harga_final)

# Struktur Program Python - Part 2

# 1. Reserved Words: Kumpulan kata-kata yang memiliki makna khusus dalam bahasa pemrograman Python. Kata False, return, dan for merupakan contoh dari reserved words.

# 2. Whitespace: Pada bahasa Python, spasi dan tab memiliki makna khusus untuk menandai serangkaian blok dalam kode Python. Hal ini akan dijelaskan secara lebih lanjut pada bagian struktur pemilihan dan struktur pengulangan pada bahasa Python.

# 3. Comments: Comments merupakan sekumpulan teks yang dituliskan di dalam sebuah program yang tidak akan mempengaruhi hasil dari sebuah program. Walaupun tidak mempengaruhi hasil program, comments merupakan salah satu komponen yang penting dalam pengembangan program. Hal tersebut dikarenakan comments dapat diselipkan di antara sekumpulan statements yang telah dituliskan, untuk berkomunikasi dengan rekan programmer lainnya dalam satu tim. 
# Terdapat dua jenis comments di dalam Python, yaitu:
# a. single line comment (comments dalam satu baris)
# b. multi line comment (comments dalam beberapa baris)


# Aturan Penamaan Python Variables

# Nama dari sebuah variabel harus dimulai dengan huruf (a-z, A-Z) atau karakter garis bawah underscore (_) dan tidak dapat dimulai dengan angka (0-9).
# Variabel hanya boleh mengandung karakter alfabet, bilangan dan underscore (a-z, A-Z, 0-9, _)
# Variabel bersifat case-sensitive yang mengartikan bahwa variabel TINGGI, tinggi, dan Tinggi merujuk pada tiga variabel berbeda.
# Selain dapat mendeklarasikan nilai dari suatu variabel secara baris per baris, aku juga dapat mendeklarasikan beberapa variabel dalam satu baris dengan menggunakan ekspresi seperti:

bill1, bill2 = 3, 4
salam = "Selamat Pagi"; penutup = "Salam Sejahtera"

# Tipe Data Dasar: Null, Boolean, Numeric dan Text

# 1. Null Type: Tipe data null dalam Python digunakan untuk menyimpan nilai kosong atau tidak ada yang dinyatakan dengan None.
# 2. Boolean Type: Tipe data boolean atau bool digunakan untuk menyimpan nilai kebenaran (True, False) dari suatu ekspresi logika.
# 3. Numeric Type: Tipe data yang digunakan untuk menyimpan data berupa angka. Terdapat dua macam tipe data numeric, yaitu int untuk menyimpan bilangan bulat (e.g.: 0, 1, 2, 404, -500, -1000) dan float untuk menyimpan bilangan riil (e.g.: 0.5, 1.01, 2.05, 4.04)
# 4. Text Type: Pada Python, tipe data string (str) digunakan untuk menyimpan data teks. Tipe data string dimulai dengan tanda kutip (baik kutip satu/ dua) dan diakhir dengan tanda kutip. Contoh: "Teks", "Contoh teks menggunakan Python", dan 'Teks pada Python'.


# Sequence Type – Part 1

# Tipe data ini digunakan untuk menampung sekumpulan data secara terorganisir.
# Bentuk dari tipe data sequence ini adalah List dan Tuple.

# Tipe data List diawali dengan tanda kurung siku buka ( [ ), memisahkan setiap elemen di dalamnya dengan tanda koma ( , ) dan ditutup dengan kurung siku tutup ( ] ). Sebagai contoh: 
contoh_list = [1, 'dua', 3, 4.0, 5]
# Setiap elemen dari list memiliki indeks yang dimulai dari angka 0 dan terus bertambah satu nilainya hingga elemen terakhir dari list. Sebagai contoh:
print(contoh_list[0])
print(contoh_list[3])
# Tipe data list bersifat mutable yang berarti setiap elemen di dalam list dapat diubah nilainya setelah proses pendeklarasian list. Sebagai contoh:
contoh_list[3] = 'empat'
print(contoh_list[3])

# Tipe data tuple juga berfungsi untuk menampung sekumpulan data. Tipe data ini diawali dengan tanda kurung buka ( ( ), memisahkan setiap elemen di dalamnya dengan tanda koma ( , ) dan ditutup dengan tanda kurung tutup ( ) ). Sebagai contoh:
contoh_tuple = ('Januari', 'Februari', 'Maret', 'April')
# Aturan indeks dan cara mengakses elemen pada sebuah tuple serupa dengan list. Sebagai contoh:
print(contoh_tuple[0])
# Berbeda dengan tipe data list, tipe data tuple bersifat immutable yang berarti elemen pada tipe data tuple tidak dapat diubah setelah proses pendeklarasiannya.

# Set Type
# Serupa dengan tipe data sequence, tipe data set digunakan untuk menampung sekumpulan data dengan tipe lainnya. Terdapat dua jenis dari tipe data set yaitu, set dan frozenset.

# Tipe data set diawali dengan tanda kurung buka kurawal ( { ), memisahkan setiap elemen di dalamnya dengan tanda koma ( , ) dan ditutup dengan tanda kurung tutup ( } ). Namun berbeda dengan tipe data sequence, seperti list, tipe data objek tidak mengizinkan adanya elemen dengan nilai yang sama dan tidak mempedulikan urutan dari elemen.

contoh_set = {'Dewi', 'Budi', 'Cici', 'Linda', 'Cici'} 
print(contoh_set)
contoh_frozen_set = ({'Dewi', 'Budi', 'Cici', 'Linda', 'Cici'})
print(contoh_frozen_set)

#  Tipe data frozenset sebenarnya hanya merupakan set yang bersifat immutable, yang artinya setiap elemen di dalam frozenset tidak dapat diubah setelah proses deklarasinya.


# Mapping Type
# Tipe data mapping dapat digunakan untuk memetakan sebuah nilai ke nilai lainnya. Dalam Python, tipe data mapping disebut dengan istilah dictionary. Tipe data dictionary dapat dideklarasikan dengan diawali oleh tanda kurung buka kurawal ( { ), memisahkan setiap elemen di dalamnya dengan tanda koma ( , ) dan ditutup dengan tanda kurung kurawal ( } ). Setiap elemen pada tipe data dictionary dideklarasikan dengan format:

# "kunci" : "nilai"
person = {'nama': 'John Doe', 'pekerjaan': 'Programmer'}
print(person['nama'])
print(person['pekerjaan'])

# Tugas Praktek
sepatu = {"nama": "Sepatu Niko", "harga": 150000, "diskon": 30000}
baju = {"nama": "Baju Unikloh", "harga": 80000, "diskon": 8000}
celana = {"nama": "Celana Lepis", "harga": 200000, "diskon": 60000}
daftar_belanja = [sepatu, baju, celana]
# Hitung harga masing-masing data setelah dikurangi diskon
harga_sepatu = sepatu["harga"] - sepatu["diskon"]
harga_baju = baju["harga"] - baju["diskon"]
harga_celana = celana["harga"] - celana["diskon"]
# Hitung harga total
total_harga = harga_sepatu + harga_baju + harga_celana 
# Hitung harga kena pajak
total_pajak = total_harga * 0.1
# Cetak total_harga + total_pajak
print(total_harga + total_pajak)


# Identity operator
x = ["Ani", "Budi"]
y = ["Ani", "Budi"]
a = x
print(a is x)
# akan menampilkan nilai True dikarenakan a dan x merujuk ke objek yang sama
print(a is not x)
# akan menampilkan nilai False dikarenakan a dan x merujuk ke objek yang sama
x = 3
print(type(x) is int)


# Membership Operator
x = ["Ani", "Budi", "Cici"]
y = "Cici"
z = "Dodi"
print(y in x) # akan menampilkan nilai   True
print(z in x) # akan menampilkan nilai  False
print(y not in x) # akan menampilkan nilai  False
print(z not in x) # akan menampilkan nilai   True


# Latihan
total_harga = 150000
potongan_harga = 0.3
pajak = 0.1 # pajak dalam persen ~ 10%
harga_bayar = 1 - potongan_harga # baris pertama
harga_bayar *= total_harga # baris kedua
pajak_bayar = pajak * harga_bayar # baris ketiga
harga_bayar += pajak_bayar # baris ke-4
print("Kode awal - harga_bayar=", harga_bayar)
# Penyederhanaan baris kode dengan menerapkan prioritas operator
total_harga = 150000
potongan_harga = 0.3
pajak = 0.1 # pajak dalam persen ~ 10%
harga_bayar = (1 - potongan_harga) * total_harga #baris pertama 
harga_bayar += harga_bayar * pajak # baris kedua
print("Penyederhanaan kode - harga_bayar=", harga_bayar)


# Statement if
x = 4
if x % 2 == 0: # jika sisa bagi x dengan 2 sama dengan 0
    print("x habis dibagi dua") # statemen aksi lebih menjorok ke dalam
# Statement if ... elif ... else
x = 7
if x % 2 == 0: # jika sisa bagi x dengan 2 sama dengan 0
    print("x habis dibagi dua")
elif x % 3 == 0: # jika sisa bagi x dengan 3 sama dengan 0
    print("x habis dibagi tiga")
elif x % 5 == 0: # jika sisa bagi x dengan 5 sama dengan 0
    print("x habis dibagi lima")
else:
    print("x tidak habis dibagi dua, tiga ataupun lima")
    
jam = 13
if jam >= 5 and jam < 12: # selama jam di antara 5 s.d. 12
    print("Selamat pagi!")
elif jam >= 12 and jam < 17: # selama jam di antara 12 s.d. 17
    print("Selamat siang!")
elif jam >= 17 and jam < 19: # selama jam di antara 17 s.d. 19
    print("Selamat sore!")
else: # selain kondisi di atas
    print("Selamat malam!")
    
# Tugas Praktek
tagihan_ke = 'Mr. Yoyo'
warehousing = { 'harga_harian': 1000000, 'total_hari':15 } 
cleansing = { 'harga_harian': 1500000, 'total_hari':10 } 
integration = { 'harga_harian':2000000, 'total_hari':15 } 
transform = { 'harga_harian':2500000, 'total_hari':10 }
sub_warehousing = warehousing['harga_harian'] * warehousing['total_hari'] 
sub_cleansing = cleansing['harga_harian'] * cleansing['total_hari'] 
sub_integration = integration['harga_harian'] * integration['total_hari'] 
sub_transform = transform['harga_harian'] * transform['total_hari']
total_harga = sub_warehousing + sub_cleansing + sub_integration + sub_transform
print("Tagihan kepada:")
print(tagihan_ke)
print("Selamat pagi, anda harus membayar tagihan sebesar:") 
print(total_harga)


# Primitive Loops - While
# contoh 1
tagihan = [50000, 75000, 125000, 300000, 200000]
# Tanpa menggunakan while loop
total_tagihan = tagihan[0] + tagihan[1] + tagihan[2] + tagihan[3] + tagihan[4]
print(total_tagihan)
# Dengan menggunakan while loop
i=0 # sebuah variabel untuk mengakses setiap elemen tagihan satu per satu
jumlah_tagihan = len(tagihan) # panjang (jumlah elemen dalam) list tagihan
total_tagihan = 0 # mula-mula, set total_tagihan ke 0
while i < jumlah_tagihan: # selama nilai i kurang dari jumlah_tagihan
    total_tagihan += tagihan[i] # tambahkan tagihan[i] ke total_tagihan
    i += 1 # tambahkan nilai i dengan 1 untuk memproses tagihan selanjutnya.
print(total_tagihan)

# contoh 2 
tagihan = [50000, 75000, -150000, 125000, 300000, -50000, 200000]
i = 0
jumlah_tagihan = len(tagihan)
total_tagihan = 0
while i < jumlah_tagihan:
    # jika terdapat tagihan ke-i yang bernilai minus (di bawah nol),
    # pengulangan akan dihentikan
    if tagihan[i] < 0:
        total_tagihan = -1
        print("terdapat angka minus dalam tagihan, perhitungan dihentikan!")
        break
    total_tagihan += tagihan[i]
    i += 1
print(total_tagihan)

# contoh 3
tagihan = [50000, 75000, -150000, 125000, 300000, -50000, 200000]
i = 0
jumlah_tagihan = len(tagihan)
total_tagihan = 0
while i < jumlah_tagihan:
    # jika terdapat tagihan ke-i yang bernilai minus (di bawah nol),
    # abaikan tagihan ke-i dan lanjutkan ke tagihan berikutnya
    if tagihan[i] < 0:
        i+= 1
        continue
    total_tagihan += tagihan[i]
    i += 1
print(total_tagihan)