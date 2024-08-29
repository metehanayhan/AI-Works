# Veri Görselleştirme - Metehan Ayhan

---

### Grafik Türleri ve Kullanım Alanları

1. **Bar Chart (Çubuk Grafik)**
    - **Ne Zaman Kullanılır:** Kategoriler arasında nicelikleri karşılaştırmak için.
    - **Kullanım Örneği:** Farklı ürünlerin satışlarını karşılaştırmak.
2. **Line Chart (Çizgi Grafik)**
    - **Ne Zaman Kullanılır:** Zaman içinde eğilimleri göstermek için.
    - **Kullanım Örneği:** Bir yıl boyunca web sitesi trafiğindeki artışı göstermek.
3. **Pie Chart (Pasta Grafik)**
    - **Ne Zaman Kullanılır:** Oranları ve yüzdeleri vurgulamak için.
    - **Kullanım Örneği:** Bir bütçedeki harcama dağılımını illüstre etmek.
4. **Scatter Plot (Saçılım Grafiği)**
    - **Ne Zaman Kullanılır:** Değişkenler arasındaki ilişkileri temsil etmek için.
    - **Kullanım Örneği:** Pazarlama harcamaları ile yatırım getirisi arasındaki korelasyonları belirlemek.
5. **Histogram**
    - **Ne Zaman Kullanılır:** Verinin dağılımını görselleştirmek için.
    - **Kullanım Örneği:** Anket katılımcılarının yaş dağılımını göstermek.
6. **Radar Chart (Radar Grafiği)**
    - **Ne Zaman Kullanılır:** Farklı boyutlar arasındaki kategorileri karşılaştırmak için.
    - **Kullanım Örneği:** Bir ürünün farklı bölgelerdeki performansını değerlendirmek.
7. **Map (Harita)**
    - **Ne Zaman Kullanılır:** Coğrafi veriyi görselleştirmek için.
    - **Kullanım Örneği:** Bir harita üzerinde bölgesel satış performansını göstermek.
8. **Heatmap (Isı Haritası)**
    - **Ne Zaman Kullanılır:** Özellikle büyük veri setlerinde veri yoğunluğu ve desenleri görselleştirmek için.
    - **Kullanım Örneği:** Bir alışveriş merkezinde müşteri aktivitelerinin yoğun olduğu noktaları belirlemek.
9. **Bubble Chart (Balon Grafik)**
    - **Ne Zaman Kullanılır:** Üç boyutlu veriyi temsil etmek için.
    - **Kullanım Örneği:** Üç boyutta gelir, maliyet ve kârı karşılaştırmak.
10. **Donut Chart (Halka Grafik)**
    - **Ne Zaman Kullanılır:** Bir bütün içindeki belirli bölümleri vurgulamak için.
    - **Kullanım Örneği:** Pazarlama harcamalarının dağılımını göstermek.

### Matplotlib ile Görselleştirme

- **Matplotlib:** Python'da en popüler grafik çizim paketi.
- **Görselleştirme:** Keşifsel analizde ve bulguları iletişimde önemli bir adımdır.
- **Oluşturacağımız Grafik Türleri:**
    - Çizgi Grafik
    - Saçılım Grafiği
    - Histogram
    - Kutu Grafik

### Matplotlib ile Grafik Çizimi

**Matplotlib** Python programlama dili ile birlikte kullanılan güçlü bir kütüphanedir ve veri görselleştirme işlemlerinde sıklıkla tercih edilir. Çeşitli grafik türlerini kullanarak veriyi daha anlamlı hale getirmek ve analiz sonuçlarını görselleştirmek mümkündür.

1. **Çizgi Grafik (Line Plot):**
    - Zaman içindeki değişiklikleri göstermek için idealdir.
    - Örnek: Günlük sıcaklık değişimlerini göstermek.
2. **Saçılım Grafiği (Scatter Plot):**
    - İki değişken arasındaki ilişkiyi göstermek için kullanılır.
    - Örnek: Aylık reklam harcamaları ve satışlar arasındaki ilişki.
3. **Histogram:**
    - Verinin dağılımını görselleştirmek için kullanılır.
    - Örnek: Bir sınıftaki öğrencilerin not dağılımı.
4. **Kutu Grafik (Box Plot):**
    - Verinin özet istatistiklerini (minimum, birinci çeyrek, medyan, üçüncü çeyrek, maksimum) göstermek için kullanılır.
    - Örnek: Farklı bölgelerdeki ev fiyatlarının dağılımı.

### Uygulamalı Örnekler

Matplotlib kullanarak yukarıda bahsedilen grafik türlerini oluşturabiliriz. İşte bazı temel örnekler:

**Çizgi Grafik Örneği:**

```python
import matplotlib.pyplot as plt

# Örnek veri
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y)
plt.title('Çizgi Grafik Örneği')
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.show()
```

**Saçılım Grafiği Örneği:**

```python

import matplotlib.pyplot as plt

# Örnek veri
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.scatter(x, y)
plt.title('Saçılım Grafiği Örneği')
plt.xlabel('X Ekseni')
plt.ylabel('Y Ekseni')
plt.show()

```

**Histogram Örneği:**

```python

import matplotlib.pyplot as plt

# Örnek veri
veri = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

plt.hist(veri, bins=5)
plt.title('Histogram Örneği')
plt.xlabel('Değer')
plt.ylabel('Frekans')
plt.show()

```

**Kutu Grafik Örneği:**

```python

import matplotlib.pyplot as plt

# Örnek veri
veri = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5]

plt.boxplot(veri)
plt.title('Kutu Grafik Örneği')
plt.ylabel('Değer')
plt.show()

```

---

### Keşifsel Veri Analizi (EDA) Adımları

EDA, veriyi anlamak ve daha derin analizler için hazırlık yapmak amacıyla gerçekleştirilir. İşte EDA sürecindeki temel adımlar:

1. **Veriyi Anlamak:**
    - Verinin kaynağı, yapısı ve türleri hakkında bilgi edinmek.
    - Verinin genel istatistiklerini hesaplamak.
2. **Veri Görselleştirme:**
    - Veriyi grafikler ve çizelgeler kullanarak görselleştirmek.
    - Verideki trendleri, kalıpları ve anormallikleri belirlemek.
3. **Hipotez Testi:**
    - Verideki ilişki ve desenleri test etmek için istatistiksel yöntemler kullanmak.
    - Hipotezleri doğrulamak veya reddetmek için analitik araçlar kullanmak.
4. **Sonuçları İletmek:**
    - Elde edilen bulguları görseller ve raporlar aracılığıyla paylaşmak.
    - Bulguları anlaşılır ve etkili bir şekilde iletmek.

### İleri Düzey Teknikler ve Araçlar

Daha ileri düzey analizler ve görselleştirmeler için çeşitli Python kütüphaneleri kullanılabilir:

1. **Seaborn:**
    - Matplotlib üzerine inşa edilmiş bir kütüphane olup, istatistiksel grafikler oluşturmak için kullanılır.
    - Daha estetik ve kolay anlaşılır grafikler sağlar.
2. **Pandas:**
    - Veri manipülasyonu ve analizi için güçlü bir araçtır.
    - Veri çerçeveleri (DataFrames) kullanarak veriyi organize eder ve analiz eder.
3. **Plotly:**
    - İnteraktif ve dinamik grafikler oluşturmak için kullanılır.
    - Web tabanlı görselleştirmeler için idealdir.
4. **Altair:**
    - Bildirimsel bir görselleştirme kütüphanesi olup, kullanıcı dostu ve etkileşimli grafikler oluşturmayı sağlar.

---

### Uygulamalı Örnek: EDA ve Görselleştirme

### 1. Veri Yükleme ve Temel İstatistikler

Öncelikle, Iris veri setini yükleyelim ve temel istatistiklerini inceleyelim.

```python

import pandas as pd

# Iris veri setini yükleme
iris = pd.read_csv('iris.csv')

# Veri setinin ilk 5 satırını görüntüleme
print(iris.head())

# Temel istatistikleri görüntüleme
print(iris.describe())

```

### 2. Veri Görselleştirme

### Histogram

Her bir özelliğin dağılımını göstermek için histogramlar oluşturabiliriz.

```python

import matplotlib.pyplot as plt

iris.hist(bins=20, figsize=(10, 8))
plt.suptitle('Iris Veri Setinin Histogramı')
plt.show()

```

### Kutu Grafik (Box Plot)

Her bir özelliğin özet istatistiklerini göstermek için kutu grafikler oluşturabiliriz.

```python

iris.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False, figsize=(10, 8))
plt.suptitle('Iris Veri Setinin Kutu Grafik Analizi')
plt.show()

```

### Çift Değişkenli Analiz: Saçılım Grafiği (Scatter Plot)

Özellikler arasındaki ilişkileri görmek için saçılım grafikleri kullanabiliriz.

```python

import seaborn as sns

sns.pairplot(iris, hue='species', height=2.5)
plt.suptitle('Iris Veri Setinin Çift Değişkenli Analizi', y=1.02)
plt.show()

```

### 3. İleri Düzey Analizler

İleri düzey analizler için Seaborn ve Plotly gibi kütüphaneleri kullanabiliriz.

### Seaborn ile Isı Haritası (Heatmap)

Özellikler arasındaki korelasyonu görmek için ısı haritası oluşturabiliriz.

```python

plt.figure(figsize=(10, 8))
sns.heatmap(iris.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Iris Veri Setinin Korelasyon Matrisi')
plt.show()

```

### Plotly ile İnteraktif Grafikler

Plotly kullanarak interaktif grafikler oluşturabiliriz. İşte bir örnek:

```python

import plotly.express as px

fig = px.scatter(iris, x='sepal_length', y='sepal_width', color='species', title='Iris Veri Setinin İnteraktif Saçılım Grafiği')
fig.show()

```

### Keşifsel Veri Analizi (EDA) Adımları

EDA sürecinde dikkat etmemiz gereken adımlar şunlardır:

1. **Veriyi Anlamak:**
    - Verinin kaynağı, yapısı ve türleri hakkında bilgi edinmek.
    - Verinin genel istatistiklerini hesaplamak.
2. **Veri Görselleştirme:**
    - Veriyi grafikler ve çizelgeler kullanarak görselleştirmek.
    - Verideki trendleri, kalıpları ve anormallikleri belirlemek.
3. **Hipotez Testi:**
    - Verideki ilişki ve desenleri test etmek için istatistiksel yöntemler kullanmak.
    - Hipotezleri doğrulamak veya reddetmek için analitik araçlar kullanmak.
4. **Sonuçları İletmek:**
    - Elde edilen bulguları görseller ve raporlar aracılığıyla paylaşmak.
    - Bulguları anlaşılır ve etkili bir şekilde iletmek.

### Özet

Veri yönetimi, EDA ve veri görselleştirme konuları, veri biliminde önemli bir yer tutar. İyi bir veri yönetimi sistemi ile verilerinizi düzenli ve analiz edilebilir hale getirebilirsiniz. EDA süreci, verilerinizi anlamanızı ve analizlerinizi daha etkili hale getirmenizi sağlar. Veri görselleştirme teknikleri ise bulgularınızı daha anlaşılır ve etkili bir şekilde iletmenizi sağlar.

---

## Pandas'a Giriş

Öncelikle, Pandas kütüphanesini yükleyip başlatmamız gerekiyor:

```python
import pandas as pd
```

### Veri Oluşturma ve Yükleme

### DataFrame Oluşturma

```python
data = {
    'isim': ['Ali', 'Ayşe', 'Fatma', 'Mehmet'],
    'yaş': [23, 45, 34, 22],
    'maas': [5000, 7000, 8000, 6500]
}

df = pd.DataFrame(data)
print(df)
```

### CSV Dosyasından Veri Yükleme

```python
df = pd.read_csv('dosya_adı.csv')
print(df.head())  # İlk 5 satırı görüntüle
```

### Veri İnceleme

### Temel İnceleme Fonksiyonları

```python
print(df.head())  # İlk 5 satır
print(df.tail())  # Son 5 satır
print(df.info())  # Genel bilgi
print(df.describe())  # Temel istatistikler
print(df.shape)  # Satır ve sütun sayısı
print(df.columns)  # Sütun adları
print(df.index)  # DataFrame'in index bilgisi
```

### Veri Seçimi ve Filtreleme

### Sütun Seçme

```python
print(df['isim'])  # Tek sütun
print(df[['isim', 'maas']])  # Birden fazla sütun
```

### Satır Seçme

```python
print(df.iloc[0])  # İlk satır (index ile)
print(df.loc[0])  # İlk satır (index adı ile)
print(df.iloc[0:2])  # İlk iki satır (dilimleme)
```

### Koşullu Seçim

```python
print(df[df['yaş'] > 30])  # Yaşı 30'dan büyük olanlar
print(df[(df['yaş'] > 30) & (df['maas'] > 6000)])  # Yaşı 30'dan büyük ve maaşı 6000'den fazla olanlar
```

### Veri Manipülasyonu

### Yeni Sütun Eklemek

```python
df['yeni_sütun'] = df['maas'] * 0.1  # Maaşın %10'unu yeni sütun olarak ekle
print(df)
```

### Sütun Adını Değiştirmek

```python
df.rename(columns={'isim': 'ad', 'yaş': 'yas'}, inplace=True)
print(df)
```

### Sütun veya Satır Silmek

```python
df.drop('yeni_sütun', axis=1, inplace=True)  # Sütun silme
df.drop([0, 1], axis=0, inplace=True)  # Satır silme
print(df)
```

### Veri Analizi

### Gruplama ve Toplama

```python
grouped = df.groupby('yas').mean()  # Yaşa göre grupla ve ortalamalarını al
print(grouped)
```

### Pivot Tablo

```python
pivot = df.pivot_table(values='maas', index='yas', columns='isim', aggfunc='mean')
print(pivot)
```

### Eksik Verilerle Çalışma

### Eksik Değerleri Görüntüleme

```python
print(df.isnull().sum())  # Her sütundaki eksik değer sayısı
```

### Eksik Değerleri Doldurma

```python
df.fillna(0, inplace=True)  # Eksik değerleri 0 ile doldur
```

### Eksik Değerleri Silme

```python
df.dropna(inplace=True)  # Eksik değer içeren satırları sil
```

### Veri Kaydetme

### CSV Dosyasına Kaydetme

```python
df.to_csv('yeni_dosya.csv', index=False)  # DataFrame'i CSV dosyasına kaydet
```

---

### NumPy'ye Giriş

Öncelikle, NumPy kütüphanesini yükleyip başlatmamız gerekiyor:

```python
import numpy as np
```

### NumPy Dizileri (Arrays) Oluşturma

### Listeyi Diziye Dönüştürme

```python
my_list = [1, 2, 3, 4, 5]
my_array = np.array(my_list)
print(my_array)
```

### Çok Boyutlu Dizi Oluşturma

```python
my_2d_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
my_2d_array = np.array(my_2d_list)
print(my_2d_array)
```

### Arange Fonksiyonu

```python
array = np.arange(0, 10, 2)  # 0'dan 10'a kadar 2 artışlarla dizi oluşturur
print(array)
```

### Linspace Fonksiyonu

```python
array = np.linspace(0, 10, 5)  # 0 ile 10 arasında 5 eşit parçaya böler
print(array)
```

### Rastgele Dizi Oluşturma

```python
random_array = np.random.rand(3, 3)  # 3x3 boyutunda rastgele sayılardan oluşan bir dizi
print(random_array)
```

### Diziler Üzerinde Temel İşlemler

### Dizi Elemanlarına Erişim

```python
print(my_array[0])  # İlk eleman
print(my_2d_array[0, 1])  # İlk satırın ikinci elemanı
```

### Dizi Elemanlarını Değiştirme

```python
my_array[0] = 10
print(my_array)
```

### Dilimleme (Slicing)

```python
print(my_array[1:4])  # 1. indexten 4. indexe kadar olan elemanlar
print(my_2d_array[:2, 1:])  # İlk iki satırın ikinci sütundan sonraki elemanları
```

### Diziler Üzerinde Matematiksel İşlemler

### Eleman Bazlı İşlemler

```python
array = np.array([1, 2, 3, 4])
print(array + 2)  # Her elemana 2 ekler
print(array * 2)  # Her elemanı 2 ile çarpar
print(array ** 2)  # Her elemanın karesini alır
```

### Dizi Fonksiyonları

```python
print(np.sum(array))  # Elemanların toplamı
print(np.mean(array))  # Elemanların ortalaması
print(np.max(array))  # En büyük eleman
print(np.min(array))  # En küçük eleman
print(np.std(array))  # Standart sapma
```

### Dizileri Birleştirme ve Bölme

### Dizileri Birleştirme

```python
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])
combined = np.concatenate((array1, array2))
print(combined)
```

### Dizileri Yeniden Şekillendirme

```python
array = np.arange(9)
reshaped = array.reshape(3, 3)  # 1 boyutlu diziyi 3x3 boyutunda diziye çevirir
print(reshaped)
```

### Boolean İndeksleme

### Koşullu Seçim

```python
array = np.array([1, 2, 3, 4, 5])
print(array[array > 2])  # 2'den büyük elemanları seçer
```

### NumPy'nin İleri Seviye Özellikleri

### Matris Çarpımı

```python
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
product = np.dot(matrix1, matrix2)
print(product)
```

### Transpoz Alma

```python
matrix = np.array([[1, 2], [3, 4]])
transpose = matrix.T
print(transpose)
```

---

# Matplotlib

Matplotlib, Python'da veri görselleştirmek için kullanılan güçlü bir kütüphanedir. Kütüphane, çeşitli grafikler ve çizimler oluşturmak için birçok fonksiyon sunar. Aşağıda, `matplotlib` kütüphanesinin temel özelliklerini gösteren bir örnek kod yer almaktadır. Bu kod, farklı türdeki grafiklerin nasıl oluşturulabileceğini ve bazı önemli komutların nasıl kullanılacağını açıklar.

```python
import matplotlib.pyplot as plt  # matplotlib'in pyplot modülünü içe aktar
import numpy as np  # NumPy'yi içe aktar, veri oluşturmak için

# Veriyi hazırlama
x = np.linspace(0, 10, 100)  # 0 ile 10 arasında 100 eşit aralıklı nokta oluştur
y = np.sin(x)  # x'in sinüs değerlerini hesapla
y2 = np.cos(x)  # x'in kosinüs değerlerini hesapla

# 1. Basit Çizgi Grafiği
plt.figure(figsize=(10, 6))  # Grafik boyutunu ayarla (10x6 inç)
plt.subplot(2, 2, 1)  # 2x2 ızgara düzeni, 1. konum

plt.plot(x, y,              # x ve y verilerini çiz
         label='Sin(x)',    # Grafikte gösterilecek açıklama
         color='blue',      # Çizgi rengini ayarla
         linestyle='-',     # Çizgi stilini belirle (düz çizgi)
         marker='o')        # Noktaların şekli (daire)

plt.title('Çizgi Grafiği')  # Grafik başlığı
plt.xlabel('X ekseni')      # X ekseninin etiketini ayarla
plt.ylabel('Y ekseni')      # Y ekseninin etiketini ayarla
plt.legend()                # Grafikteki açıklamaları göster
plt.grid(True)              # Izgara çizgilerini ekle

# 2. Dağılım Grafiği (Scatter Plot)
plt.subplot(2, 2, 2)  # 2x2 ızgara düzeni, 2. konum

plt.scatter(x, y,               # x ve y verilerini noktalar halinde göster
            color='red',        # Noktaların rengini ayarla
            marker='x',         # Noktaların şeklini belirle (çarpı)
            s=50)              # Noktaların boyutunu ayarla

plt.title('Dağılım Grafiği')   # Grafik başlığı
plt.xlabel('X ekseni')         # X ekseninin etiketini ayarla
plt.ylabel('Y ekseni')         # Y ekseninin etiketini ayarla
plt.grid(True)                 # Izgara çizgilerini ekle

# 3. Bar Grafiği (Bar Chart)
categories = ['A', 'B', 'C', 'D']  # Kategorilerin isimleri
values = [4, 7, 1, 8]             # Her bir kategoriye karşılık gelen değerler
plt.subplot(2, 2, 3)  # 2x2 ızgara düzeni, 3. konum

plt.bar(categories, values,   # Kategoriler ve değerler ile bar grafiği çiz
         color='green')       # Barların rengini ayarla

plt.title('Bar Grafiği')      # Grafik başlığı
plt.xlabel('Kategori')       # X ekseninin etiketini ayarla
plt.ylabel('Değer')          # Y ekseninin etiketini ayarla

# 4. Histogram
data = np.random.randn(1000)  # 1000 rastgele veri noktası oluştur
plt.subplot(2, 2, 4)  # 2x2 ızgara düzeni, 4. konum

plt.hist(data,            # Veri ile histogram çiz
         bins=30,         # Histogramdaki aralık sayısı
         color='purple',  # Histogramın rengini ayarla
         edgecolor='black')  # Histogram barlarının kenar çizgilerini ayarla

plt.title('Histogram')    # Grafik başlığı
plt.xlabel('Değerler')    # X ekseninin etiketini ayarla
plt.ylabel('Frekans')     # Y ekseninin etiketini ayarla

# Grafiklerin düzenini iyileştirme
plt.tight_layout()  # Grafiklerin düzenini ve boşlukları ayarla

# Grafiklerin gösterimi
plt.show()  # Grafikleri ekranda göster
```

### Ayrıntılı Açıklamalar

### 1. Veriyi Hazırlama

- `np.linspace()`: Belirli bir aralıkta eşit aralıklı sayılar oluşturur. Burada 0 ile 10 arasında 100 eşit aralıklı nokta oluşturuyoruz.
- `np.sin()`, `np.cos()`: Bu fonksiyonlar, x verisinin sinüs ve kosinüs değerlerini hesaplar. Bu, çizgi grafiğinde kullanılacak veriyi sağlar.

### 2. Grafik Oluşturma

- `plt.figure()`: Grafik boyutunu belirler. `figsize` parametresi, grafiğin genişliğini ve yüksekliğini inç cinsinden ayarlar.
- `plt.subplot()`: Grafik düzenini tanımlar. Burada, 2x2 ızgara düzeninde her bir grafiğin konumunu belirleriz. (Örneğin, `plt.subplot(2, 2, 1)` 2x2 ızgara düzeninin ilk konumunu belirtir.)

### 3. Çizgi Grafiği

- `plt.plot()`: Çizgi grafiği oluşturur. `color`, `linestyle`, ve `marker` gibi parametrelerle grafiğin görünümünü özelleştirebilirsiniz.
- `plt.legend()`: Grafikteki verileri açıklamak için açıklamalar ekler.

### 4. Dağılım Grafiği (Scatter Plot)

- `plt.scatter()`: Noktalar halinde veri noktalarını gösterir. `color`, `marker`, ve `s` ile nokta özelliklerini ayarlarsınız.

### 5. Bar Grafiği (Bar Chart)

- `plt.bar()`: Kategorilere göre veri değerlerini barlarla gösterir. Kategorileri ve değerleri alır.

### 6. Histogram

- `plt.hist()`: Verinin frekans dağılımını gösterir. `bins` parametresi, veri aralıklarının sayısını belirler.

### 7. Grafik Düzenini İyileştirme

- `plt.tight_layout()`: Grafiklerin düzenini otomatik olarak ayarlar ve grafikleri daha düzenli hale getirir.

### 8. Grafiklerin Gösterimi

- `plt.show()`: Grafikleri ekranda gösterir.

---