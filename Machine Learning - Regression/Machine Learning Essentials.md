# Makine Öğrenmesi - Metehan Ayhan

# Makine Öğrenmesi Nedir?

Makine öğrenmesi (ML), bilgisayarların verilerden öğrenerek ve deneyimlerinden faydalanarak belirli görevleri otomatik olarak yerine getirmelerini sağlayan bir yapay zeka (AI) dalıdır. ML algoritmaları, belirli bir görev için programlanmamış olup, bunun yerine büyük miktarda veriyi analiz ederek ve bu verilerden çıkarımlar yaparak kararlar alır veya tahminlerde bulunur.

### Makine Öğrenmesi Kategorileri

Makine öğrenmesi üç ana kategoriye ayrılır:

1. **Denetimli Öğrenme (Supervised Learning):**
    - **Sınıflandırma (Classification):** Verilerin belirli kategorilere ayrılması işlemidir. Örneğin, e-postaların spam veya spam olmayan olarak sınıflandırılması.
    - **Regresyon (Regression):** Sürekli bir değerin tahmin edilmesidir. Örneğin, ev fiyatlarının tahmini.
2. **Denetimsiz Öğrenme (Unsupervised Learning):**
    - **Kümeleme (Clustering):** Verilerin benzerliklerine göre gruplandırılmasıdır. Örneğin, müşteri segmentasyonu.
    - **Boyut Azaltma (Dimensionality Reduction):** Verilerin daha düşük boyutlu bir temsile indirgenmesi. Örneğin, veri görselleştirme için PCA (Principal Component Analysis).
3. **Pekiştirmeli Öğrenme (Reinforcement Learning):** Ajanların (agents) bir ortamda eylemler gerçekleştirerek ve bu eylemlerden geri bildirim alarak en iyi stratejiyi öğrenmeleridir. Örneğin, oyun oynayan bir yapay zeka.

### Detaylı Algoritmalar ve Kullanım Alanları

### 1. Denetimli Öğrenme Algoritmaları:

- **Naive Bayes Sınıflandırıcı:** Basit, fakat güçlü bir sınıflandırma algoritmasıdır. Özellikle metin sınıflandırma (e-posta spam tespiti) gibi görevlerde kullanılır.
- **Karar Ağaçları (Decision Trees):** Verileri dallara ayırarak karar vermeyi sağlayan algoritmalardır.
- **Destek Vektör Makineleri (Support Vector Machines):** Verileri sınıflandırmak için optimal bir ayırıcı hiper düzlem bulan algoritmalardır.
- **K-En Yakın Komşu (K-Nearest Neighbors):** Yeni bir verinin sınıfını, en yakınındaki K komşusunun sınıfına bakarak belirleyen algoritmalardır.
- **Lineer Regresyon:** Sürekli bir hedef değeri tahmin etmek için kullanılan basit bir regresyon yöntemidir.
- **Lasso ve Ridge Regresyon:** Lineer regresyonun geliştirilmiş versiyonları olup, overfitting'i (aşırı uyumu) önlemeye çalışırlar.

### 2. Denetimsiz Öğrenme Algoritmaları:

- **K-Means Kümeleme:** Verileri K sayıda kümeye ayıran popüler bir kümeleme algoritmasıdır.
- **DBSCAN Kümeleme:** Gürültülü verilere karşı dayanıklı, yoğunluk tabanlı bir kümeleme yöntemidir.
- **Principal Component Analysis (PCA):** Boyut azaltma için kullanılan, verilerdeki en önemli bileşenleri bulmayı amaçlayan bir tekniktir.

### 3. Pekiştirmeli Öğrenme Algoritmaları:

- **Q-Öğrenme (Q-Learning):** Ortamdan aldığı ödüllere göre bir politika öğrenmeye çalışan model-free bir algoritmadır.
- **TD-Öğrenme (Temporal Difference Learning):** Gelecekteki ödülleri tahmin ederek politikalar geliştiren bir algoritmadır.

### Uygulama Alanları

- **Denetimli Öğrenme:** Finansal tahminler, tıbbi teşhisler, pazarlama analizi, spam tespiti.
- **Denetimsiz Öğrenme:** Müşteri segmentasyonu, pazar analizi, veri görselleştirme.
- **Pekiştirmeli Öğrenme:** Oyun yapay zekası, robotik, otonom araçlar, borsa ticareti.

---

Regresyon, sürekli bir bağımlı değişkenin (hedef veya çıktı değişkeni) bir veya daha fazla bağımsız değişken (girdi değişkenleri) ile olan ilişkisini modellemek için kullanılan bir makine öğrenmesi tekniğidir. En yaygın kullanılan regresyon türü lineer regresyondur.

### Lineer Regresyon Nedir?

Lineer regresyon, bağımlı değişken ile bir veya daha fazla bağımsız değişken arasındaki doğrusal ilişkiyi modellemek için kullanılır. Tek değişkenli lineer regresyon (simple linear regression) yalnızca bir bağımsız değişkeni içerir, çoklu lineer regresyon (multiple linear regression) ise birden fazla bağımsız değişkeni içerir.

Lineer regresyonun matematiksel formülü:
y=β0+β1x1+β2x2+…+βnxny = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_n x_ny=β0​+β1​x1​+β2​x2​+…+βn​xn​

- yyy: Bağımlı değişken
- x1,x2,…,xnx_1, x_2, \ldots, x_nx1​,x2​,…,xn​: Bağımsız değişkenler
- β0\beta_0β0​: Y-intercept (kesim noktası)
- β1,β2,…,βn\beta_1, \beta_2, \ldots, \beta_nβ1​,β2​,…,βn​: Bağımsız değişkenlerin katsayıları (eğim)

### Python ile Lineer Regresyon Uygulaması

Python'da `scikit-learn` kütüphanesi kullanılarak lineer regresyon kolayca uygulanabilir. Örnek bir uygulama üzerinden adım adım açıklayalım.

### Adım 1: Gerekli Kütüphaneleri İndirme

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Adım 2: Veri Hazırlığı

Örnek olarak, basit bir veri kümesi oluşturalım.

```python
# Örnek veri seti
X = np.array([1, 2, 4, 3, 5]).reshape(-1, 1)  # Bağımsız değişken
y = np.array([1, 3, 3, 2, 5])  # Bağımlı değişken
```

### Adım 3: Veriyi Eğitim ve Test Setlerine Ayırma

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Adım 4: Modeli Eğitme

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Adım 5: Tahmin Yapma

```python
y_pred = model.predict(X_test)
```

### Adım 6: Model Performansını Değerlendirme

```python
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# R-squared (R²) Skoru
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
```

### Adım 7: Sonuçları Görselleştirme

```python
plt.scatter(X, y, color='blue')  # Veri noktaları
plt.plot(X_test, y_pred, color='red', linewidth=2)  # Regresyon doğrusu
plt.xlabel('Bağımsız Değişken')
plt.ylabel('Bağımlı Değişken')
plt.title('Lineer Regresyon')
plt.show()
```

Bu adımlar tamamlandığında, Python ile basit bir lineer regresyon modeli oluşturmuş ve eğitmiş olacaksınız. Regresyon modeli, bağımlı değişkenin bağımsız değişkenler tarafından nasıl etkilendiğini anlamanızı sağlar. Daha karmaşık veri kümeleri ve daha fazla bağımsız değişken içeren modellerde de benzer adımlar uygulanabilir.

---

### Çoklu Lineer Regresyon

### Adım 1: Gerekli Kütüphaneleri İndirme

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

### Adım 2: Veri Hazırlığı

Örnek bir veri kümesi oluşturalım. Bu veri kümesinde `X1` ve `X2` bağımsız değişkenler, `y` ise bağımlı değişken olacak.

```python
# Örnek veri seti
data = {
    'X1': [1, 2, 4, 3, 5],
    'X2': [2, 3, 4, 3, 2],
    'y': [1, 3, 3, 2, 5]
}

df = pd.DataFrame(data)

# Bağımsız değişkenler ve bağımlı değişken
X = df[['X1', 'X2']].values
y = df['y'].values
```

### Adım 3: Veriyi Eğitim ve Test Setlerine Ayırma

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Adım 4: Modeli Eğitme

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### Adım 5: Tahmin Yapma

```python
y_pred = model.predict(X_test)
```

### Adım 6: Model Performansını Değerlendirme

```python
# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
# R-squared (R²) Skoru
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
```

### Adım 7: Sonuçları Görselleştirme (Opsiyonel)

Çoklu regresyon modelinde, sonuçların görselleştirilmesi zor olabilir çünkü bağımsız değişkenler iki boyutludur. Ancak, tahmin edilen değerlerle gerçek değerleri karşılaştırarak bir grafik oluşturabilirsiniz.

```python
plt.scatter(range(len(y_test)), y_test, color='blue', label='Gerçek Değerler')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Tahmin Edilen Değerler')
plt.xlabel('Örnek Numarası')
plt.ylabel('Bağımlı Değişken')
plt.title('Çoklu Lineer Regresyon')
plt.legend()
plt.show()
```

Bu örnek, çoklu bağımsız değişkenleri (özellikleri) içeren bir lineer regresyon modelinin nasıl oluşturulacağını ve değerlendirileceğini göstermektedir. Bağımsız değişkenler, veri çerçevesinde her biri bir sütun olacak şekilde hazırlanmalı ve modelin eğitilmesi ve tahmin yapılması için `train_test_split` ile eğitim ve test setlerine ayrılmalıdır.

## Regresyon Türleri:

### 1. Doğrusal (Lineer) Regresyon

- **Tanım:** Bağımlı değişken ile bağımsız değişkenler arasındaki doğrusal ilişkiyi modelleyen regresyon türüdür.
- **Kullanım Alanları:** Ev fiyatlarının tahmini, şirketlerin kâr tahminleri, çalışanların maaş tahminleri.

### 2. Çoklu Doğrusal (Multiple Linear) Regresyon

- **Tanım:** Birden fazla bağımsız değişkenin bulunduğu doğrusal regresyon türüdür.
- **Kullanım Alanları:** Bir evin fiyatını tahmin ederken, metrekare, oda sayısı, lokasyon gibi birçok değişkenin etkisi.

### 3. Polinomial (Polynomial) Regresyon

- **Tanım:** Bağımsız değişkenler ve bağımlı değişken arasındaki ilişki doğrusal olmadığında kullanılır; ilişkiyi bir polinom fonksiyonu ile modellemeye çalışır.
- **Kullanım Alanları:** Araçların hız ve fren mesafesi arasındaki ilişki, nüfus büyüme tahminleri.

### 4. Lojistik (Logistic) Regresyon

- **Tanım:** Bağımlı değişkenin kategorik olduğu (genellikle ikili) durumlarda kullanılır. Sonuçlar, belirli bir olayın olma olasılığını tahmin eder.
- **Kullanım Alanları:** Hastalık teşhisi (hastanın hasta olup olmadığı), kredi başvurularının onaylanma olasılığı.

### 5. Ridge ve Lasso Regresyon

- **Tanım:** Lineer regresyonun düzenlileştirilmiş versiyonlarıdır. Ridge regresyon, katsayıları küçültmek için L2 normunu kullanırken, Lasso regresyon L1 normunu kullanır.
- **Kullanım Alanları:** Özellikle çok sayıda bağımsız değişkenin bulunduğu ve aşırı uyum (overfitting) riskinin olduğu durumlarda kullanılır.

### 6. Elastik Net (Elastic Net) Regresyon

- **Tanım:** Ridge ve Lasso regresyonun kombinasyonudur. Hem L1 hem de L2 normlarını kullanarak düzenlileştirme yapar.
- **Kullanım Alanları:** Yüksek boyutlu veri setlerinde (çok sayıda özellik) etkili bir düzenlileştirme yöntemidir.

### 7. Destek Vektör Regresyonu (Support Vector Regression, SVR)

- **Tanım:** Destek vektör makinelerinin (SVM) regresyon için uyarlanmış halidir. Veriyi bir düzlem üzerinde en iyi ayrıştıran hiperdüzlemi bulur.
- **Kullanım Alanları:** Karmaşık ve doğrusal olmayan veri setlerinde kullanılır; örneğin, finansal zaman serileri tahmini.

### Gerçek Hayat Problemlerinde Regresyon Kullanımı

### 1. Ev Fiyat Tahmini

- **Problem:** Bir evin satılacağı fiyatı tahmin etmek.
- **Kullanılan Regresyon Türü:** Çoklu doğrusal regresyon.
- **Bağımsız Değişkenler:** Metrekare, oda sayısı, yaş, konum, okula yakınlık.
- **Bağımlı Değişken:** Satış fiyatı.

### 2. Satış Tahminleri

- **Problem:** Bir mağazanın gelecek ayki satışlarını tahmin etmek.
- **Kullanılan Regresyon Türü:** Lineer regresyon veya SVR.
- **Bağımsız Değişkenler:** Geçmiş satış verileri, mevsimsel trendler, kampanyalar.
- **Bağımlı Değişken:** Gelecek ayki satış miktarı.

### 3. Hastalık Teşhisi

- **Problem:** Bir hastanın belirli bir hastalığa sahip olup olmadığını tahmin etmek.
- **Kullanılan Regresyon Türü:** Lojistik regresyon.
- **Bağımsız Değişkenler:** Hastanın yaşı, cinsiyeti, kan değerleri, semptomlar.
- **Bağımlı Değişken:** Hastalığın varlığı (1: var, 0: yok).

### 4. Finansal Tahminler

- **Problem:** Bir hisse senedinin gelecekteki fiyatını tahmin etmek.
- **Kullanılan Regresyon Türü:** Polinomial regresyon veya SVR.
- **Bağımsız Değişkenler:** Geçmiş fiyat verileri, işlem hacmi, piyasa endeksleri.
- **Bağımlı Değişken:** Hisse senedi fiyatı.

Regresyon analizleri, birçok farklı alanda kullanılabilir ve her bir türü, belirli bir veri yapısına veya probleme daha uygun olabilir. Regresyon türünün seçimi, veri yapısına ve problemin doğasına bağlı olarak yapılmalıdır.

---

## Classification (Sınıflandırma) Nedir?

Sınıflandırma, gözetimli öğrenme (supervised learning) kapsamında yer alan ve bir veri setini önceden belirlenmiş kategorilere ayırma işlemidir. Bu yöntem, giriş verilerini belirli etiketlere göre sınıflandırmak için kullanılır. Sınıflandırma, her bir veri noktasını sınıflandırma algoritmasının belirlediği bir kategoriye veya sınıfa atar.

### Temel Kavramlar ve Terimler

- **Bağımsız Değişkenler (Features):** Modelin öğrenmesi için kullanılan verinin özellikleridir. Örneğin, bir e-posta sınıflandırma işleminde e-postanın uzunluğu, başlık kelimeleri, vb. bağımsız değişkenler olabilir.
- **Bağımlı Değişken (Label):** Tahmin edilmesi gereken kategoridir. Örneğin, bir e-posta sınıflandırmasında bağımlı değişken, "spam" veya "spam değil" olabilir.
- **Eğitim Verisi (Training Data):** Modelin öğrenmesi için kullanılan etiketli veri setidir.
- **Test Verisi (Test Data):** Modelin performansını değerlendirmek için kullanılan etiketli veri setidir.

### Sınıflandırma Algoritmaları

1. **Naive Bayes**
    - **Tanım:** Bayes teoremine dayanan ve tüm özelliklerin birbirinden bağımsız olduğunu varsayan bir olasılık tabanlı algoritmadır.
    - **Kullanım Alanları:** E-posta spam filtresi, doküman sınıflandırma.
2. **Karar Ağaçları (Decision Trees)**
    - **Tanım:** Veriyi özelliklerine göre dallara ayırarak sınıflandıran ve ağaç yapısında kararlar veren algoritmalardır.
    - **Kullanım Alanları:** Müşteri segmentasyonu, kredi risk analizi.
3. **Destek Vektör Makineleri (Support Vector Machines, SVM)**
    - **Tanım:** Veriyi iki sınıfa ayıran en iyi hiper düzlemi bulan algoritmadır.
    - **Kullanım Alanları:** Yüz tanıma, metin sınıflandırma.
4. **K-Nearest Neighbors (KNN)**
    - **Tanım:** Bir veri noktasını en yakın komşularına (k komşusu) göre sınıflandıran algoritmadır.
    - **Kullanım Alanları:** Öneri sistemleri, anomali tespiti.
5. **Random Forest**
    - **Tanım:** Birden fazla karar ağacının bir araya gelmesiyle oluşturulan ve her bir ağacın sonucu üzerinden en sık görülen sınıfı tahmin eden algoritmadır.
    - **Kullanım Alanları:** Hastalık teşhisi, müşteri segmentasyonu.
6. **Lojistik Regresyon**
    - **Tanım:** Bağımlı değişkenin iki sınıftan biri olduğu durumlarda kullanılan doğrusal bir modeldir. Çıkış değeri bir olasılık olup 0 ile 1 arasında yer alır.
    - **Kullanım Alanları:** İkili sınıflandırma problemleri, kredi onaylama.

### Gerçek Hayat Problemlerinde Kullanımı

### 1. E-posta Spam Filtresi

- **Problem:** Gelen e-postaların spam olup olmadığını tespit etmek.
- **Kullanılan Algoritma:** Naive Bayes, SVM, Lojistik Regresyon.
- **Bağımsız Değişkenler:** E-posta başlığı, metin uzunluğu, kelime frekansları.
- **Bağımlı Değişken:** Spam (1), Spam Değil (0).

### 2. Tıbbi Teşhis

- **Problem:** Hastaların belirli bir hastalığa sahip olup olmadığını tespit etmek.
- **Kullanılan Algoritma:** Random Forest, Karar Ağaçları, SVM.
- **Bağımsız Değişkenler:** Hastanın yaş, cinsiyet, tıbbi test sonuçları.
- **Bağımlı Değişken:** Hastalık var (1), Hastalık yok (0).

### 3. Müşteri Segmentasyonu

- **Problem:** Müşterileri benzer özelliklere göre gruplara ayırmak.
- **Kullanılan Algoritma:** K-Means, Karar Ağaçları, Random Forest.
- **Bağımsız Değişkenler:** Satın alma geçmişi, demografik bilgiler, ürün tercihleri.
- **Bağımlı Değişken:** Müşteri segmenti.

### Sınıflandırma Adımları

1. **Veri Toplama:** Sınıflandırma işlemi için gerekli verileri toplama.
2. **Veri Ön İşleme:** Eksik değerlerin doldurulması, veri ölçekleme, kategorik değişkenlerin kodlanması.
3. **Özellik Seçimi:** Modelin performansını artırmak için en önemli özelliklerin seçimi.
4. **Model Eğitimi:** Eğitim verisi kullanılarak seçilen sınıflandırma algoritmasının eğitimi.
5. **Model Değerlendirme:** Test verisi ile modelin performansını değerlendirme (hata matrisi, doğruluk, F1 skoru, vb.).
6. **Model Optimizasyonu:** Hiperparametre ayarları, çapraz doğrulama ile modelin iyileştirilmesi.
7. **Model Kullanımı:** Eğitim ve değerlendirme süreci tamamlandıktan sonra gerçek veriler üzerinde modelin kullanılması.

### Python ile Sınıflandırma Örneği

Aşağıda, bir veri seti üzerinde SVM kullanarak bir sınıflandırma modeli oluşturma örneği verilmiştir.

### Adım 1: Gerekli Kütüphaneleri İndirme

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
```

### Adım 2: Veri Hazırlığı

```python
# Örnek veri seti (Iris veri seti)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Adım 3: Model Eğitimi

```python
# SVM modelini oluşturma ve eğitme
model = SVC(kernel='linear')
model.fit(X_train, y_train)
```

### Adım 4: Tahmin Yapma

```python
# Test verisi ile tahmin yapma
y_pred = model.predict(X_test)
```

### Adım 5: Model Değerlendirme

```python
# Hata Matrisi
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Doğruluk
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Detaylı Sınıflandırma Raporu
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
```

Bu örnek, bir sınıflandırma modelinin Python kullanılarak nasıl oluşturulacağını ve değerlendirileceğini göstermektedir. Model, iris veri seti üzerinde SVM kullanarak eğitilmiştir ve modelin doğruluğu ve diğer değerlendirme metrikleri hesaplanmıştır. Sınıflandırma, birçok gerçek hayat probleminde yaygın olarak kullanılır ve doğru algoritma ve veri ile oldukça etkili sonuçlar verebilir.

### Sınıflandırma Modeli Değerlendirme Metrikleri

Sınıflandırma modellerini değerlendirirken, çeşitli performans metrikleri kullanılır. Bu metrikler, modelin doğruluğunu, hassasiyetini ve genel performansını anlamamıza yardımcı olur. Aşağıda, en yaygın kullanılan sınıflandırma değerlendirme metriklerinden bazıları ve bunların nasıl hesaplandığı anlatılmaktadır:

### 1. Confusion Matrix (Hata Matrisi)

Confusion Matrix, sınıflandırma modelinin performansını değerlendirmek için kullanılan bir tablodur. Bu tablo, modelin doğru ve yanlış tahminlerini dört kategoriye ayırır:

- **True Positive (TP):** Modelin doğru bir şekilde pozitif olarak tahmin ettiği örnekler.
- **True Negative (TN):** Modelin doğru bir şekilde negatif olarak tahmin ettiği örnekler.
- **False Positive (FP):** Modelin yanlış bir şekilde pozitif olarak tahmin ettiği örnekler (Type I error).
- **False Negative (FN):** Modelin yanlış bir şekilde negatif olarak tahmin ettiği örnekler (Type II error).

|  | Predicted Positive | Predicted Negative |
| --- | --- | --- |
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

### 2. Accuracy (Doğruluk)

Accuracy, modelin doğru tahminlerinin toplam tahminlere oranıdır. Tüm sınıfların doğru tahmin edilme oranını gösterir.

### 3. Precision (Kesinlik)

Precision, modelin pozitif tahminlerinin ne kadar doğru olduğunu ölçer. Yani, modelin pozitif tahmin ettiği değerlerin gerçekten pozitif olma oranıdır.

### 4. Recall (Duyarlılık veya Hassasiyet)

Recall, modelin gerçek pozitif örnekleri ne kadar iyi tespit ettiğini ölçer. Yani, gerçek pozitif örneklerin ne kadarının doğru tahmin edildiğini gösterir.

### 5. F1 Score

F1 Score, Precision ve Recall metriklerinin harmonik ortalamasıdır. Modelin genel performansını ölçmek için kullanılır ve dengesiz veri setlerinde özellikle önemlidir.

### Python ile Metriklerin Hesaplanması

Aşağıda, bir sınıflandırma modelinin değerlendirilmesi için bu metriklerin Python kullanılarak nasıl hesaplanacağını gösteren bir örnek verilmiştir:

```python
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Gerçek etiketler (y_test) ve modelin tahminleri (y_pred)
y_test = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1])

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
```

### Örnek Çıktı

```
Confusion Matrix:
[[3 2]
 [1 4]]
Accuracy: 0.7
Precision: 0.6666666666666666
Recall: 0.8
F1 Score: 0.7272727272727272
```

Bu örnek, `y_test` ve `y_pred` arasındaki metriklerin nasıl hesaplanacağını gösterir. Modelin doğruluk, kesinlik, duyarlılık ve F1 skorlarını değerlendirerek, modelin performansını daha iyi anlayabiliriz. Bu metrikler, modelin zayıf ve güçlü yönlerini belirlememize yardımcı olur ve modelin iyileştirilmesi gereken alanları ortaya çıkarır.

---

### Denetimli Öğrenme (Supervised Learning)

Denetimli öğrenme iki ana kategoriye ayrılır:

1. **Sınıflandırma (Classification):** Modelin, girdileri belirli sınıflara ayırması gereken durumlar. Örneğin, bir e-postanın spam olup olmadığını belirlemek, hastanın hastalığının olup olmadığını tespit etmek.
2. **Regresyon (Regression):** Modelin, sürekli bir değeri tahmin etmesi gereken durumlar. Örneğin, bir evin fiyatını tahmin etmek, hava sıcaklığını tahmin etmek.

### Denetimsiz Öğrenme (Unsupervised Learning)

Denetimsiz öğrenme ise, modelin etiketlenmemiş veri ile eğitildiği bir öğrenme türüdür. Burada model, verinin yapısını ve desenlerini keşfetmeye çalışır. Denetimsiz öğrenmenin iki ana türü vardır:

1. **Kümeleme (Clustering):** Benzer veri noktalarını aynı gruba (küme) ayırma işlemi. Örneğin, müşterileri alışveriş davranışlarına göre segmentlere ayırmak, benzer türdeki haber makalelerini gruplamak.
    - KMeans
    - Hierarchical Clustering
    - DBSCAN
    - Mean-Shift
2. **Boyut İndirgeme (Dimensionality Reduction):** Yüksek boyutlu veriyi daha düşük boyutlu bir temsil ile özetleme işlemi. Bu, veri görselleştirme ve veri ön işleme gibi alanlarda kullanılır.
    - PCA (Principal Component Analysis)
    - t-SNE
    - LDA (Linear Discriminant Analysis)

### Denetimsiz Öğrenme Metrikleri

Denetimsiz öğrenme modellerini değerlendirmek için de bazı metrikler kullanılır, ancak bu metrikler denetimli öğrenme modellerininki kadar doğrudan değildir. Örneğin, kümeleme modellerini değerlendirirken kullanılan metrikler:

- **Silhouette Skoru:** Kümeleme kalitesini ölçer. Değer -1 ile 1 arasında değişir. 1'e yakın değerler, iyi bir kümelemeyi gösterir.
- **Davies-Bouldin İndeksi:** Kümeleme kalitesini ölçer. Daha düşük değerler, daha iyi kümelemeyi gösterir.
- **Inertia:** KMeans algoritmasında, kümeler içindeki veri noktalarının toplam varyansını ölçer. Daha düşük değerler, daha kompakt kümeleri gösterir.

---

## Kümeleme (Clustering) Nedir?

Kümeleme, denetimsiz öğrenme (unsupervised learning) yöntemlerinden biridir ve benzer veri noktalarını gruplara (küme) ayırmayı amaçlar. Kümeleme, verideki yapıları ve desenleri keşfetmek için kullanılır ve veri ön işleme, veri analizi ve veri madenciliği gibi alanlarda yaygın olarak uygulanır. Kümeleme yöntemleri, veri noktalarının benzerliklerine veya uzaklıklarına göre gruplandırılmasını sağlar.

### Kümeleme Türleri

Kümeleme yöntemleri birkaç ana kategoriye ayrılabilir:

1. **Merkezi Kümeleme (Centroid-based Clustering):** Bu yöntemde kümeler, küme merkezleri etrafında toplanır. KMeans algoritması bu kategoriye örnektir.
2. **Yoğunluk Tabanlı Kümeleme (Density-based Clustering):** Kümeler, veri noktalarının yoğun olduğu bölgelerde oluşur. DBSCAN ve Mean-Shift bu kategoriye örnektir.
3. **Hiyerarşik Kümeleme (Hierarchical Clustering):** Veri noktaları, bir ağaç yapısı içinde hiyerarşik olarak gruplanır. Kümeleme, aglomeratif (birleştirici) veya bölücü (parçalama) yöntemlerle yapılabilir.
4. **Model Tabanlı Kümeleme (Model-based Clustering):** Verinin belirli bir dağılım modeline (genellikle Gaussian) göre kümelenmesini sağlar. Gaussian Mixture Models (GMM) bu kategoriye örnektir.

### Örnek Kümeleme Algoritmaları

### 1. KMeans Kümeleme

**KMeans** algoritması, veriyi kkk adet küme merkezine (centroid) ayırır ve her veri noktasını en yakın küme merkezine atar. Algoritma iteratif olarak çalışır ve her adımda küme merkezlerini günceller.

**Adımlar:**

1. kkk adet rastgele küme merkezi belirlenir.
2. Her veri noktası, en yakın küme merkezine atanır.
3. Küme merkezleri, kendilerine atanan veri noktalarının ortalaması alınarak güncellenir.
4. Adım 2 ve 3 tekrarlanır, küme merkezleri değişmeyene kadar devam edilir.

**Python Örneği:**

```python
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Örnek veri seti oluşturma
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# KMeans modelini oluşturma ve eğitme
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Tahmin edilen kümeler
y_kmeans = kmeans.predict(X)

# Küme merkezlerini görselleştirme
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.show()
```

### 2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

**DBSCAN** algoritması, veri noktalarının yoğun olduğu bölgeleri tespit ederek kümeler oluşturur. Gürültü (noise) noktalarını tespit etme yeteneğine sahiptir ve küme sayısını önceden belirtmeye gerek yoktur.

**Parametreler:**

- `eps`: İki veri noktası arasındaki maksimum mesafe, bu mesafede noktalar komşu olarak kabul edilir.
- `min_samples`: Bir noktayı çekirdek nokta (core point) olarak kabul etmek için gerekli minimum komşu sayısı.

**Python Örneği:**

```python
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Örnek veri seti oluşturma
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# DBSCAN modelini oluşturma ve eğitme
dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(X)

# Tahmin edilen kümeler
y_dbscan = dbscan.labels_

# Küme merkezlerini görselleştirme
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan, s=50, cmap='viridis')
plt.show()
```

### 3. Hiyerarşik Kümeleme

**Hiyerarşik kümeleme**, veri noktalarını hiyerarşik bir ağaç yapısı içinde gruplandırır. İki ana yaklaşımı vardır:

- **Aglomeratif (Birleştirici):** Her veri noktası başlangıçta kendi başına bir küme olarak kabul edilir. Kümeler, en yakın iki kümenin birleştirilmesiyle oluşturulur. Bu işlem, tüm veri noktaları tek bir küme oluşturana kadar devam eder.
- **Bölücü (Parçalayıcı):** Tüm veri noktaları başlangıçta tek bir küme olarak kabul edilir. Küme, en uzak iki noktanın ayrılmasıyla oluşturulur. Bu işlem, her veri noktası kendi başına bir küme oluşturana kadar devam eder.

**Python Örneği (Aglomeratif):**

```python
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt

# Örnek veri seti oluşturma
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# Hiyerarşik modelini oluşturma ve eğitme
agglo = AgglomerativeClustering(n_clusters=4)
y_agglo = agglo.fit_predict(X)

# Küme merkezlerini görselleştirme
plt.scatter(X[:, 0], X[:, 1], c=y_agglo, s=50, cmap='viridis')
plt.show()
```

### Kümeleme Algoritmalarının Gerçek Hayat Uygulamaları

Kümeleme algoritmaları, birçok gerçek hayat probleminde uygulanabilir:

1. **Müşteri Segmentasyonu:** Müşterileri alışveriş davranışlarına göre gruplamak ve pazarlama stratejilerini buna göre ayarlamak.
2. **Anomalilik Tespiti:** Finansal dolandırıcılık tespiti gibi anormal davranışları belirlemek.
3. **Biyoinformatik:** Genetik verileri analiz ederek benzer genetik özelliklere sahip bireyleri gruplandırmak.
4. **Görüntü Segmentasyonu:** Görüntüdeki nesneleri veya bölgeleri gruplandırmak.
5. **Pazarlama ve Reklamcılık:** Hedeflenen reklam kampanyaları oluşturmak için kullanıcıları segmentlere ayırmak.
6. **Belge Kümeleme:** Benzer içeriklere sahip belgeleri veya haber makalelerini gruplandırmak.

Kümeleme, veri madenciliği ve veri analizi süreçlerinde güçlü bir araçtır ve veri setlerinin iç yapısını anlamak için kullanılır. Farklı algoritmalar, farklı veri türleri ve problemler için uygundur, bu nedenle doğru algoritmanın seçilmesi önemlidir.