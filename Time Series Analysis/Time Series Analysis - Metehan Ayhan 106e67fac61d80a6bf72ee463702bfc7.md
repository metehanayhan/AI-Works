# Time Series Analysis - Metehan Ayhan

Zaman serisi analizi, geçmişten günümüze çeşitli bilimsel ve teknik alanlarda yaygın olarak kullanılan güçlü bir yöntemdir. Ekonomiden finansal piyasalara, mühendislikten biyomedikal verilere kadar geniş bir uygulama alanı vardır. Bu rehberde, zaman serisi analizini en ince detaylarıyla ele alacağız, temel kavramlardan ileri düzey yöntemlere kadar adım adım inceleyeceğiz.

---

### **İçindekiler**

1. **Zaman Serisi Nedir?**
    - Tanım
    - Zaman Serisi Verilerinin Özellikleri
    - Zaman Serisi Örnekleri
2. **Zaman Serisi Bileşenleri**
    - Trend
    - Mevsimsellik (Seasonality)
    - Döngü (Cycle)
    - Gürültü (Noise)
3. **Zaman Serisi Modelleri**
    - Deterministik ve Stokastik Modeller
    - Doğrusal ve Doğrusal Olmayan Modeller
    - AR (AutoRegressive) Modeli
    - MA (Moving Average) Modeli
    - ARMA ve ARIMA Modelleri
    - SARIMA (Seasonal ARIMA) Modeli
    - ARCH ve GARCH Modelleri
    - VAR (Vector Autoregression) Modelleri
4. **Zaman Serisi Tahmin Yöntemleri**
    - Naive Tahmin Yöntemi
    - Basit Ortalama Yöntemi
    - Basit Hareketli Ortalama (SMA)
    - Ağırlıklı Hareketli Ortalama (WMA)
    - Exponential Smoothing (Üstel Düzeltme)
    - Holt-Winters Modeli
5. **Zaman Serisi İstatistikleri**
    - Otokorelasyon ve Gecikme (Lags)
    - ACF (Autocorrelation Function)
    - PACF (Partial Autocorrelation Function)
    - Stationarity (Durağanlık)
    - KPSS ve ADF Testleri
6. **Zaman Serisinin Durağanlığa Dönüştürülmesi**
    - Fark Alma (Differencing)
    - Logaritmik Dönüşüm
    - Trend Çıkarma
7. **Modellerin Değerlendirilmesi**
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error)
    - AIC (Akaike Information Criterion) ve BIC (Bayesian Information Criterion)
8. **Python ile Zaman Serisi Analizi**
    - Zaman Serisi Verilerinin İşlenmesi (Pandas)
    - Zaman Serisi Modellerinin Uygulanması (Statsmodels, Prophet)
    - Mevsimsellik ve Trendlerin Ayrıştırılması (Seasonal Decompose)
    - SARIMA ile Mevsimsel Tahminler
    - Hyperparameter Tuning
9. **Uygulamalar ve Case Study'ler**
    - Finansal Piyasa Verilerinde Zaman Serisi Analizi
    - Ekonomik Verilerle ARIMA Tahminleri
    - Satış Tahmini ve Stok Yönetimi
    - Doğal Olayların Zaman Serisi Modelleri (Meteoroloji, Depremler)
    - COVID-19 Salgını ve Epidemiyolojik Modeller
10. **İleri Düzey Konular**
    - Deep Learning ve Zaman Serisi Analizi (RNN, LSTM)
    - Büyük Veride Zaman Serisi (Big Data ve Spark)
    - Anomali Tespiti ve Zaman Serisi
    - Granger Nedenselliği (Granger Causality)

---

# 1. Bölüm

### **Zaman Serisi Nedir?**

Zaman serisi, belirli zaman aralıklarında düzenli olarak kaydedilen veri noktalarının bir dizisidir. Bu veri noktaları genellikle tek bir gözlemden elde edilir ve ardışık bir şekilde zaman içinde ilerleyen olayları temsil eder. Zaman serisi analizinin temel amacı, geçmişte kaydedilen verilere bakarak mevcut durumu anlamak ve gelecekteki olası değerleri tahmin etmektir. Ekonomi, finans, meteoroloji, biyomedikal ve mühendislik gibi birçok farklı alanda zaman serisi verileri analiz edilir.

Zaman serileri genellikle iki şekilde kategorize edilir:

- **Kesikli Zaman Serileri (Discrete Time Series)**: Gözlemler belirli zaman aralıklarında yapılır. Örneğin, günlük hisse senedi fiyatları veya haftalık satış verileri.
- **Sürekli Zaman Serileri (Continuous Time Series)**: Gözlemler kesintisizdir, zaman sürekli bir akış halindedir. Örneğin, bir cihazın sürekli olarak kaydettiği sıcaklık değişimleri.

### **Zaman Serisi Verilerinin Özellikleri**

Zaman serisi verileri, diğer veri türlerinden belirli yönlerden ayrılır. Bu özellikler, zaman serisi analizini diğer istatistiksel analizlerden farklı kılan temel faktörlerdir. İşte zaman serisi verilerinin bazı belirleyici özellikleri:

### 1. **Zamana Bağımlılık**

Zaman serilerinin en ayırt edici özelliği, veri noktalarının zamana bağlı olmasıdır. Veriler, birbiri ardına belirli bir zaman dilimi içerisinde kaydedildiği için, önceki veri noktaları gelecekteki veri noktalarını etkileyebilir. Bu, zaman serilerini geleneksel regresyon analizinden ayıran en temel farktır; çünkü geleneksel analizde veriler zamansal bir bağımlılık göstermezken, zaman serisi analizinde gözlemler ardışık ve birbirine bağımlıdır.

### 2. **Trend (Uzun Dönemli Eğilim)**

Zaman serileri, uzun bir süre boyunca genel bir artış, azalış veya durağanlık gösterebilir. Bu uzun vadeli hareketlere **trend** adı verilir. Örneğin, bir şirketin yıllık gelirinde yıllar boyunca sürekli bir artış gözlemlenebilir. Trend, zaman serilerinin uzun vadeli değişkenliğini anlamak ve tahmin etmek için kritik bir bileşendir.

### 3. **Mevsimsellik (Seasonality)**

Mevsimsellik, zaman serisi verilerinde belirli bir dönemsel kalıbın tekrar etmesidir. Örneğin, perakende satışlar genellikle yılın belirli aylarında (örneğin, tatil dönemlerinde) artış gösterir. Mevsimsellik, genellikle yıllık, aylık veya haftalık döngülerde kendini gösterir ve bu döngüler zaman serisi analizinde önemli bir yer tutar.

### 4. **Durağanlık (Stationarity)**

Zaman serisi analizinde önemli bir kavram olan durağanlık, verinin istatistiksel özelliklerinin zaman içinde sabit kalması durumudur. Bir zaman serisi durağan olduğunda, ortalaması ve varyansı zamanla değişmez. Durağan olmayan zaman serileri, genellikle belirli bir trende veya mevsimsel bileşene sahiptir ve analiz öncesinde genellikle durağan hale getirilmesi gerekir.

### 5. **Otokorelasyon**

Zaman serilerinde, veri noktaları birbirine bağımlı olabilir. Yani, belirli bir veri noktası, önceki veri noktalarının bir fonksiyonu olabilir. Bu bağımlılık, **otokorelasyon** olarak adlandırılır. Otokorelasyon, zaman serilerinin gelecekteki değerlerini tahmin ederken önemli bir ipucu sunar. Örneğin, bir zaman serisinde bir gün önceki hava durumu, bugünkü hava durumu üzerinde etkili olabilir.

### 6. **Rastgelelik (Noise)**

Bir zaman serisinde gözlemler arasında doğal olarak meydana gelen rastgele dalgalanmalara gürültü (noise) denir. Gürültü, veri setindeki rastgele varyasyonları temsil eder ve genellikle analiz sırasında filtrelenmesi veya modellenmesi gereken bir unsurdur. Gürültü, her zaman serisi verisinde bir dereceye kadar bulunur ve bunu anlamak, veriyi doğru bir şekilde modelleyebilmek açısından önemlidir.

### **Zaman Serisi Örnekleri**

Zaman serisi analizinin kullanıldığı birçok farklı alan ve veri türü vardır. İşte bazı yaygın zaman serisi örnekleri:

### 1. **Finansal Veriler**

Finans piyasaları, zaman serisi analizinin en yaygın olarak kullanıldığı alanlardan biridir. Örneğin, bir hisse senedinin günlük kapanış fiyatları, zaman serisi verilerinin klasik bir örneğidir. Yatırımcılar, bu fiyat verilerini analiz ederek, hisse senedinin gelecekteki değerini tahmin etmeye çalışırlar. Diğer finansal zaman serisi örnekleri arasında döviz kurları, faiz oranları ve tahvil fiyatları yer alır.

### 2. **Ekonomik Göstergeler**

Ekonomi alanında, zaman serisi analizi, bir ülkenin GSYİH (Gayri Safi Yurtiçi Hasıla), işsizlik oranı, enflasyon oranı gibi ekonomik göstergelerini incelemek için kullanılır. Ekonomistler, bu veriler üzerinden ekonominin gelecekteki durumunu tahmin edebilir ve buna göre politikalar geliştirebilirler.

### 3. **Meteorolojik Veriler**

Meteoroloji, zaman serisi analizinin sıkça kullanıldığı bir başka alandır. Günlük sıcaklıklar, rüzgar hızları, yağış miktarları gibi meteorolojik veriler, zaman serileri olarak kaydedilir ve gelecekteki hava durumunu tahmin etmek için kullanılır. Bu tür verilerde genellikle belirgin bir mevsimsellik ve otokorelasyon bulunur.

### 4. **Enerji Tüketimi**

Bir şehirdeki günlük elektrik tüketimi zaman serisi verisi olarak kaydedilir. Bu verilerde hem trend hem de mevsimsellik gözlemlenebilir. Örneğin, kış aylarında enerji tüketimi artabilir, yaz aylarında ise düşebilir. Bu tür veriler enerji yönetimi ve planlaması için hayati öneme sahiptir.

### 5. **Trafik Verileri**

Bir şehirdeki günlük trafik yoğunluğu zaman serisi verisi olarak değerlendirilebilir. Bu veriler, yolların daha iyi yönetilmesi, gelecekteki trafik sıkışıklıklarının tahmin edilmesi ve toplu taşıma planlaması için analiz edilir. Trafik verilerinde genellikle günlük ve haftalık mevsimsellik bulunur; örneğin, hafta içi trafik yoğunluğu hafta sonuna göre farklılık gösterebilir.

### 6. **Sağlık ve Biyomedikal Veriler**

Hastaların kalp atış hızı, kan basıncı, kandaki oksijen seviyeleri gibi biyomedikal veriler de zaman serisi analizine tabi tutulabilir. Bu tür veriler, sağlık durumunu izlemek ve hastalığın seyrini tahmin etmek için kullanılır. Örneğin, bir hastanın kalp atış hızı sürekli olarak izlenerek, ani değişiklikler durumunda erken müdahale edilebilir.

**Sonuç olarak**, zaman serisi verileri, zamana bağımlı, otokorelasyon gösteren, trend ve mevsimsellik gibi bileşenler içerebilen, gürültüyle boğuşan ve birçok uygulama alanı olan veri türleridir. Zaman serisi analizini doğru bir şekilde yapmak için bu özellikleri anlamak kritik öneme sahiptir.

---

# 2. Bölüm

### **Zaman Serisi Bileşenleri**

Zaman serileri, belirli yapı taşlarına sahiptir. Bu yapı taşlarını anlamak, zaman serilerini analiz ederken veriyi modellememize, tahmin yapmamıza ve değişkenliği anlamamıza olanak tanır. Dört temel bileşen, zaman serilerini karakterize eder: **Trend**, **Mevsimsellik**, **Döngü** ve **Gürültü**. Bu bileşenler, bir zaman serisinin belirli örüntülerini ve sapmalarını açıklamak için kullanılır.

---

### **1. Trend (Uzun Dönemli Eğilim)**

**Trend**, zaman serisindeki uzun vadeli yönelim veya eğilimi ifade eder. Verilerin zaman içinde genel olarak bir artış, azalma veya durağanlık gösterdiği uzun vadeli değişimdir. Genellikle, zaman serilerinde sürekli bir değişim gözlemlendiğinde, bu değişim bir trend olarak kabul edilir.

### **Trend Çeşitleri:**

- **Pozitif Trend**: Zaman ilerledikçe veri değerlerinin sürekli arttığı bir eğilimdir. Örneğin, teknolojinin gelişmesiyle birlikte dünya çapında internet kullanıcı sayısının sürekli artması pozitif bir trende örnektir.
- **Negatif Trend**: Zaman içinde veri değerlerinin azaldığı bir eğilimdir. Örneğin, eski model bir ürünün satışlarının yeni model çıktıkça düşmesi negatif trende örnek verilebilir.
- **Durağan (No Trend)**: Zaman ilerledikçe veri değerlerinde net bir artış veya azalış görülmediğinde durağan bir trendten bahsedilir. Örneğin, bir ürüne olan talebin uzun bir süre boyunca sabit kalması.

### **Trendlerin Zaman Serisi Analizine Etkisi:**

Trendler, zaman serisinin uzun vadeli bileşenini modellemeye yardımcı olur. Bir zaman serisi durağan değilse ve açık bir trend gösteriyorsa, analiz öncesinde bu trendin çıkarılması gerekebilir. Bu, özellikle gelecekteki tahminler için verinin daha doğru olmasını sağlar. Genellikle, hareketli ortalamalar (moving average) veya filtreleme yöntemleri kullanılarak trend ortadan kaldırılır.

---

### **2. Mevsimsellik (Seasonality)**

**Mevsimsellik**, zaman serisi verilerinde belirli periyodik döngülerin tekrar etmesidir. Mevsimsel hareketler, verilerin yılın veya günün belirli zaman dilimlerinde düzenli aralıklarla aynı örüntüyü sergilemesidir. Bu periyodik dalgalanmalar, veri üzerinde belirgin kısa vadeli değişimleri gösterir.

### **Mevsimselliğin Özellikleri:**

- **Periyodik Olma**: Mevsimsellik, belirli bir dönemde sürekli tekrar eden örüntülerdir. Örneğin, her yılın yaz aylarında turizm sektöründeki gelir artışları gibi.
- **Döngü Süresi**: Mevsimsel bileşenler genellikle yıllık, aylık, haftalık ya da günlük döngülerle kendini gösterir. Örneğin, perakende satışlarda Kasım ve Aralık aylarında gözlenen artış bir mevsimsel döngü örneğidir.
- **Düzenlilik**: Mevsimsel hareketler, belirli aralıklarla düzenli olarak ortaya çıkar. Örneğin, her hafta sonu trafik yoğunluğunda gözlenen azalma bir mevsimsel örüntü olabilir.

### **Mevsimsellik Örnekleri:**

- **Perakende Satışları**: Yılbaşı döneminde ve tatillerde artan alışverişler mevsimsel hareketleri gösterir.
- **Hava Durumu Verileri**: Sıcaklık ve yağış verileri belirli mevsimsel kalıplar sergiler.
- **Turizm**: Tatil dönemlerinde artan turistik seyahatler, özellikle yaz aylarında yükselen otel doluluk oranları mevsimsel döngülere güzel bir örnektir.

### **Mevsimselliğin Zaman Serisi Analizine Etkisi:**

Mevsimsellik, gelecekteki tahminlerde önemli bir rol oynar çünkü belirli dönemlerde düzenli olarak tekrar eden kalıpları dikkate alır. Mevsimsel bileşenleri anlamak, bir modelin doğruluğunu artırabilir ve periyodik dalgalanmaların etkisini daha iyi analiz etmenize olanak tanır. Genellikle, mevsimsel düzenin çıkarılması veya modele dahil edilmesi için **sezon düzeltmeleri** uygulanır.

---

### **3. Döngü (Cycle)**

**Döngü** (cycle), mevsimselliğe benzer şekilde zaman içinde tekrar eden dalgalanmaları temsil eder, ancak döngüler genellikle daha uzun süreli ve düzensiz aralıklarla ortaya çıkar. Döngüler, ekonomik ve iş döngüleri gibi dış faktörlerin neden olduğu, genellikle birkaç yılda bir gözlemlenen geniş çaplı değişimlerdir.

### **Döngülerin Özellikleri:**

- **Daha Uzun Süreli**: Döngüler, mevsimsel bileşenlerden daha uzun sürer. Ekonomik döngüler genellikle birkaç yıl sürebilir.
- **Düzensiz Periyotlar**: Mevsimsel bileşenlerin aksine, döngülerin periyotları düzensizdir. Yani, döngüler her yıl veya her ay aynı şekilde ortaya çıkmaz. Örneğin, bir ekonomik durgunluk her 5 yılda bir yaşanabilir, ancak bu süre sabit olmayabilir.
- **Ekonomik ve Sosyal Faktörlere Bağlılık**: Döngüler, genellikle ekonomik büyüme, durgunluk, sosyal olaylar veya doğal afetler gibi faktörlerin neden olduğu büyük ölçekli değişikliklerden kaynaklanır.

### **Döngü Örnekleri:**

- **Ekonomik Döngüler**: Ekonomik durgunluklar, genişleme dönemleri ve ekonomik büyüme aşamaları tipik döngüsel olaylardır.
- **İş Döngüleri**: Şirketlerin zaman içinde belirli dönemlerdeki büyüme ve küçülme aşamaları da döngüsel hareketleri gösterebilir.
- **Doğal Kaynak Döngüleri**: Tarım ürünlerinde hava koşullarına bağlı olarak yıllık üretim miktarındaki değişiklikler de döngüsel olabilir.

### **Döngülerin Zaman Serisi Analizine Etkisi:**

Döngüler, zaman serisi analizi sırasında genellikle modelleme zorlukları yaratabilir, çünkü düzensiz periyotlar ve geniş zaman aralıklarıyla kendilerini gösterirler. Döngülerin doğru bir şekilde modellenmesi, uzun vadeli tahminlerde önemli bir faktördür.

---

### **4. Gürültü (Noise)**

**Gürültü**, bir zaman serisindeki rastgele ve öngörülemeyen dalgalanmalardır. Gürültü, zaman serisinin modellenemeyen, anlaşılamayan ve dış etkenlerden kaynaklanan bileşenidir. Zaman serilerindeki gürültü, trend, mevsimsellik ve döngü gibi diğer bileşenlerle ilişkili olmayan ve tamamen rastgele değişimlerden oluşur.

### **Gürültünün Özellikleri:**

- **Rastgelelik**: Gürültü tamamen rastgeledir ve belirli bir örüntü göstermez. Yani, gürültüde düzenli bir hareket veya döngü bulunmaz.
- **Modellenemezlik**: Gürültü, modeller tarafından tahmin edilemez. Modeller, gürültüyü filtrelemeye veya etkilerini minimize etmeye çalışır, ancak tamamen ortadan kaldırmak mümkün değildir.
- **Dış Etkenlere Bağlılık**: Gürültü, genellikle dış faktörlerin beklenmedik etkilerinden kaynaklanır. Örneğin, bir doğal afet, ekonomi üzerindeki ani ve beklenmedik bir değişikliğe neden olabilir ve bu etki zaman serisine gürültü olarak yansır.

### **Gürültü Örnekleri:**

- **Finansal Piyasalarda**: Hisse senedi fiyatlarındaki ani dalgalanmalar genellikle gürültü olarak kabul edilir. Bu dalgalanmalar, yatırımcıların ani kararları veya dışsal olaylardan kaynaklanabilir.
- **Meteorolojik Verilerde**: Hava durumu tahminlerinde meydana gelen ani ve tahmin edilemeyen değişiklikler gürültü olarak adlandırılabilir. Örneğin, ani bir fırtına, sıcaklık veya yağış tahminlerinde beklenmedik değişikliklere yol açabilir.

### **Gürültünün Zaman Serisi Analizine Etkisi:**

Gürültü, zaman serisi analizinde zorluk çıkaran en karmaşık bileşendir. Zaman serilerinde gürültüyü minimize etmek ve ana örüntüyü ortaya çıkarmak için genellikle filtreleme teknikleri kullanılır. Gürültü ne kadar düşükse, modelin doğruluğu o kadar artar. Ancak, tüm zaman serilerinde bir miktar gürültü her zaman olacaktır ve tamamen yok edilemez.

---

# 3. Bölüm

### **Zaman Serisi Modelleri**

Zaman serisi analizinde, farklı türdeki modeller, verinin yapısına ve tahmin amacına göre seçilir. Bu modeller, verilerin gelecekte nasıl bir yol izleyeceğini öngörmek veya veri içindeki örüntüleri tanımlamak için kullanılır. Zaman serisi modelleri, **deterministik** ve **stokastik**, **doğrusal** ve **doğrusal olmayan** olarak sınıflandırılabilir. Ayrıca, zaman serilerindeki bağımlılıkları modelleyen çeşitli matematiksel yaklaşımlar (AR, MA, ARIMA gibi) kullanılır.

---

### **Deterministik ve Stokastik Modeller**

### **Deterministik Modeller**

Deterministik modeller, zaman serisinde kesin bir yapı olduğunu varsayar. Bu yapı, dış faktörler tarafından değiştirilemeyen ve net bir şekilde tahmin edilebilen bir örüntü içerir. Örneğin, bir deterministik modelde, mevsimsel veya trend gibi değişimler tamamen öngörülebilir. Bu modellerde rastgelelik yoktur.

- **Örnek**: Bir saatin tik tak sesleri deterministik bir süreç olarak kabul edilebilir, çünkü belirli aralıklarla düzenli olarak tekrar eder.

### **Stokastik Modeller**

Stokastik modeller, zaman serisinin rastgele bileşenlere sahip olduğunu ve gelecekteki değerlerin tamamen belirli bir örüntüye dayanamayacağını varsayar. Stokastik modellerde, her veri noktası bir önceki değerden etkilenebilir, ancak rastgele bir sapma ile tahmin edilir. Bu modeller, gerçek dünyadaki birçok zaman serisini daha iyi açıklar çünkü rastgelelik bu süreçlerde önemli bir rol oynar.

- **Örnek**: Finansal piyasalardaki hisse senedi fiyatları stokastik olarak kabul edilir. Fiyatlar, önceki fiyatlara bağımlı olmakla birlikte, rastgele dalgalanma gösterir.

---

### **Doğrusal ve Doğrusal Olmayan Modeller**

### **Doğrusal Modeller**

Doğrusal modeller, bir zaman serisinin gelecekteki değerlerinin geçmiş değerlerinin doğrusal bir kombinasyonu olarak ifade edilebileceğini varsayar. Bu modellerde, veriler arasındaki ilişki sabit bir oranda artış ya da azalış gösterir.

- **Örnek**: AR (AutoRegressive) ve MA (Moving Average) modelleri genellikle doğrusal modellerdir.

### **Doğrusal Olmayan Modeller**

Doğrusal olmayan modeller, veriler arasındaki ilişkilerin daha karmaşık olduğunu ve sabit bir oranda artış ya da azalış göstermediğini varsayar. Bu modeller, finansal zaman serilerinde veya karmaşık olaylarda daha yaygın olarak kullanılır.

- **Örnek**: ARCH (Autoregressive Conditional Heteroskedasticity) ve GARCH (Generalized Autoregressive Conditional Heteroskedasticity) modelleri, zaman serisinin varyansının doğrusal olmadığını varsayan doğrusal olmayan modellerdir.

---

### **AR (AutoRegressive) Modeli**

**AR modeli**, zaman serisinin önceki değerlerine dayanarak tahmin yapılmasını sağlayan bir modeldir. Bu modelde, her veri noktası, önceki dönemlerdeki gözlemlerle açıklanır. **p** gecikme derecesine sahip bir AR(p) modeli, geçmişteki p gözlemi kullanarak gelecekteki değeri tahmin eder.

- **Kullanım Alanı**: AR modelleri, özellikle bir zaman serisinin geçmiş gözlemlerine bağlı olduğu durumlarda etkilidir. Örneğin, hisse senedi fiyatları veya ekonomik göstergelerde kullanılabilir.

---

### **MA (Moving Average) Modeli**

**MA modeli** (Moving Average - Hareketli Ortalama), mevcut veri noktasının, önceki hata terimlerinin (yani rastgele şokların) ağırlıklı ortalaması olarak açıklandığı bir modeldir. **q** gecikmeli bir MA(q) modeli, geçmiş q hata terimi kullanılarak mevcut değeri tahmin eder.

- **Kullanım Alanı**: MA modelleri, veri setinde rastgele şoklar veya kısa vadeli dalgalanmaların önemli olduğu durumlarda kullanılır. Örneğin, finansal verilerde beklenmeyen olayların etkisini incelemek için kullanılır.

---

### **ARMA ve ARIMA Modelleri**

### **ARMA (AutoRegressive Moving Average) Modeli**

**ARMA modeli**, AR ve MA modellerinin bir kombinasyonudur. Bu model, hem geçmiş değerler (AR kısmı) hem de geçmiş hata terimlerini (MA kısmı) kullanarak mevcut değeri tahmin eder.

### **ARIMA (AutoRegressive Integrated Moving Average) Modeli**

**ARIMA modeli**, ARMA modelinin bir genişletmesidir ve özellikle durağan olmayan zaman serileri için kullanılır. ARIMA modeli, veriyi durağan hale getirmek için fark alma işlemini (differencing) kullanır. ARIMA(p, d, q) modelinde, **p** AR terimleri, **d** fark derecesi, **q** ise MA terimlerini ifade eder.

- **Kullanım Alanı**: ARIMA modelleri, mevsimsellik ve trend içeren zaman serilerini modellemek için sıkça kullanılır. Özellikle, ekonomik göstergeler ve finansal verilerde yaygın olarak uygulanır.

---

### **SARIMA (Seasonal ARIMA) Modeli**

**SARIMA modeli**, ARIMA modeline mevsimsellik bileşenini ekleyerek, mevsimsel örüntülerin olduğu zaman serileri için kullanılır. Bu model, hem mevsimsel hem de mevsimsel olmayan bileşenleri içerebilir. SARIMA(p, d, q)(P, D, Q, m) modelinde, mevsimsel terimler de (P, D, Q) eklenmiştir ve **m** mevsimsel periyot uzunluğunu temsil eder.

- **Kullanım Alanı**: SARIMA modelleri, mevsimselliğin olduğu satış verileri, sıcaklık, enerji talebi gibi zaman serilerinde kullanılır.

---

### **ARCH ve GARCH Modelleri**

### **ARCH (Autoregressive Conditional Heteroskedasticity) Modeli**

**ARCH modeli**, zaman serisinin varyansındaki dalgalanmaları modellemek için kullanılır. Özellikle finansal zaman serilerinde, varyansın zaman içinde sabit olmadığını göstermek için kullanılır.

### **GARCH (Generalized ARCH) Modeli**

**GARCH modeli**, ARCH modelinin bir genişlemesidir ve zaman içindeki hem geçmiş hata terimlerinin hem de geçmiş varyansların etkisini modellemede kullanılır. GARCH modelleri, volatilitenin zamanla değiştiği finansal verilerde yaygın olarak uygulanır.

---

### **VAR (Vector Autoregression) Modeli**

**VAR modeli**, çok değişkenli (multivariate) zaman serilerinde kullanılır. Bu modelde, her değişken, hem kendi geçmiş değerlerine hem de diğer değişkenlerin geçmiş değerlerine bağlı olarak tahmin edilir. VAR modeli, çoklu zaman serisinin birlikte analiz edilmesi gereken durumlarda kullanılır.

- **Kullanım Alanı**: Makroekonomik verilerde, birbirleriyle ilişkili birden fazla zaman serisinin analiz edilmesinde kullanılır.

---

# 4. Bölüm

### **Zaman Serisi Tahmin Yöntemleri**

Zaman serisi analizinde, gelecekteki değerleri tahmin etmek için kullanılan çeşitli yöntemler mevcuttur. Bu tahmin yöntemleri, verinin yapısına ve analiz amacına göre seçilir. Aşağıda, zaman serilerinde kullanılan bazı temel tahmin yöntemleri açıklanmıştır:

---

### **Naive Tahmin Yöntemi**

**Naive tahmin yöntemi**, en basit ve doğrudan tahmin yöntemlerinden biridir. Bu yöntemde, gelecekteki veri noktası, geçmişteki en son gözlemin aynısı olarak kabul edilir.

- **Avantajları**:
    - Kolay ve hızlı bir yöntemdir.
    - Durağan ve kısa vadeli tahminler için kullanılabilir.
- **Dezavantajları**:
    - Trend, mevsimsellik veya döngü gibi bileşenleri göz ardı eder.
    - Daha karmaşık zaman serilerinde düşük doğruluk sağlar.

---

### **Basit Ortalama Yöntemi**

**Basit Ortalama Yöntemi**, zaman serisinin tüm geçmiş değerlerinin basit ortalamasını alarak tahmin yapılmasını sağlar. Bu yöntemde, tüm gözlemler eşit ağırlığa sahiptir.

- **Avantajları**:
    - Hesaplaması kolaydır.
    - Trend veya mevsimsellik olmayan zaman serileri için makul sonuçlar verebilir.
- **Dezavantajları**:
    - Tüm gözlemlere eşit ağırlık verilmesi, en son gözlemin tahmin üzerinde yeterince etkili olmamasına neden olabilir.
    - Trend ve mevsimsellik gibi zaman serisinin önemli bileşenlerini dikkate almaz.

---

### **Basit Hareketli Ortalama (SMA)**

**Basit Hareketli Ortalama** (SMA) yöntemi, zaman serisinin en son **k** gözleminin ortalamasını alarak gelecekteki değeri tahmin etmeye dayanır. Bu yöntemde, yalnızca en son gözlemler dikkate alınır.

- **Avantajları**:
    - Basit bir yöntemdir ve kısa vadeli tahminler için etkilidir.
    - En son gözlemlere daha fazla odaklanarak daha güncel tahminler yapılmasına olanak tanır.
- **Dezavantajları**:
    - K seçimine bağlıdır; doğru sayıda geçmiş gözlem seçilmezse tahmin doğruluğu düşebilir.
    - Trend ve mevsimsellik gibi uzun vadeli kalıpları yakalamakta yetersiz kalır.

---

### **Ağırlıklı Hareketli Ortalama (WMA)**

**Ağırlıklı Hareketli Ortalama** (WMA), basit hareketli ortalama yöntemine benzer, ancak her bir gözleme farklı bir ağırlık verilir. Genellikle en son gözlemlere daha yüksek ağırlık verilir.

- **Avantajları**:
    - En son gözlemler üzerinde daha fazla odaklanarak daha güncel tahminler sağlar.
    - Esnektir ve ağırlıklar kullanıcı tarafından ayarlanabilir.
- **Dezavantajları**:
    - Ağırlıkların doğru seçimi tahminin başarısını doğrudan etkiler.
    - Trend ve mevsimsellik gibi özellikleri modellemede sınırlıdır.

---

### **Exponential Smoothing (Üstel Düzeltme)**

**Exponential Smoothing** (Üstel Düzeltme), geçmiş gözlemlere üstel bir ağırlık verir, yani en son gözlemlere daha fazla ağırlık verilirken daha eski gözlemler giderek daha az etkili hale gelir. Bu yöntem, kısa vadeli tahminlerde oldukça etkilidir.

### **Basit Üstel Düzeltme**

- **Avantajları**:
    - Zaman serisinin en son değerine daha fazla önem verir.
    - Kısa vadeli tahminlerde basit ve etkili bir yöntemdir.
- **Dezavantajları**:
    - Trend ve mevsimselliği modelleyemez.
    - Düzeltme katsayısının doğru seçilmesi önemlidir.

### **Holt’un Doğrusal Üstel Düzeltmesi**

**Holt’s Exponential Smoothing**, üstel düzeltmeye trend bileşenini ekleyerek zaman serilerindeki trendleri de modelleyebilir.

### **Holt-Winters Üstel Düzeltme**

**Holt-Winters Modeli**, mevsimselliği modellemek için üstel düzeltme yöntemini genişletir. Bu model, üç bileşeni içerir: düzey (level), trend ve mevsimsellik.

- **Avantajları**:
    - Hem trend hem de mevsimsel bileşenleri modelleyebilir.
    - Özellikle mevsimsel ve trend içeren zaman serileri için kullanışlıdır.
- **Dezavantajları**:
    - Daha karmaşık ve daha fazla parametre ayarı gerektirir.
    - Kısa vadeli tahminlerde daha basit yöntemlere göre aşırıya kaçabilir.

---

### **Holt-Winters Modeli**

**Holt-Winters Modeli**, üstel düzeltme yönteminin bir genişlemesi olarak, trend ve mevsimsel bileşenleri içeren zaman serileri için kullanılır. **Çarpımsal** ve **toplamsal** olmak üzere iki farklı versiyonu vardır.

- **Çarpımsal Holt-Winters**: Mevsimsel bileşenin veri ile çarpıldığı durumlardır. Bu, mevsimsellik etkisinin veri ile orantılı olduğu durumlarda kullanılır.
- **Toplamsal Holt-Winters**: Mevsimsel bileşenin veriye eklenmesi ile kullanılır ve genellikle verinin trendi daha belirgin olmadığı durumlarda uygulanır.
- **Kullanım Alanları**: Özellikle satış tahmini, enerji talebi ve iklim verileri gibi mevsimselliğin önemli olduğu zaman serilerinde sıkça tercih edilir.

---

# 5. Bölüm

### **Zaman Serisi İstatistikleri**

Zaman serilerinde analiz ve tahmin yaparken, verinin iç yapısını anlamak için çeşitli istatistiksel araçlar kullanılır. Bu araçlar, zaman serisinin belirli özelliklerini ortaya çıkarır ve modelleme sürecinde rehberlik eder. Zaman serisi istatistiklerinin en yaygın olanları aşağıda detaylı olarak açıklanmıştır:

---

### **Otokorelasyon ve Gecikme (Lags)**

**Otokorelasyon**, bir zaman serisinin kendi geçmiş değerleri ile olan korelasyonunu ölçen bir istatistiktir. Yani, bir zaman serisi ile gecikmiş versiyonu arasındaki ilişkiyi ifade eder.

- **Tanım**: Otokorelasyon, serinin mevcut gözlemi ile geçmişteki belirli bir zaman aralığındaki (gecikme ya da lag) gözlemler arasındaki korelasyonu gösterir.
- **Gecikme (Lag)**: Gecikme, bir zaman serisindeki değerler arasındaki zaman farkını ifade eder. Örneğin, k=1 gecikmesi, şu anki veri noktası ile bir önceki veri noktası arasındaki ilişkiyi analiz eder. Gecikme sayısı arttıkça, daha eski gözlemlerle olan korelasyon incelenir.
- **Kullanım Alanı**:
    - Zaman serilerindeki kalıpları belirlemek.
    - Tahmin modelleri için uygun gecikme sayısını seçmek.
    - Mevsimsellik ve döngüsellik gibi yapıları tespit etmek.

---

### **ACF (Autocorrelation Function)**

**ACF** (Otokorelasyon Fonksiyonu), zaman serisinin farklı gecikme adımları için otokorelasyon değerlerini hesaplar ve bir grafik üzerinde gösterir. Bu fonksiyon, serinin ne kadar süre boyunca kendi geçmişiyle ilişkili olduğunu anlamamıza yardımcı olur.

- **Tanım**: ACF, belirli bir gecikmedeki otokorelasyon katsayılarını hesaplayarak bu değerleri bir grafik üzerinde gösterir. Gecikme arttıkça korelasyonun azalıp azalmadığını görmemizi sağlar.
- **Grafik**: ACF grafiği, gecikme sayısı (lag) ile otokorelasyon katsayıları arasındaki ilişkiyi gösterir. Genellikle sıfır civarında azalan bir otokorelasyon kalıbı gözlemlenir.
- **Kullanım Alanı**:
    - Zaman serisinin mevsimsel ya da döngüsel bileşenlerini tespit etmek.
    - Serideki otokorelasyon yapısına bağlı olarak hangi modelin (AR, MA, ARMA, ARIMA) daha uygun olduğunu belirlemek.

---

### **PACF (Partial Autocorrelation Function)**

**PACF** (Kısmi Otokorelasyon Fonksiyonu), bir zaman serisindeki belirli bir gecikmedeki otokorelasyonu, önceki tüm gecikmelerin etkisini ortadan kaldırarak ölçer. ACF'den farklı olarak, sadece o gecikmedeki otokorelasyonun "saf" etkisini ortaya koyar.

- **Tanım**: PACF, bir zaman serisinin belirli bir gecikmedeki otokorelasyonunu, daha önceki tüm gecikmelerin etkisini çıkararak hesaplar. Bu, serideki doğrudan ilişkiyi belirlemeye yarar.
- **Kullanım Alanı**:
    - Serinin hangi gecikmelerde önemli bir otokorelasyon gösterdiğini anlamak.
    - Zaman serisi modellerinde (özellikle AR ve ARIMA) kaç adet gecikmeli terim kullanılması gerektiğine karar vermek.
- **PACF Grafiği**: PACF grafiği, gecikme sayısına göre kısmi otokorelasyon katsayılarını gösterir. ACF grafiğinden farklı olarak, sadece belirli gecikmelerde yüksek otokorelasyon gösteren değerler dikkat çeker.

---

### **Stationarity (Durağanlık)**

**Stationarity**, bir zaman serisinin istatistiksel özelliklerinin (ortalama, varyans, kovaryans) zamanla değişmediği durumu ifade eder. Bir zaman serisinin durağan olup olmadığı, tahmin modellerinin doğruluğunu ve uygunluğunu belirlemede kritik bir faktördür.

- **Durağanlık Şartları**:
    - Serinin ortalaması sabit olmalıdır.
    - Varyans zaman içinde değişmemelidir.
    - İki zaman noktası arasındaki kovaryans, zaman farkına bağlı olmalı, ancak zamanın kendisine bağlı olmamalıdır.
- **Durağan Olmayan Seriler**:
    - Zaman serisinin trend veya mevsimsel bileşenler içermesi, durağanlığın bozulmasına neden olabilir. Bu tür serilerde modelleme yapılmadan önce seriyi durağan hale getirmek gerekir (diferansiyelleme gibi yöntemler kullanılarak).
- **Kullanım Alanı**:
    - Durağan olmayan zaman serileri, otoregresif modeller için genellikle uygun değildir.
    - Durağanlık testleri, zaman serisinin uygun bir modele dönüştürülüp dönüştürülmeyeceğini anlamak için kullanılır.

---

### **KPSS ve ADF Testleri**

Zaman serilerinin durağan olup olmadığını belirlemek için kullanılan iki yaygın istatistiksel test **KPSS** ve **ADF** testleridir.

### **ADF (Augmented Dickey-Fuller) Testi**

**ADF Testi**, zaman serisinin birim köke sahip olup olmadığını test eder. Birim kök, serinin durağan olmadığını gösterir.

- **Hipotezler**:
    - **Null Hipotezi (H0)**: Zaman serisi birim köke sahiptir (durağan değildir).
    - **Alternatif Hipotez (H1)**: Zaman serisi durağandır.
- **Test Sonucu**:
    - Eğer ADF testi sonucunda p-değeri düşük çıkarsa (gen

elde 0.05'ten küçükse), null hipotez reddedilir ve serinin durağan olduğu kabul edilir. Eğer p-değeri yüksekse, serinin durağan olmadığı sonucuna varılır.

### **KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Testi**

**KPSS Testi**, durağanlık hipotezini doğrudan test eden bir yöntemdir. ADF testinden farklı olarak, KPSS testinde null hipotez serinin durağan olduğunu varsayar.

- **Hipotezler**:
    - **Null Hipotezi (H0)**: Zaman serisi durağandır.
    - **Alternatif Hipotez (H1)**: Zaman serisi durağan değildir (birim köke sahiptir).
- **Test Sonucu**:
    - Eğer KPSS testi sonucunda p-değeri düşükse, null hipotez reddedilir ve serinin durağan olmadığı sonucuna varılır.
    - Yüksek bir p-değeri ise serinin durağan olduğuna işaret eder.

### **ADF ve KPSS Testlerinin Karşılaştırılması**

- **ADF Testi**, durağanlık için birim kökün var olup olmadığını test eder ve durağan olmayan serileri belirlemeye odaklanır.
- **KPSS Testi**, durağanlık varsayımını test eder ve durağan olup olmadığını doğrulamak için kullanılır.
- Her iki testi birlikte kullanmak, daha güvenilir sonuçlar elde etmek için yaygın bir yaklaşımdır. Örneğin, ADF serinin durağan olduğunu söylerken, KPSS aynı sonuca ulaşabiliyorsa, serinin gerçekten durağan olduğuna dair daha sağlam bir yargıya varılabilir.

---

### **Sonuç**

Zaman serisi istatistikleri, zaman serisi modellemesinde temel unsurlardır. **Otokorelasyon** ve **Gecikme** analizleri, serinin yapısını ve iç ilişkilerini anlamaya yardımcı olurken, **ACF** ve **PACF** grafiklerini yorumlamak, hangi modelin kullanılacağına dair ipuçları verir. **Durağanlık** testleri (ADF ve KPSS) ise, modelin doğruluğunu sağlamak için zaman serisinin uygun bir yapıya sahip olup olmadığını belirlemeye yarar. Bu analiz araçları, doğru bir zaman serisi tahmini yapmak için kritik bir rol oynar.

---

# 6. Bölüm

### **Zaman Serisinin Durağanlığa Dönüştürülmesi**

Zaman serisi analizinde en kritik adımlardan biri, seriyi durağan hale getirmektir. Çünkü birçok zaman serisi modeli (örneğin ARIMA) durağan serilerde daha iyi performans gösterir. Durağan olmayan serilerde ise model tahminlerinin doğruluğu düşebilir. Serinin durağan hale getirilmesi için kullanılan başlıca yöntemler şunlardır:

---

### **Fark Alma (Differencing)**

**Fark alma**, bir zaman serisini durağan hale getirmek için en yaygın kullanılan yöntemdir. Fark alma, serinin iki ardışık gözlemi arasındaki farkın alınmasıyla gerçekleştirilir. Bu işlem, serideki trendi ve mevsimsel bileşenleri ortadan kaldırarak durağanlık sağlar.

- **Kullanım Alanı**:
    - Trendin olduğu zaman serilerinde trendi ortadan kaldırmak.
    - Mevsimsellik içeren serilerde mevsimsel farklar almak.

### **Logaritmik Dönüşüm**

**Logaritmik dönüşüm**, zaman serisinde varyansı sabitlemek ve büyük dalgalanmaları azaltmak amacıyla kullanılan bir yöntemdir. Bu işlem, özellikle serinin yüksek varyans gösterdiği durumlarda etkilidir.

- **Kullanım Alanı**:
    - Pozitif ve artan varyansa sahip serilerde varyansı stabilize etmek.
    - Örneğin, finansal zaman serilerinde, logaritmik dönüşüm dalgalanmaların etkisini azaltır.

### **Trend Çıkarma (Detrending)**

Bir zaman serisinin durağan hale getirilmesi için trend bileşeninin çıkarılması gerekebilir. Trend çıkarma, serideki uzun vadeli artış veya azalışların ortadan kaldırılmasını sağlar.

- **Polinomsel Trend Çıkarma**: Eğer trend doğrusal değilse, polinomsel bir regresyon modeli kullanılarak trend çıkarılabilir.
- **Hareketli Ortalama ile Trend Çıkarma**: Hareketli ortalama yöntemiyle serinin uzun dönemli eğilimleri çıkarılarak kalan serinin durağan olup olmadığı incelenir.

---

# 7. Bölüm

### **Modellerin Değerlendirilmesi**

Zaman serisi modelleri tahmin yaparken kullanılan çeşitli performans metrikleri ve model seçimi kriterleri vardır. Bu metrikler, modelin ne kadar iyi çalıştığını ölçmek için kullanılır.

---

### **MSE (Mean Squared Error)**

**MSE**, tahmin edilen değerler ile gerçek değerler arasındaki farkların karesinin ortalamasını alır. Karesi alındığı için büyük hatalar daha fazla ceza alır, bu yüzden MSE büyük sapmaları vurgular.

- **Yorumlama**: MSE düşük olduğunda modelin hatası azdır ve daha iyi tahminler yapmaktadır.

### **MAE (Mean Absolute Error)**

**MAE**, tahmin edilen değerler ile gerçek değerler arasındaki farkların mutlak değerlerinin ortalamasını alır. MSE’ye göre daha basit ve yorumlaması kolaydır, çünkü karesel hata yerine doğrusal bir hatayı ölçer.

- **Yorumlama**: MAE, hatanın büyüklüğünü doğrudan ifade eder ve kolayca yorumlanabilir.

### **MAPE (Mean Absolute Percentage Error)**

**MAPE**, tahmin edilen değerler ile gerçek değerler arasındaki hatanın, gerçek değere göre yüzdesel olarak ifadesidir. Bu, tahmin hatalarını yüzdelik olarak ifade ettiği için diğer metriklere göre daha anlaşılır olabilir.

- **Yorumlama**: MAPE, hatanın ortalama yüzdesini gösterir. Örneğin, %10 MAPE, tahminlerde ortalama %10 hata yapıldığını gösterir.

### **AIC (Akaike Information Criterion) ve BIC (Bayesian Information Criterion)**

Bu iki kriter, zaman serisi modellerinin karşılaştırılmasında ve seçiminde kullanılır. Her iki kriter de modelin doğruluğunu ve karmaşıklığını dengeleyerek en iyi modeli seçmeye yardımcı olur.

### **AIC (Akaike Information Criterion)**

**AIC**, modelin karmaşıklığını ve veriye ne kadar iyi uyduğunu dengeleyen bir kriterdir. AIC değeri düşük olan model, hem daha iyi hem de daha basit bir model olarak kabul edilir.

- **Yorumlama**: AIC değeri ne kadar düşükse model o kadar iyidir. Ancak, sadece AIC değeri düşük olan modellerin aşırı karmaşık olmaması da gerekir.

### **BIC (Bayesian Information Criterion)**

**BIC**, AIC’ye benzer bir kriterdir ancak daha karmaşık modelleri daha fazla cezalandırır. Özellikle daha büyük veri setlerinde BIC kullanımı daha uygun olabilir.

- **Yorumlama**: BIC, genellikle AIC’den daha katı bir kriterdir. Model seçerken BIC’nin düşük olduğu model daha çok tercih edilir.

---

### **Sonuç**

Zaman serisinin durağanlığa dönüştürülmesi, tahmin modellerinin başarısı için önemli bir adımdır. **Fark alma**, **logaritmik dönüşüm** ve **trend çıkarma** gibi yöntemler, seriyi durağan hale getirirken, tahmin modellerinin performansını ölçmek için **MSE**, **MAE**, **MAPE** gibi hata metrikleri kullanılır. Ayrıca, **AIC** ve **BIC** kriterleri model seçimini kolaylaştırarak en uygun modeli belirlemede kritik rol oynar.

---

# 8. Bölüm

### **Python ile Zaman Serisi Analizi**

Zaman serisi analizinde Python, güçlü kütüphaneleriyle bu alandaki temel ve ileri düzey yöntemlerin uygulanmasına olanak tanır. Python'da zaman serisi verilerinin işlenmesi ve modellenmesi için kullanılan başlıca kütüphaneler **Pandas**, **Statsmodels**, **Prophet** ve **Scikit-learn** gibi araçlardır. Bu başlık altında zaman serisi analizini adım adım inceleyeceğiz.

---

### **Zaman Serisi Verilerinin İşlenmesi (Pandas)**

**Pandas** kütüphanesi, zaman serisi verilerini işlemekte en yaygın kullanılan Python kütüphanelerinden biridir. Zaman serisi verilerini analiz etmek için tarih ve saat bilgilerine duyarlı olan **`DatetimeIndex`** yapısı ile çalışır.

### **Zaman Serisi Verisinin Yüklenmesi**

Zaman serisi verileri genellikle CSV, Excel gibi dosyalarda tutulur ve Pandas ile kolayca yüklenebilir.

```python
import pandas as pd

# CSV dosyasından veri yükleme
df = pd.read_csv('zaman_serisi.csv', parse_dates=['Tarih'], index_col='Tarih')
```

### **Veri Manipülasyonu**

Pandas ile zaman serisi verisi üzerinde manipülasyonlar yapabiliriz.

- **Zaman Aralıklarına Göre Yeniden Örnekleme (Resampling)**: Haftalık, aylık veya yıllık bazda ortalamaları almak için kullanılır.

```python
# Haftalık ortalama
weekly_avg = df.resample('W').mean()

# Aylık toplam
monthly_sum = df.resample('M').sum()
```

- **Kaydırma (Shifting)**: Zaman serisini belirli bir süre ileri veya geri kaydırabiliriz.

```python
# Bir ay ileri kaydırma
shifted = df.shift(1)
```

- **Rolling (Hareketli Ortalama)**: Hareketli ortalama hesaplamak için kullanılır.

```python
# 3 günlük hareketli ortalama
rolling_avg = df.rolling(window=3).mean()
```

---

### **Zaman Serisi Modellerinin Uygulanması (Statsmodels, Prophet)**

### **Statsmodels**

**Statsmodels**, zaman serisi analizinde kullanılan birçok klasik modeli sunar: **AR**, **MA**, **ARMA**, **ARIMA**, **SARIMA**, vb. Bu modeller hem mevsimsel hem de mevsimsel olmayan veri setlerine uygulanabilir.

- **ARIMA Modeli Uygulaması**:

```python
import statsmodels.api as sm
# ARIMA modeli için veri setini hazırlama
model = sm.tsa.ARIMA(df['veri'], order=(p, d, q))  # p, d, q değerlerini belirleyin
arima_model = model.fit()

# Tahmin yapma
forecast = arima_model.forecast(steps=10)  # 10 adımlık tahmin
print(forecast)
```

### **Prophet**

**Facebook Prophet**, zaman serisi tahminleri için kullanılan güçlü bir kütüphanedir ve mevsimsel, trend ve tatil etkilerini tahmin edebilir. Prophet, eksik veri ve düzensiz örnekleme gibi durumlarla da baş edebilir. Kullanımı oldukça basittir ve genellikle günlük veya haftalık verilere uygulanır.

- **Prophet Modeli Uygulaması**:

```python
from fbprophet import Prophet

# Prophet için veri setini hazırlama
df.reset_index(inplace=True)
df_prophet = df.rename(columns={'Tarih': 'ds', 'veri': 'y'})

# Modeli oluşturma ve eğitme
model = Prophet()
model.fit(df_prophet)

# Gelecek tahmini (örneğin 365 gün için)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Sonuçların görselleştirilmesi
model.plot(forecast)
```

---

### **Mevsimsellik ve Trendlerin Ayrıştırılması (Seasonal Decompose)**

Zaman serisi verisinin **mevsimsellik**, **trend** ve **gürültü** bileşenlerine ayrılması, seriyi daha iyi anlamak ve modellemeyi geliştirmek için önemlidir. Python’da **`statsmodels`** kütüphanesi ile bu işlemi gerçekleştirebiliriz.

- **Seasonal Decompose Uygulaması**:

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Mevsimsellik ve trendin ayrıştırılması
decomposition = seasonal_decompose(df['veri'], model='additive', period=12)

# Bileşenlerin görselleştirilmesi
decomposition.plot()
```

Bu işlem, zaman serisini **trend**, **mevsimsellik** ve **rezidü** (gürültü) olarak ayrıştırır.

---

### **SARIMA ile Mevsimsel Tahminler**

**SARIMA (Seasonal ARIMA)** modeli, mevsimsel bileşenleri içeren zaman serileri için ARIMA modelinin genişletilmiş halidir. SARIMA modeli, mevsimsel etkilerin olduğu serilerde daha başarılı tahminler sağlar.

- **SARIMA Modeli Uygulaması**:

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA modeli tanımlama
sarima_model = SARIMAX(df['veri'], order=(p, d, q), seasonal_order=(P, D, Q, m))
sarima_fit = sarima_model.fit()

# Tahmin yapma
forecast = sarima_fit.forecast(steps=12)  # 12 adım ileriye dönük tahmin
print(forecast)
```

---

### **Hyperparameter Tuning (Hiperparametre Ayarlaması)**

Zaman serisi modellerinde **p**, **d**, **q** gibi hiperparametrelerin doğru seçilmesi, modelin tahmin performansını önemli ölçüde etkiler. Hiperparametre ayarlaması, modelin doğruluğunu artırmak için parametrelerin optimize edilmesini sağlar.

- **Grid Search ile Hiperparametre Ayarlaması** (SARIMA Örneği):

```python
import itertools
import numpy as np

# SARIMA için parametre aralığı belirleme
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

best_aic = np.inf
best_param = None

# Parametreler arasında döngü
for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            mod = SARIMAX(df['veri'], order=param, seasonal_order=seasonal_param)
            results = mod.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = (param, seasonal_param)
        except:
            continue

print('En iyi parametreler:', best_param)
```

Bu yöntemle, en uygun SARIMA parametrelerini bulabilir ve tahminlerin doğruluğunu artırabilirsiniz.

---

### **Sonuç**

Python ile zaman serisi analizi, güçlü kütüphaneler sayesinde hem basit hem de karmaşık modellere olanak sağlar. **Pandas** ile veri işleme, **Statsmodels** ve **Prophet** ile modelleme, **Seasonal Decompose** ile trend ve mevsimsellik ayrıştırma, **SARIMA** ile mevsimsel tahminler ve **hiperparametre ayarlaması** ile model optimizasyonu gibi adımlar, zaman serisi analizinde temel süreçlerdir.

---

# 9. Bölüm

### **Uygulamalar ve Case Study'ler**

Zaman serisi analizinin geniş uygulama alanları bulunmaktadır. Finansal piyasa verilerinden ekonomik göstergelere, satış tahminlerinden doğal olaylara kadar birçok alanda zaman serisi analizini uygulamak mümkündür. Bu başlık altında, farklı alanlardaki zaman serisi analizine dayanan uygulamaları ve vaka çalışmalarını inceleyeceğiz.

---

### **1. Finansal Piyasa Verilerinde Zaman Serisi Analizi**

Finansal piyasalar sürekli değişen dinamiklere sahiptir, bu nedenle zaman serisi analizleri bu piyasalar için oldukça kritiktir. Yatırımcılar, hisse senedi fiyatlarının, döviz kurlarının veya tahvil getirilerinin gelecekteki davranışını tahmin etmek için zaman serisi modellerini kullanırlar.

### **Örnek: Hisse Senedi Fiyat Tahmini (ARIMA ile)**

- **Problem:** Geçmiş hisse senedi fiyatlarını kullanarak gelecekteki fiyatları tahmin etmek.
- **Yaklaşım:** ARIMA modeli, geçmiş veri desenlerini kullanarak gelecekteki fiyatların ne olacağını öngörmek için kullanılır.

**Uygulama:**

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Veri yükleme
df = pd.read_csv('hisse_senedi_fiyatlari.csv', index_col='Tarih', parse_dates=True)

# ARIMA modeli oluşturma
model = ARIMA(df['Fiyat'], order=(5, 1, 0))
model_fit = model.fit()

# Tahmin yapma
forecast = model_fit.forecast(steps=30)  # 30 günlük tahmin
print(forecast)
```

**Sonuç:** Bu modelle, hisse senedi fiyatlarının gelecekteki değişimlerini tahmin ederek yatırım stratejilerini belirlemek mümkündür.

---

### **2. Ekonomik Verilerle ARIMA Tahminleri**

Ekonomik göstergeler, merkez bankaları ve hükümetler için kritik öneme sahiptir. Enflasyon, işsizlik oranı ve GSYİH gibi göstergelerin zaman serisi modelleriyle analizi, gelecekteki ekonomik koşulların öngörülmesine yardımcı olabilir.

### **Örnek: Enflasyon Tahmini (ARIMA ile)**

- **Problem:** Geçmiş enflasyon oranlarını kullanarak gelecekteki oranları tahmin etmek.
- **Yaklaşım:** ARIMA modeli, ekonomik verilerdeki trend ve mevsimsel etkileri modellemek için kullanılır.

**Uygulama:**

```python
# Enflasyon verileri ile ARIMA tahmini
model = ARIMA(df['Enflasyon'], order=(2, 1, 2))
model_fit = model.fit()

# 12 aylık tahmin yapma
forecast = model_fit.forecast(steps=12)
print(forecast)
```

**Sonuç:** Bu tahminler, merkez bankalarının faiz politikalarını ve hükümetlerin maliye politikalarını şekillendirmek için kullanılabilir.

---

### **3. Satış Tahmini ve Stok Yönetimi**

Perakende sektöründe şirketler, satışları tahmin ederek stok yönetimini optimize etmeye çalışır. Zaman serisi analizi, satışların gelecekte nasıl değişeceğini öngörerek stokların yeterli seviyede tutulmasını sağlar.

### **Örnek: Mağaza Satış Tahmini (SARIMA ile)**

- **Problem:** Geçmiş satış verilerini kullanarak gelecekteki satışları tahmin etmek.
- **Yaklaşım:** SARIMA modeli, satışlarda gözlenen mevsimsel desenleri de dikkate alarak tahmin yapar.

**Uygulama:**

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# SARIMA modeli ile satış tahmini
sarima_model = SARIMAX(df['Satış'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit()

# 6 aylık satış tahmini
forecast = sarima_fit.forecast(steps=6)
print(forecast)
```

**Sonuç:** Bu tahminler, mağaza sahiplerinin gelecekteki stok ihtiyaçlarını ve siparişleri daha doğru planlamalarına yardımcı olur.

---

### **4. Doğal Olayların Zaman Serisi Modelleri (Meteoroloji, Depremler)**

Zaman serisi analizi, meteoroloji ve depremler gibi doğal olayları tahmin etmekte de kullanılır. Örneğin, hava durumu tahminlerinde sıcaklık, yağış miktarı gibi veriler zaman serisi modelleriyle analiz edilir. Benzer şekilde, depremler için sismik verilerin analizi yapılabilir.

### **Örnek: Sıcaklık Tahmini (Exponential Smoothing ile)**

- **Problem:** Geçmiş sıcaklık verilerini kullanarak gelecekteki sıcaklıkları tahmin etmek.
- **Yaklaşım:** Exponential Smoothing yöntemi, özellikle kısa vadeli tahminlerde etkilidir.

**Uygulama:**

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Üstel Düzeltme ile sıcaklık tahmini
model = ExponentialSmoothing(df['Sıcaklık'], trend='add', seasonal='add', seasonal_periods=12)
fit = model.fit()

# Gelecek 12 aylık sıcaklık tahmini
forecast = fit.forecast(steps=12)
print(forecast)
```

**Sonuç:** Bu tür tahminler, tarım, inşaat ve enerji sektörlerinde önemli kararlar almaya yardımcı olur.

---

### **5. COVID-19 Salgını ve Epidemiyolojik Modeller**

COVID-19 salgını, dünya genelinde vaka ve ölüm sayılarının tahmin edilmesini önemli hale getirdi. Epidemiyolojik zaman serisi modelleri, salgının yayılma hızını ve gelecekteki vaka sayısını tahmin ederek halk sağlığı politikalarına yön verir.

### **Örnek: COVID-19 Vaka Sayısı Tahmini (ARIMA ile)**

- **Problem:** Geçmiş COVID-19 vaka sayısını kullanarak gelecekteki vakaları tahmin etmek.
- **Yaklaşım:** ARIMA modeli, salgının yayılma hızındaki değişiklikleri tahmin etmek için kullanılır.

**Uygulama:**

```python
# COVID-19 vaka sayısı tahmini
model = ARIMA(df['Vaka_Sayısı'], order=(5, 1, 2))
model_fit = model.fit()

# 30 günlük vaka tahmini
forecast = model_fit.forecast(steps=30)
print(forecast)
```

**Sonuç:** Bu tür analizler, sağlık otoritelerinin salgına karşı alacakları önlemleri daha iyi planlamalarına olanak tanır.

---

### **Sonuç**

Zaman serisi analizi, finans, ekonomi, perakende, doğal olaylar ve sağlık gibi birçok alanda uygulanabilir. Her bir alan için farklı zaman serisi modelleri ve tahmin yöntemleri kullanılabilir. Bu tür analizler, gelecekteki olayları tahmin etmenin yanı sıra stratejik kararlar almak için de kritik öneme sahiptir.

---

# 10. Bölüm

### **İleri Düzey Konular**

Zaman serisi analizi, temel yöntemlerin ötesinde, derin öğrenme, büyük veri, anomali tespiti ve nedensellik analizi gibi ileri düzey konuları da kapsar. Bu başlık altında, bu ileri düzey konulara dair detaylı bir inceleme yapacağız.

---

### **1. Deep Learning ve Zaman Serisi Analizi**

Derin öğrenme, zaman serisi verilerinin analizi için güçlü araçlar sunar. RNN (Recurrent Neural Networks) ve LSTM (Long Short-Term Memory) gibi modeller, zaman serisi verilerindeki bağımlılıkları yakalamak için kullanılır.

### **RNN (Recurrent Neural Networks)**

RNN, zaman serisi verilerindeki sıralı bağımlılıkları modellemek için kullanılır. Temel RNN, önceki zaman adımlarındaki bilgileri hatırlama yeteneğine sahiptir, ancak uzun vadeli bağımlılıkları yakalamakta zorluk çekebilir.

**Örnek: RNN ile Zaman Serisi Tahmini**

```python
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Veriyi yükleme ve hazırlama
df = pd.read_csv('zaman_serisi.csv', index_col='Tarih', parse_dates=True)
data = df['Değer'].values

# Veriyi zaman serisi olarak yeniden şekillendirme
X = np.array([data[i:i+10] for i in range(len(data)-10)])
y = np.array([data[i+10] for i in range(len(data)-10)])

# RNN Modeli oluşturma
model = Sequential()
model.add(SimpleRNN(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X, y, epochs=20, batch_size=32)
```

### **LSTM (Long Short-Term Memory)**

LSTM, RNN'in uzun vadeli bağımlılıkları öğrenme yeteneğini geliştirmiş bir versiyonudur. Bellek hücreleri sayesinde bilgi kaybını minimize eder ve daha uzun süreli bağımlılıkları yakalar.

**Örnek: LSTM ile Zaman Serisi Tahmini**

```python
from keras.layers import LSTM

# LSTM Modeli oluşturma
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X, y, epochs=20, batch_size=32)
```

---

### **2. Büyük Veride Zaman Serisi (Big Data ve Spark)**

Büyük veri analitiğinde zaman serisi verileri, özellikle yüksek frekansta veri toplandığında büyük boyutlara ulaşabilir. **Apache Spark** gibi dağıtık işlem sistemleri, bu tür büyük veri kümeleriyle çalışmak için kullanılır.

### **Spark ile Zaman Serisi Analizi**

Spark, büyük veri işlemleri için güçlü bir araçtır ve **PySpark** ile zaman serisi analizini gerçekleştirebiliriz.

**Örnek: PySpark ile Zaman Serisi Analizi**

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType

# Spark oturumu başlatma
spark = SparkSession.builder.appName("ZamanSerisiAnalizi").getOrCreate()

# Veri yükleme
df = spark.read.csv('zaman_serisi.csv', header=True, inferSchema=True)

# Zaman damgası dönüştürme
df = df.withColumn('Tarih', col('Tarih').cast(TimestampType()))

# Zaman serisi analizi yapmak için veriyi hazırlama
df.createOrReplaceTempView("zaman_serisi")
result = spark.sql("SELECT * FROM zaman_serisi WHERE Tarih > '2023-01-01'")

# Sonuçları görme
result.show()
```

---

### **3. Anomali Tespiti ve Zaman Serisi**

Zaman serisi verilerindeki anomali tespiti, olağandışı veya beklenmedik durumları belirlemek için kullanılır. Anomaliler, finansal dolandırıcılık, makine arızaları veya sağlık sorunları gibi durumları tespit etmek için kritik öneme sahiptir.

### **Örnek: Anomali Tespiti için ARIMA ve Z-Score Kullanımı**

**ARIMA Modeli ile Anomali Tespiti**

```python
from statsmodels.tsa.arima_model import ARIMA

# ARIMA Modeli ile tahmin yapma
model = ARIMA(df['Veri'], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.predict(start=len(df), end=len(df)+10)

# Anomali tespiti için Z-Score hesaplama
from scipy.stats import zscore

df['Z_Score'] = zscore(df['Veri'])
anomalies = df[df['Z_Score'].abs() > 3]  # Z-Score 3'ten büyük olanlar anomali olarak kabul edilir

print(anomalies)
```

### **Örnek: LSTM ile Anomali Tespiti**

LSTM modelini, normal ve anormal verileri öğrenmesi için eğitip, tahminler yaparak anomali tespiti gerçekleştirebiliriz.

```python
from keras.layers import LSTM, Dense
from keras.models import Sequential

# Model oluşturma
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X, y, epochs=20, batch_size=32)

# Tahmin yapma
predictions = model.predict(X)

# Anomali tespiti
errors = np.abs(predictions - y)
anomalies = np.where(errors > np.percentile(errors, 95))  # Örnek olarak %95'lik dilim
print(anomalies)
```

---

### **4. Granger Nedenselliği (Granger Causality)**

Granger nedenselliği, iki zaman serisi arasındaki nedenselliği test etmek için kullanılan bir yöntemdir. Bir zaman serisinin diğerini tahmin edebilme gücünü değerlendirir. Granger nedenselliği, bir serinin gelecekteki değerlerini tahmin etmek için diğer serilerin bilgi taşıyıp taşımadığını test eder.

### **Örnek: Granger Nedenselliği Testi**

```python
from statsmodels.tsa.stattools import grangercausalitytests

# İki zaman serisi arasında Granger nedenselliği testi
# df1 ve df2: iki farklı zaman serisi
results = grangercausalitytests(df[['df1', 'df2']], maxlag=4, verbose=True)

# Sonuçları yorumlama
for lag, result in results.items():
    f_statistic, p_value = result[0]['ssr_ftest'][0], result[0]['ssr_ftest'][1]
    print(f"Lag: {lag}, F-Statistik: {f_statistic}, p-değeri: {p_value}")
```

**Sonuçlar:** Eğer p-değeri 0.05'ten küçükse, bir zaman serisi diğerini Granger nedenselliğine sahiptir denir.

---

### **Sonuç**

İleri düzey zaman serisi analizinde, derin öğrenme yöntemleri (RNN, LSTM), büyük veri sistemleri (Spark), anomali tespiti ve nedensellik analizleri önemli bir yer tutar. Derin öğrenme, özellikle uzun vadeli bağımlılıkların modellemesinde etkili olurken, büyük veri sistemleri büyük veri kümeleri ile çalışmayı mümkün kılar. Anomali tespiti, olağandışı durumları belirlemekte kritik rol oynar ve Granger nedenselliği, iki zaman serisi arasındaki nedenselliği analiz eder. Bu ileri düzey konular, zaman serisi analizinin gücünü ve kapsamını genişleterek daha derinlemesine ve kapsamlı analizler yapılmasına olanak sağlar.