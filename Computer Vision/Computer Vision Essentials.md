# Bilgisayarlı Görü - Metehan Ayhan

## **Bilgisayarlı Görüye Giriş**

### **1.1 Bilgisayarlı Görü Nedir?**

Bilgisayarlı görü, bilgisayarların görüntüleri analiz etmesini ve anlamasını sağlayan bir disiplin olup, çeşitli uygulamalar için görüntü verilerini işleyerek bilgi çıkarımı yapar. Bu alan, yapay zeka ve bilgisayarlı görü algoritmalarını kullanarak görüntülerden anlamlı bilgiler elde etmeyi amaçlar.

### **Görüntü İşleme ve Bilgisayarlı Görü Farkı**

- **Görüntü İşleme:** Görüntülerin sayısal olarak işlenmesini içerir. Amaç, görüntüyü iyileştirmek veya belirli özellikleri çıkarmaktır.
- **Bilgisayarlı Görü:** Görüntülerden anlam çıkarmak için kullanılan algoritmaları ve sistemleri içerir. Bu, nesne tanıma, yüz tanıma gibi daha karmaşık görevleri kapsar.

### **1.2 Tarihçe ve Gelişim**

Bilgisayarlı görü, 1960'ların sonlarında basit görüntü işleme teknikleriyle başlamış, 1980'lerde ve 1990'larda önemli gelişmeler kaydetmiştir. Derin öğrenme ve büyük veri çağının başlamasıyla, bilgisayarlı görü alanında büyük bir devrim yaşanmıştır.

### **Erken Dönem Araştırmaları**

- **1960'lar:** İlk görüntü işleme algoritmaları ve temel araştırmalar.
- **1980'ler:** İlk nesne tanıma algoritmaları ve görüntü segmentasyon teknikleri.
- **1990'lar:** Makine öğrenmesi ve sınıflandırma yöntemlerinin gelişimi.

### **Modern Dönem**

- **2000'ler:** Derin öğrenmenin gelişimi ve büyük veri setlerinin kullanılmaya başlanması.
- **2010'lar ve sonrası:** CNN'lerin başarısı, medikal görüntüleme ve otonom araçlarda yaygın uygulama.

### **1.3 Temel Kavramlar**

Bilgisayarlı görünün temel kavramları, görüntülerin nasıl temsil edildiği, analiz edildiği ve işleme alındığına dair bilgi sağlar. Bu kavramlar, görüntü verilerinin nasıl işleneceği ve analiz edileceği hakkında bilgi verir.

### **Görüntü Temsili**

- **Pikseller:** Görüntüler, piksellerden oluşur ve her piksel belirli bir renk ve yoğunluk değerine sahiptir.
- **Renk Uzayları:** Renkler, RGB, HSV gibi farklı renk uzaylarında temsil edilir.

### **Özellik Çıkartma**

- **Kenar Algılama:** Görüntülerdeki önemli kenarları ve sınırları belirler.
- **Nokta Özellikleri:** Önemli noktaları ve özellikleri çıkarır.

---

## **Bölüm 2: Görüntü İşleme Temelleri**

### **2.1 Görüntü Temsil ve Ön İşleme**

Görüntü temsili, görüntülerin sayısal veriler olarak nasıl temsil edildiğini içerir. Ön işleme, görüntülerin daha iyi analiz edilebilmesi için yapılan işlemleri kapsar.

### **Görüntü Temsili**

- **Gri Ölçekli Görüntüler:** Tek bir renk kanalına sahip görüntüler.
- **Renkli Görüntüler:** RGB veya diğer renk uzaylarında temsil edilen görüntüler.

### **Ön İşleme Teknikleri**

- **Gürültü Giderme:** Görüntülerdeki istenmeyen gürültüleri azaltmak için kullanılan teknikler.
- **Normalizasyon:** Görüntülerin belirli bir aralığa dönüştürülmesi.

### **2.2 Filtreleme ve Kenar Algılama**

Filtreleme, görüntülerdeki önemli detayları ortaya çıkarmak için kullanılır. Kenar algılama, görüntülerdeki kenarları ve sınırları belirlemek için kullanılan bir tekniktir.

### **Filtreleme Teknikleri**

- **Sobel ve Prewitt Filtreleri:** Kenarları belirlemek için kullanılan basit filtreler.
- **Gaussian Filtreleri:** Gürültüyü azaltmak ve görüntüleri yumuşatmak için kullanılır.

### **Kenar Algılama**

- **Canny Kenar Algılama:** Görüntülerdeki keskin kenarları tespit eden bir algoritma.
- **Hough Dönüşümü:** Doğruları ve diğer şekilleri tespit eder.

### **2.3 Histogram ve Renk Dönüşümleri**

Histogramlar, görüntüdeki piksel değerlerinin dağılımını gösterir. Renk dönüşümleri ise farklı renk uzayları arasında geçiş yaparak görüntülerin analizini kolaylaştırır.

### **Histogramlar**

- **Histogram Eşitleme:** Kontrastı artırmak için kullanılan bir tekniktir.
- **Renk Histogramları:** Renk bileşenlerinin dağılımını gösterir.

### **Renk Dönüşümleri**

- **RGB'den HSV'ye:** Renk uzayları arasında dönüşüm.
- **Renk Dönüşüm Matrisleri:** Görüntülerin farklı renk uzaylarında temsil edilmesi.

---

## **Bölüm 3: Konvolüsyonel Sinir Ağları (CNN) ile Bilgisayarlı Görü**

### **3.1 Konvolüsyonel Sinir Ağları ve Mimarileri**

Konvolüsyonel Sinir Ağları (CNN'ler), görüntülerin otomatik olarak özelliklerini çıkarmak için kullanılan derin öğrenme modelleridir. Bu bölümde CNN'lerin temel yapı taşları ve yaygın mimarileri ele alınır.

### **CNN Yapısı**

- **Konvolüsyon Katmanları:** Görüntü üzerindeki filtrelerin uygulandığı katmanlar.
- **Havuzlama Katmanları:** Görüntülerin boyutunu azaltarak önemli özellikleri çıkaran katmanlar.
- **Tam Bağlantılı Katmanlar:** Özelliklerin sınıflandırılmasına yardımcı olan katmanlar.

### **Yaygın Mimariler**

- **LeNet:** İlk CNN mimarilerinden biridir, genellikle basit görevlerde kullanılır.
- **AlexNet:** Daha derin bir ağ yapısına sahiptir ve büyük veri setlerinde başarılıdır.
- **VGG:** Daha derin ve daha karmaşık bir mimariye sahip olup, daha iyi performans sağlar.
- **ResNet:** Kalan bağlantılar kullanarak daha derin ağların eğitimini sağlar.

### **3.2 Görüntü Sınıflandırma**

Görüntü sınıflandırma, görüntüleri belirli kategorilere ayırma sürecidir. CNN'ler bu süreçte oldukça etkilidir.

### **Sınıflandırma Teknikleri**

- **Öznitelik Çıkartma ve Sınıflandırma:** Görüntülerden özniteliklerin çıkarılması ve sınıflandırıcılarla sınıflandırılması.
- **Aktivasyon Fonksiyonları:** ReLU, sigmoid, ve softmax gibi aktivasyon fonksiyonlarının kullanımı.

### **Veri Setleri**

- **MNIST:** El yazısı rakamları içeren bir veri seti.
- **CIFAR-10:** 10 farklı sınıftan oluşan renkli görüntüler içeren veri seti.

### **3.3 Nesne Algılama ve Segmentasyon**

Nesne algılama, görüntülerdeki belirli nesneleri tanıma ve konumlandırma sürecidir. Segmentasyon ise görüntüleri farklı bölgelere ayırma işlemidir.

### **Nesne Algılama Teknikleri**

- **YOLO (You Only Look Once):** Tek bir ağ ile nesne tespiti ve sınıflandırma.
- **SSD (Single Shot Multibox Detector):** Farklı nesne boyutlarını tespit etmek için kullanılır.

### **Segmentasyon Teknikleri**

- **U-Net:** Tıbbi görüntü segmentasyonu için yaygın olarak kullanılan bir mimari.
- **Mask R-CNN:** Nesne algılama ve segmentasyonu aynı anda gerçekleştiren bir ağ.

---

## **Bölüm 4: İleri Görüntü Analizi Teknikleri**

### **4.1 Yüz Tanıma ve Özellik Çıkartma**

Yüz tanıma, kişileri tanımak ve kimliklerini doğrulamak için kullanılan bir tekniktir. Özellik çıkartma, yüz özelliklerini tanımlamak için kullanılır.

### **Yüz Tanıma Teknikleri**

- **Eigenfaces ve Fisherfaces:** Yüzlerin özniteliklerini çıkaran klasik yöntemler.
- **DeepFace ve FaceNet:** Derin öğrenme tabanlı modern yüz tanıma yöntemleri.

### **Özellik Çıkartma**

- **Haar Cascades:** Yüz ve diğer nesneleri tespit etmek için kullanılan bir yöntem.
- **LBP (Local Binary Patterns):** Yüz özelliklerini çıkaran bir yöntem.

### **4.2 Nesne Takibi ve Video Analizi**

Nesne takibi, hareketli nesnelerin video üzerinde takip edilmesini sağlar. Video analizi ise dinamik içerikleri analiz etmek için kullanılır.

### **Nesne Takibi Teknikleri**

- **Kalman Filtreleri:** Nesnelerin hareketlerini tahmin etmek için kullanılır.
- **YOLO ve SORT:** Nesne tespiti ve takip için kullanılan birleşik yöntemler.

### **Video Analizi**

- **Aksiyon Tanıma:** Videolarda belirli hareketlerin ve aksiyonların tanınması.
- **Anomali Tespiti:** Video akışında olağan dışı olayların tespiti.

### **4.3 3D Görüntüleme ve Derinlik Algılama**

3D görüntüleme, üç boyutlu yapıların analiz edilmesini sağlar. Derinlik algılama ise görüntülerdeki derinlik bilgisini çıkarır.

### **3D Görüntüleme Teknikleri**

- **LiDAR ve Stereo Görüntüleme:** Derinlik ve 3D yapıları elde etmek için kullanılan yöntemler.
- **Point Cloud ve Mesh:** 3D modelleme ve analiz için kullanılan veri yapıları.

### **Derinlik Algılama**

- **Depth Cameras:** Derinlik bilgisi sağlayan özel kameralar.
- **Stereovizyon:** İki veya daha fazla görüntü kullanarak derinlik hesaplama.

---

## **Bölüm 5: Bilgisayarlı Görü Uygulamaları ve İnovasyonlar**

### **5.1 Görüntü Sınıflandırma**

Görüntü sınıflandırma, görüntüleri belirli kategorilere ayırmak için kullanılan temel bilgisayarlı görü uygulamalarından biridir. Bu işlem, derin öğrenme yöntemleriyle büyük bir başarı elde etmiştir.

### **Görüntü Sınıflandırma Teknikleri**

- **Öznitelik Çıkartma ve Sınıflandırma:** Geleneksel yöntemlerde öznitelikler manuel olarak çıkarılırken, derin öğrenme ile özniteliklerin otomatik olarak öğrenilmesi sağlanır. CNN'ler (Convolutional Neural Networks) bu sürecin merkezindedir.
- **Aktivasyon Fonksiyonları:** Sınıflandırma ağlarında yaygın olarak kullanılan aktivasyon fonksiyonları arasında ReLU (Rectified Linear Unit), sigmoid ve softmax bulunur. ReLU, ağların daha hızlı öğrenmesini sağlar; sigmoid ve softmax, sınıflandırma sonuçlarının olasılıklarını hesaplar.

### **Görüntü Sınıflandırma Mimarileri**

- **LeNet:** İlk başarılı derin öğrenme tabanlı görüntü sınıflandırma modelidir. Özellikle el yazısı rakamları gibi basit görevlerde etkili olmuştur.
- **AlexNet:** 2012'de ImageNet yarışmasında büyük bir başarı yakalayarak derin ağların geniş çaplı kullanıma geçmesini sağlamıştır. Daha derin ve karmaşık bir yapıya sahiptir.
- **VGG:** Daha derin ağlar kullanarak, küçük filtreler ve derinlik artırımı ile yüksek başarı sağlar. Özellikle görüntü sınıflandırma görevlerinde yüksek performans gösterir.
- **ResNet:** Kalan bağlantılar (residual connections) kullanarak, çok derin ağların eğitimini mümkün kılar. Bu sayede daha derin ve etkili modeller oluşturulabilir.

### **Görüntü Sınıflandırma Veri Setleri**

- **MNIST:** El yazısı rakamları içeren bir veri setidir. Eğitim ve test için yaygın olarak kullanılır.
- **CIFAR-10:** 10 farklı sınıfa ait renkli görüntüler içeren veri setidir. Gelişmiş modellerin test edilmesinde kullanılır.
- **ImageNet:** Büyük ölçekli bir veri setidir ve derin öğrenme modellerinin genel görüntü tanıma yeteneklerini test etmek için kullanılır.

---

### **5.2 Nesne Algılama ve Segmentasyon**

Nesne algılama, bir görüntüdeki belirli nesneleri tanımlayıp konumlandırmak için kullanılır. Segmentasyon ise görüntüyü farklı bölgelere ayırma işlemidir.

### **Nesne Algılama Teknikleri**

- **YOLO (You Only Look Once):** Tek bir sinir ağı ile nesne tespiti ve sınıflandırma yapan bir tekniktir. Gerçek zamanlı nesne algılama yeteneği sunar.
- **SSD (Single Shot Multibox Detector):** Farklı ölçeklerde nesneleri tespit edebilen bir modeldir. YOLO'ya benzer şekilde, yüksek hız ve doğruluk sağlar.
- **Faster R-CNN:** Region Proposal Network (RPN) kullanarak, nesne algılamada daha yüksek doğruluk sağlar. Nesne tespiti ve sınıflandırma için kullanılır.

### **Segmentasyon Teknikleri**

- **U-Net:** Tıbbi görüntülerde organ ve lezyonları segmentlemek için kullanılan bir CNN mimarisidir. Özellikle küçük veri setlerinde etkili sonuçlar verir.
- **Mask R-CNN:** Nesne algılama ve segmentasyonu aynı anda gerçekleştiren bir ağdır. Her nesneye ait ayrıntılı maskeler sağlar.
- **DeepLab:** Çoklu ölçekli öznitelikler kullanarak segmentasyon yapar. Görüntülerdeki detayları ayrıntılı bir şekilde segmentler.

### **Nesne Takibi**

- **Kalman Filtreleri:** Nesnelerin hareketlerini tahmin etmek için kullanılan matematiksel bir yöntemdir. Özellikle video analizi ve izleme uygulamalarında kullanılır.
- **SORT (Simple Online and Realtime Tracking):** Nesne algılama sonuçlarından hareketli nesneleri takip eden bir algoritmadır. YOLO veya SSD ile birlikte kullanılabilir.

---

### **5.3 Yüz Tanıma ve Özellik Çıkartma**

Yüz tanıma, kişileri tanımak ve kimliklerini doğrulamak için kullanılan bir tekniktir. Özellik çıkartma, yüzlerin özniteliklerini tanımlamak için kullanılır.

### **Yüz Tanıma Teknikleri**

- **Eigenfaces ve Fisherfaces:** Yüzlerin özniteliklerini çıkaran klasik yöntemlerdir. Principal Component Analysis (PCA) ve Linear Discriminant Analysis (LDA) kullanılır.
- **DeepFace ve FaceNet:** Derin öğrenme tabanlı modern yüz tanıma yöntemleridir. DeepFace, Facebook tarafından geliştirilmiş bir modelken, FaceNet, Google tarafından geliştirilmiştir. Her iki yöntem de yüksek doğruluk sağlar.

### **Özellik Çıkartma**

- **Haar Cascades:** Yüzleri tespit etmek için kullanılan bir yöntemdir. Genellikle hızlı ve etkili sonuçlar sağlar.
- **LBP (Local Binary Patterns):** Yüz özniteliklerini çıkarmak için kullanılan bir yöntemdir. Yüzlerin yapı özelliklerini tanımlar.

---

### **5.4 3D Görüntüleme ve Derinlik Algılama**

3D görüntüleme, üç boyutlu yapıların analiz edilmesini sağlar. Derinlik algılama ise görüntülerdeki derinlik bilgisini çıkarır.

### **3D Görüntüleme Teknikleri**

- **LiDAR (Light Detection and Ranging):** Lazer ışınları kullanarak çevresindeki nesnelerin 3D haritasını çıkarır. Otonom araçlar ve coğrafi analizlerde kullanılır.
- **Stereo Görüntüleme:** İki veya daha fazla görüntü kullanarak derinlik bilgisini hesaplar. Paralel kameralar kullanılarak 3D yapı elde edilir.
- **Point Cloud ve Mesh:** 3D modelleme ve analiz için kullanılan veri yapılarıdır. Point cloud, nesnelerin 3D koordinatlarını içerirken, mesh, bu koordinatları birleştirerek yüzey oluşturur.

### **Derinlik Algılama**

- **Depth Cameras:** Derinlik bilgisi sağlayan özel kameralar kullanır. Kinect ve Intel RealSense gibi cihazlar bu kategoriye girer.
- **Stereovizyon:** İki görüntü arasındaki derinlik farkını hesaplayarak derinlik haritası oluşturur. Genellikle eşleşen özellikler kullanılarak yapılır.

---

## **Bölüm 6: Performans Değerlendirme ve Model Optimizasyonu**

### **6.1 Performans Değerlendirme Yöntemleri**

Performans değerlendirme, bir modelin başarısını ve doğruluğunu ölçmek için kullanılan çeşitli teknikleri içerir. Bu bölümde, model performansını değerlendirmek için kullanılan metrikler ve yöntemler ele alınacaktır.

### **Performans Metrikleri**

- **Doğruluk (Accuracy):** Modelin doğru tahmin ettiği örneklerin toplam örneklere oranıdır. Genel başarıyı ölçer, ancak dengesiz veri setlerinde yanıltıcı olabilir.
- **Hassasiyet (Precision):** Modelin pozitif olarak sınıflandırdığı örneklerin gerçekten pozitif olma oranıdır. Yanlış pozitifleri minimize etmek için kullanılır.
- **Duyarlılık (Recall):** Gerçek pozitif örneklerin model tarafından doğru tahmin edilme oranıdır. Yanlış negatifleri minimize etmeye çalışır.
- **F1 Skoru:** Hassasiyet ve duyarlılığın harmonik ortalamasıdır. Hem hassasiyet hem de duyarlılığı dikkate alır.
- **AUC-ROC (Area Under Curve - Receiver Operating Characteristic):** Modelin tüm sınıflandırma eşiklerinde performansını gösterir. Yüksek AUC değeri, iyi bir model performansını işaret eder.

### **Karmaşıklık ve Hata Analizi**

- **Karmaşıklık Matrisi (Confusion Matrix):** Modelin tahminlerini ve gerçek etiketleri karşılaştırarak, doğru ve yanlış sınıflandırmaları gösterir. Bu matris, doğruluk, hassasiyet, duyarlılık ve F1 skorunun hesaplanmasına yardımcı olur.
- **Hata Analizi:** Modelin yanlış sınıflandırdığı örnekleri inceleyerek, modelin zayıf noktalarını ve geliştirilmesi gereken alanları belirler.

### **Çapraz Doğrulama (Cross-Validation)**

- **K-Fold Çapraz Doğrulama:** Veriyi K eşit parçaya böler ve her parça, modelin eğitim ve testinde kullanılır. Bu yöntem, modelin genellenebilirliğini değerlendirmek için kullanılır.
- **Leave-One-Out Çapraz Doğrulama (LOOCV):** Her bir veri noktası bir test örneği olarak kullanılır ve model kalan verilerle eğitilir. Özellikle küçük veri setlerinde tercih edilir.

---

### **6.2 Model Optimizasyonu ve İyileştirme**

Model optimizasyonu, modelin performansını artırmak için kullanılan teknikleri içerir. Bu bölüm, modelin daha iyi performans göstermesi için uygulanan yöntemleri detaylandırır.

### **Hiperparametre Optimizasyonu**

- **Grid Search:** Belirlenen hiperparametrelerin tüm kombinasyonlarını deneyerek en iyi sonuçları bulur. Bu yöntem, geniş bir hiperparametre aralığını kapsar ancak zaman alıcı olabilir.
- **Random Search:** Hiperparametreler için rastgele kombinasyonlar dener. Grid search’e göre daha hızlı olabilir ancak tüm aralığı kapsamayabilir.
- **Bayesian Optimization:** Hiperparametre alanını modelleyerek, daha etkili ve verimli bir arama yapar. Bu yöntem, modelin performansını artırmak için iteratif olarak en iyi hiperparametreleri bulur.

### **Düzenlileştirme Teknikleri**

- **L1 ve L2 Düzenlileştirme:** Modelin karmaşıklığını azaltmak ve aşırı öğrenmeyi önlemek için kullanılır. L1 düzenlileştirme, özniteliklerin bazılarını sıfıra indirirken, L2 düzenlileştirme tüm özniteliklerin değerlerini küçültür.
- **Dropout:** Eğitim sırasında rastgele nöronları kapatarak modelin genellenebilirliğini artırır ve aşırı öğrenmeyi azaltır.
- **Batch Normalization:** Eğitim sırasında katmanlar arasındaki verilerin dağılımını normalleştirir. Bu, eğitim sürecini hızlandırır ve modelin performansını artırır.

### **Eğitim Süreci Optimizasyonu**

- **Öğrenme Oranı (Learning Rate):** Modelin ağırlıklarını güncellerken ne kadar adım atılacağını belirler. Uygun öğrenme oranı seçimi, modelin hızlı ve etkili bir şekilde öğrenmesini sağlar.
- **Momentum:** Öğrenme sürecinde önceki ağırlık güncellemelerinin etkisini korur ve eğitim sürecinin daha stabil olmasını sağlar.

### **Veri Augmentasyonu**

- **Veri Augmentasyonu Teknikleri:** Görüntü verileri için döndürme, ölçekleme, kaydırma ve aydınlatma değişiklikleri gibi yöntemlerle veri setini artırır. Bu, modelin genel performansını ve genellenebilirliğini iyileştirir.
- **Dengesiz Veri Setleriyle Başa Çıkma:** Sınıflar arasında dengesizlik olduğunda, örnekleme yöntemleri (aşırı örnekleme ve alt örnekleme) ve sınıf ağırlıkları kullanarak modelin dengesiz veri setlerine uyum sağlamasına yardımcı olur.

---

### **6.3 Model İzleme ve Bakımı**

Model izleme ve bakımı, modelin performansını sürekli olarak izlemek ve gerektiğinde güncellemeler yapmayı içerir.

### **Performans İzleme**

- **Canlı İzleme:** Modelin gerçek zamanlı performansını izlemek için kullanılan araçlar ve teknikler. Bu, modelin üretim ortamındaki başarısını değerlendirmek için önemlidir.
- **Performans Düşüşü Analizi:** Modelin performansının zamanla nasıl değiştiğini analiz ederek, modelin güncellenmesi gerekip gerekmediğini belirler.

### **Model Güncelleme**

- **Yeniden Eğitim:** Modelin performansını korumak için belirli aralıklarla yeniden eğitilmesi. Bu, yeni verilerle modelin güncellenmesini sağlar.
- **Versiyon Kontrolü:** Model değişikliklerini izlemek ve yönetmek için kullanılan teknikler. Modelin çeşitli versiyonlarının saklanması ve yönetilmesi, performans karşılaştırmaları için önemlidir.

### **Veri ve Model Güvenliği**

- **Veri Güvenliği:** Eğitim ve test verilerinin güvenliği ve gizliliği sağlanmalıdır. Veri sızıntıları ve yetkisiz erişimlere karşı önlemler alınmalıdır.
- **Model Güvenliği:** Modelin kötüye kullanımına karşı korunması. Modelin tahminleri üzerinde yapay etkiler oluşturabilecek saldırılara karşı dayanıklı olması gerekmektedir.

---

## **Sonuç**

Bu bölüm, model performansını değerlendirmek ve optimize etmek için kullanılan yöntemleri kapsamlı bir şekilde ele alır. Performans metriklerinden hiperparametre optimizasyonuna, veri augmentasyonundan model bakımına kadar geniş bir yelpazeyi kapsar. Bu teknikler, modelin başarısını artırmak ve verimli bir şekilde çalışmasını sağlamak için kritik öneme sahiptir.