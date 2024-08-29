# Doğal Dil İşleme - Metehan Ayhan

# **Doğal Dil İşleme (NLP - Natural Language Processing)**

yapay zeka ve dilbilim alanlarının kesişiminde yer alan bir teknoloji dalıdır. Temel amacı, bilgisayarların insan dilini anlamasını, işlemesini ve analiz etmesini sağlamaktır. Bu sayede, bilgisayarlar yazılı veya sözlü dili, tıpkı insanlar gibi anlama yeteneği kazanır.

### NLP'nin Temel Amaçları

- **Dil Anlama:** Bilgisayarların metin veya konuşma verilerini anlamalarını sağlamak. Örneğin, bir kullanıcının yazdığı bir mesajın anlamını çıkarabilmek.
- **Dil Üretme:** Bilgisayarların, anlamlı ve dilbilgisel olarak doğru cümleler üretmelerini sağlamak. Örneğin, bir sohbet botunun kullanıcıya mantıklı cevaplar verebilmesi.
- **Dil Tabanlı Tahmin:** Metinlerdeki veya konuşmalardaki eksik bilgileri tahmin etmek. Örneğin, bir kelimeyi doğru bir şekilde tamamlamak veya bir cümledeki hatayı düzeltmek.

### NLP'nin Gerçek Hayatta Kullanımı

NLP, günlük hayatımızda birçok yerde kullanılıyor:

- **Sohbet Botları:** Müşteri hizmetlerinde sıkça karşılaştığın otomatik yanıt sistemleri, kullanıcıların sorularını anlayıp cevap verebilmek için NLP teknolojilerini kullanır.
- **Çeviri Uygulamaları:** Google Translate gibi araçlar, bir dili başka bir dile çevirmek için NLP yöntemlerini kullanır.
- **Yazım ve Dilbilgisi Denetleyicileri:** Microsoft Word veya Grammarly gibi araçlar, yazım hatalarını bulup düzeltmek için NLP algoritmalarını kullanır.
- **Sesli Asistanlar:** Siri, Alexa veya Google Asistan gibi cihazlar, sesli komutları anlayarak yanıt vermek için NLP tekniklerinden yararlanır.

### 1. **Metin Özetleme (Text Summarization):**

- **Açıklama:** Metin özetleme, uzun bir metni daha kısa, ancak anlamını koruyan bir forma dönüştürme işlemidir. Burada amaç, en önemli bilgileri seçip özetlemektir.
- **Örnek:** Diyelim ki bir kitap hakkında bir özet yazmak istiyorsun. Kitap 300 sayfa, ancak sen bu kitabın özünü birkaç paragrafta ifade etmek istiyorsun. Metin özetleme algoritmaları bu süreci otomatik hale getirir. Örneğin, bir haber makalesinin önemli noktalarını alarak kısa bir özet oluşturabilir.

### 2. **Konu Modelleme (Topic Modeling):**

- **Açıklama:** Konu modelleme, büyük bir metin kümesinden ana temaları veya konuları çıkarmak için kullanılan bir yöntemdir. Bu, metindeki hangi konuların sıkça geçtiğini ve bu konuların nasıl gruplandırıldığını bulmak için kullanılır.
- **Örnek:** Bir blog sitesi yönettiğini düşün. Yüzlerce makale var ve her birinin hangi konuya odaklandığını bilmek istiyorsun. Konu modelleme, bu makalelerin ana temalarını belirlemede yardımcı olabilir, örneğin "yapay zeka", "makine öğrenmesi", "doğal dil işleme" gibi başlıklar altında gruplayabilir.

### 3. **Makine Çevirisi (Machine Translation):**

- **Açıklama:** Makine çevirisi, bir dilden başka bir dile otomatik çeviri yapma işlemidir. Burada hedef, orijinal anlamı koruyarak metni diğer bir dilde ifade edebilmektir.
- **Örnek:** Google Çeviri'yi düşün. Türkçe bir metni İngilizceye çevirmek istediğinde, bu araç otomatik olarak çeviriyi yapar. Örneğin, "Merhaba, nasılsın?" cümlesi "Hello, how are you?" olarak çevrilir. Bu işlemde kullanılan algoritmalar, cümle yapısını ve kelime anlamlarını dikkate alarak çeviriyi yapar.

### 4. **Konuşma Tanıma (Speech Recognition):**

- **Açıklama:** Konuşma tanıma, sesli konuşmaları yazılı metne dönüştürme işlemidir. Bu teknoloji, özellikle sesli komutlar veya dikte işlemleri için kullanılır.
- **Örnek:** Siri veya Google Asistan gibi sesli asistanları düşün. Telefona "Hava durumu nedir?" diye sorduğunda, asistan bu sesi alır ve yazılı metne dönüştürerek anlamlandırır. Sonrasında, hava durumu bilgilerini sana sunar.

### 5. **Lemmatizasyon (Lemmatization):**

- **Açıklama:** Lemmatizasyon, kelimeleri kök ya da temel hallerine indirgeme işlemidir. Bu, dilin morfolojik özelliklerini anlamak için kullanılır.
- **Örnek:** "Koşuyorum", "Koşuyoruz", "Koştular" gibi farklı çekimlerdeki kelimeler aslında aynı kökten gelir: "Koşmak". Lemmatizasyon algoritmaları bu farklı formları tespit eder ve kök formu olan "koşmak" ile eşleştirir.

### 6. **Sözcük Türü Etiketleme (Part-of-Speech Tagging):**

- **Açıklama:** Bu süreçte, bir cümledeki her kelime dilbilgisel kategorisine göre etiketlenir. Örneğin, bir kelimenin isim, fiil, sıfat vb. olduğunu belirlemek amaçlanır.
- **Örnek:** "Kedi hızlı koştu." cümlesinde "kedi" bir isim, "hızlı" bir sıfat ve "koştu" bir fiildir. Sözcük türü etiketleme algoritmaları, cümledeki her kelimenin dilbilgisel türünü otomatik olarak belirler.

### 7. **Dilbilgisi Hatası Düzeltme (Grammatical Error Correction):**

- **Açıklama:** Bu süreç, metinlerdeki dilbilgisel hataları tespit edip düzeltmeye odaklanır.
- **Örnek:** Bir öğrenci İngilizce bir yazı yazarken "She go to school every day." gibi bir cümle kurarsa, bu cümlede dilbilgisel bir hata vardır. Doğru cümle "She goes to school every day." olmalıdır. Bu tür hataları otomatik olarak düzelten araçlar, yazılı metinlerin kalitesini artırır.

### 8. **Akademik Atıf Ağı Analizi (Scholarly Citation Network Analysis):**

- **Açıklama:** Bu analiz, akademik makaleler arasındaki atıfları inceleyerek, bu atıflar arasındaki ilişkileri ve bağlantıları ortaya koyar.
- **Örnek:** Diyelim ki yapay zeka üzerine yazılmış bir makale okudun ve bu makale birçok başka çalışmaya atıfta bulunuyor. Atıf ağı analizi, bu makalenin hangi diğer çalışmalardan etkilendiğini ve bu çalışmaların birbirleriyle nasıl bağlantılı olduğunu gösterir.

### 9. **Varlık Adlandırma (Named Entity Linking):**

- **Açıklama:** Metinlerdeki belirli varlıkları tanıma ve bunları metinle ilişkilendirme işlemidir. Bu varlıklar genellikle kişi adları, yerler, kuruluşlar vb. olabilir.
- **Örnek:** Bir gazete makalesinde "Elon Musk Tesla'yı kurdu." cümlesi geçiyorsa, burada "Elon Musk" bir kişi adı, "Tesla" ise bir kuruluş olarak tanımlanabilir. Bu işlem, metin içerisindeki önemli varlıkları tespit etmede kullanılır.

### 10. **Metinden-Metine ve Metinden-Görüntüye Üretim (Text-to-Text and Text-to-Image Generation):**

- **Açıklama:** Bu teknoloji, verilen metin girdisine dayanarak yeni metinsel veya görsel içerik oluşturma işlemidir.
- **Örnek:** "Bir yaz günü sahil kenarında yürüyen bir kadın" ifadesini girdin diyelim. Metinden-görüntüye üretim, bu ifadeye uygun bir sahil resmi oluşturabilir. Ya da metinden-metine üretim, bu ifadeyi kullanarak daha detaylı bir öykü oluşturabilir.

---

## NLP Uygulama Alanları

### 1. **Makine Çevirisi (Machine Translation):**

- Bir dili başka bir dile otomatik olarak çevirmek için kullanılır. Örneğin, Google Translate gibi araçlar, metinleri veya konuşmaları bir dilden diğerine çevirmek için bu teknolojiyi kullanır.

### 2. **Konuşma Tanıma (Speech Recognition):**

- **Metinden Konuşmaya (Text to Speech)** ve **Konuşmadan Metne (Speech to Text)** çeviri yapar. Örneğin, sesli asistanlar (Siri, Google Asistan) kullanıcının konuşmalarını metne dönüştürür ve bu metni işler.

### 3. **Duygu Analizi (Sentiment Analysis):**

- Bir metnin pozitif, negatif ya da nötr duygular içerip içermediğini belirler. Örneğin, sosyal medya gönderilerindeki yorumları analiz ederek, halkın bir ürün ya da hizmet hakkında ne düşündüğünü anlamak için kullanılır.

### 4. **Soru Cevaplama (Question Answering):**

- Kullanıcıların sorduğu sorulara yanıt verebilen sistemlerdir. Siri, Alexa, Cortana gibi kişisel asistanlar ve sohbet botları (ChatGPT, POE) bu teknolojiyi kullanarak sorulara anlamlı cevaplar verir.

### 5. **Otomatik Özetleme (Automatic Summarization):**

- Uzun metinleri daha kısa ve özlü özetler haline getirir. Örneğin, bir haber makalesinin ana noktalarını içeren bir özet oluşturmak için kullanılır.

### 6. **Veri Gazeteciliği (Data Journalism):**

- Büyük veri kümelerini analiz ederek haber ve raporlar oluşturur. Gazeteciler, veri analizine dayalı olarak haberleri hazırlamak için NLP araçlarını kullanabilirler.

### 7. **Metin Sınıflandırma (Text Classification):**

- Metinleri belirli kategorilere ayırır. Örneğin, bir e-postayı "spam" veya "gelen kutusu" olarak sınıflandırmak bu teknoloji ile mümkündür.

### 8. **Karakter Tanıma (Character Recognition):**

- Yazılı karakterleri tarar ve dijital metne dönüştürür. OCR (Optik Karakter Tanıma) teknolojisi, basılı metinleri dijital metinlere dönüştürmek için kullanılır.

### 9. **Yazım Denetimi (Spell Checking):**

- Metinlerdeki yazım hatalarını tespit edip düzeltir. Word gibi kelime işlemciler, yazım denetimi yapmak için bu teknolojiyi kullanır.

Bu uygulamalar, NLP'nin geniş kullanım alanlarından sadece birkaçıdır ve günümüzde birçok endüstride ve günlük yaşamda büyük kolaylıklar sağlamaktadır.

---

## Veri biliminde ve makine öğreniminde

yaygın olarak kullanılan çeşitli mesafe ölçüm yöntemleri vardır. Bu yöntemler, iki veri noktası arasındaki benzerlik veya farklılıkları değerlendirmek için kullanılır. Şimdi her birini sırayla açıklayalım:

### 1. **Euclidean Distance (Öklidyen Mesafesi):**

- İki nokta arasındaki en kısa mesafeyi ölçer. Bir doğru üzerinde iki nokta arasındaki mesafe gibi düşünülebilir.
- Örnek: Bir şehirdeki iki konum arasındaki doğrudan mesafe.

### 2. **Manhattan Distance:**

- Noktalar arasındaki mesafeyi, sadece dikey ve yatay hareketlerle hesaplar. Adını, Manhattan'daki ızgara benzeri sokak düzeninden alır.
- Örnek: Bir şehirde, köşeler arasında yürürken alınan yol.

### 3. **Chebyshev Distance:**

- İki nokta arasındaki en büyük eksenel farkı kullanır. Satrançta bir kralın bir kareden diğerine giderken kat ettiği en kısa yol gibi düşünülebilir.
- Örnek: Satrançta, bir kareden diğerine gitmek için kralın aldığı adım sayısı.

### 4. **Minkowski Distance:**

- Euclidean ve Manhattan mesafelerinin genelleştirilmiş bir halidir. P parametresi ile belirlenir ve Euclidean mesafesi P=2, Manhattan mesafesi ise P=1 olduğunda elde edilir.
- Örnek: Euclidean ve Manhattan mesafelerini birleştiren bir ölçüm.

### 5. **Cosine Similarity (Kosinüs Benzerliği):**

- İki vektör arasındaki açıyı ölçer ve genellikle metin madenciliği gibi yüksek boyutlu veri kümesiyle çalışırken kullanılır.
- Örnek: İki belgenin içerik benzerliğini ölçmek.

### 6. **Pearson Correlation (Pearson Korelasyonu):**

- İki değişken arasındaki doğrusal ilişkiyi ölçer. 1, pozitif doğrusal ilişkiyi; -1, negatif doğrusal ilişkiyi; 0 ise ilişkisizlik durumunu gösterir.
- Örnek: Sıcaklık ile dondurma satışları arasındaki ilişki.

### 7. **Mahalanobis Distance:**

- Veri noktalarının ortalamadan ne kadar uzakta olduğunu, aynı zamanda veri kümesindeki değişkenlerin varyansını ve kovaryansını da dikkate alarak hesaplar.
- Örnek: Bir anormallik algılama algoritmasında kullanılması.

### 8. **Squared Euclidean Distance (SED - Karelenmiş Öklidyen Mesafesi):**

- Euclidean mesafesinin karesi olarak hesaplanır. Bu, büyük mesafeleri daha fazla cezalandırarak veri noktalarını ayırmada kullanılabilir.
- Örnek: Kümeleme algoritmalarında, noktalar arasındaki farklılıkları vurgulamak için kullanılır.

### 9. **Jaccard Similarity (Jaccard Benzerliği):**

- İki kümenin kesişimlerinin birleşimlerine oranını hesaplar. 1, tam benzerliği; 0 ise hiçbir ortak öğe olmadığını gösterir.
- Örnek: İki müşteri listesi arasındaki benzerliği ölçmek.

### 10. **Levenshtein Distance:**

- Bir dizeyi diğerine dönüştürmek için gereken minimum değişiklik (ekleme, silme, değiştirme) sayısını hesaplar.
- Örnek: Yazım denetleyicilerinde, kelime önerilerini bulmak için kullanılır.

### 11. **Sørensen-Dice Similarity:**

- Jaccard'a benzer, fakat kesişimi iki kat alarak daha duyarlı hale getirir.
- Örnek: Metin madenciliğinde, belgeler arasındaki benzerliği ölçmek için kullanılır.

### 12. **Jensen-Shannon Divergence:**

- İki olasılık dağılımı arasındaki farkı ölçer. Simetrik ve her zaman pozitif bir ölçümdür.
- Örnek: Bilgi teorisi alanında, farklı dağılımlar arasındaki benzerlikleri ölçmek için kullanılır.

### 13. **Canberra Distance:**

- Değerlerin büyüklük farklarını, bu değerlerin toplamına oranlayarak hesaplar. Bu, özellikle küçük farkların büyük olduğu durumlarda etkilidir.
- Örnek: Çevresel veri analizinde kullanılır.

### 14. **Hamming Distance:**

- İki dize arasındaki farklılıkları bit bazında ölçer. Hamming mesafesi, iki ikili sayı arasındaki farklı bitlerin sayısını verir.
- Örnek: Veri iletiminde hata tespit etme ve düzeltme algoritmaları.

### 15. **Spearman Correlation (Spearman Korelasyonu):**

- İki değişken arasındaki sıralama ilişkisinin gücünü ölçer. Non-parametrik bir korelasyon ölçütüdür.
- Örnek: Öğrencilerin sınav notları ile onların genel sıralamaları arasındaki ilişkiyi ölçmek.

### 16. **Chi-Square (Ki-Kare) Testi:**

- Gözlemlenen ve beklenen frekanslar arasındaki farkı ölçer. Çoğunlukla kategorik verilerde bağımsızlık testi için kullanılır.
- Örnek: Anket sonuçları ile beklenen sonuçlar arasındaki farkı analiz etmek.

Bu mesafe ölçümleri, farklı veri yapıları ve problem türlerine göre seçilerek kullanılır. Her biri, belirli bir bağlamda verilerin benzerliğini veya farklılığını değerlendirmek için uygundur.

---

## "Lexical"

(Türkçe: "leksikal") kelimesi, dilbilim alanında kullanılan bir terimdir ve genellikle kelime bilgisiyle veya kelimelerin anlamlarıyla ilgili konuları ifade eder. Özellikle, dilin kelime dağarcığına ve kelimelerin kullanımına dair olan her şeyi kapsar. "Lexical" terimi, kelimelerin anlamları, biçimleri ve kullanım yerleriyle ilgili çalışmaları içerir.

Örneğin, "lexical analysis" (leksikal analiz), bir dilin kelime yapılarını analiz etmeye yönelik bir süreçtir ve genellikle programlama dillerinde kelime işleme sırasında kullanılır. Bu analiz, metin içindeki kelimeleri (ya da token'ları) tanımlamak ve sınıflandırmak amacıyla yapılır.

Kısacası, "lexical", kelimelerin kendileriyle, onların anlamlarıyla ve nasıl kullanıldıklarıyla ilgilidir.

---

## Dil işleme sürecinin adımları:

### 1. **Veri İşiyle Uğraşma (Data Mugging)**

- Bu aşama, ham verinin düzenlenmesi ve anlamlı bir hale getirilmesi için gerekli olan süreçtir. Burada, verilerin içindeki eksikliklerin, tutarsızlıkların ve hataların belirlenmesi ve düzeltilmesi gibi işlemler yapılır. Bu, veriyi daha güvenilir hale getirmek ve daha sonraki analizler için uygun hale getirmek için önemlidir.

### 2. **Metin Temizleme (Text Cleansing)**

- Metin temizleme, metin verisindeki gereksiz ve istenmeyen bilgilerin temizlenmesi sürecidir. Bu aşamada, noktalama işaretleri, sayılar, özel karakterler gibi analiz için gerekli olmayan ögeler metinden çıkarılır. Bu işlem, veriyi daha saf ve işlemeye uygun hale getirir.

### 3. **Spesifik Ön İşleme (Specific Preprocessing)**

- Bu aşama, belirli bir analiz türü veya görev için gerekli olan özel ön işlemleri içerir. Örneğin, belirli kelime kalıplarını çıkarma veya belirli dil özelliklerini düzeltme gibi. Bu adım, metnin işlenecek olan belirli algoritmalar veya modeller için uygun hale getirilmesini sağlar.

### 4. **Tokenizasyon (Tokenization)**

- Tokenizasyon, metni daha küçük parçalara (genellikle kelimelere veya cümlelere) ayırma sürecidir. Her bir küçük parça "token" olarak adlandırılır. Bu adım, metni analiz edilebilir hale getirir çünkü bilgisayarlar genellikle metinle bu küçük parçalar üzerinden çalışır.

### 5. **Duraklama Kelimelerinin Çıkarılması (Stop Word Removal)**

- Duraklama kelimeleri, analiz açısından genellikle bilgi taşımayan ve sıkça kullanılan kelimelerdir (örneğin, "ve", "bu", "için"). Bu adımda, metinden bu tür kelimeler çıkarılır, böylece daha anlamlı kelimeler ve öbekler üzerinde yoğunlaşılabilir.

### 6. **Kök Bulma veya Lemmatizasyon (Stemming or Lemmatization)**

- Bu işlem, kelimelerin köklerine indirgenmesi veya kelimelerin temel formuna dönüştürülmesi anlamına gelir. Kök bulma, kelimenin sonundaki ekleri kaldırarak temel formunu bulmaya çalışır, lemmatizasyon ise kelimenin anlamlı kök formunu belirler. Bu, metni standart bir forma getirerek, benzer anlam taşıyan kelimelerin aynı şekilde analiz edilmesini sağlar.

Bu adımlar, ham metin verisini temizleyip düzenleyerek, onu analiz ve modelleme için uygun hale getirmek amacıyla yapılan işlemleri kapsar. Her bir adım, metin madenciliği ve doğal dil işleme süreçlerinde önemli rol oynar.

---

### 1. Doğal Dil İşleme (NLP) Nedir?

NLP, bilgisayarların insan dilini anlama, yorumlama ve üretme yeteneğini geliştirmeyi amaçlayan bir yapay zeka dalıdır. Bu alanda amaç, doğal dillerdeki (yazılı veya sözlü) metinleri analiz etmek ve anlamlandırmaktır. Örneğin, bir metni özetlemek, duygu analizi yapmak, metni kategorilere ayırmak gibi görevler NLP'nin alanına girer.

### 2. NLP'nin Temel Bileşenleri

NLP çeşitli aşamalardan ve bileşenlerden oluşur. İşte bazı temel kavramlar:

- **Tokenization (Kelimelere Ayırma):** Bir metni daha küçük parçalara (kelimeler, cümleler) ayırma işlemidir. Örneğin, "Merhaba dünya!" cümlesini ["Merhaba", "dünya", "!"] olarak parçalayabiliriz.
- **Stemming ve Lemmatization:** Kelimeleri köklerine indirgeyerek ortak anlamlarını bulmaya çalışır. Örneğin, "koşmak", "koşuyor", "koştu" kelimeleri "koş" kökünde birleşir.
- **Stop Words (Durdurma Kelimeleri):** "ve", "ama", "ise" gibi genellikle anlam taşımayan ve analizde çıkarılan kelimelerdir.
- **Bag of Words (Kelime Torbası):** Bir metindeki kelimeleri sayarak (frekanslarına göre) temsil eder. Kelime sırası göz ardı edilir.
- **TF-IDF (Term Frequency - Inverse Document Frequency):** Kelimenin, bir belgedeki önemini belirlemek için kullanılır. Sık kullanılan kelimelerden daha nadir kullanılan kelimelere ağırlık verir.

### 3. NLP Süreci

NLP süreçlerinde genellikle şu adımlar izlenir:

1. **Veri Toplama:** NLP projelerinde kullanılacak metin verilerini toplamak.
2. **Ön İşleme (Preprocessing):** Veriyi temizlemek, kelimelere ayırmak, durdurma kelimelerini çıkarmak, ve köklerine indirmek gibi işlemleri içerir.
3. **Özellik Çıkarımı (Feature Extraction):** Metinden anlamlı özellikler çıkarmak (kelime frekansı, TF-IDF gibi).
4. **Model Eğitimi:** Çıkarılan özellikler kullanılarak makine öğrenmesi modellerini eğitmek.
5. **Model Değerlendirme ve Test Etme:** Modelin performansını test etmek ve değerlendirmek.
6. **Sonuçları Yorumlama ve Kullanma:** Modelin sonuçlarını analiz etmek ve kullanmak.

### 4. Python ile NLP'ye Giriş

Şimdi Python'da NLP'yi nasıl uygulayabileceğimizi gösterelim. İlk olarak, en yaygın kullanılan kütüphanelerden biri olan **NLTK**'yi (Natural Language Toolkit) kullanacağız. Daha sonra **spaCy** ve **Transformers** gibi daha gelişmiş kütüphanelere de geçeceğiz.

### Adım 1: Gerekli Kütüphaneleri Yükleme

Jupyter Notebook'ta gerekli kütüphaneleri yükleyelim:

```python
!pip install nltk
import nltk
nltk.download('punkt')
```

### Adım 2: Metin İşleme

Bir metni kelimelere ayıralım (tokenization):

```python
from nltk.tokenize import word_tokenize

metin = "Merhaba dünya! NLP öğrenmeye başladım."
kelimeler = word_tokenize(metin)
print(kelimeler)
```

### Adım 3: Durdurma Kelimeleri Çıkarma

Türkçe durdurma kelimelerini çıkaralım:

```python
from nltk.corpus import stopwords
nltk.download('stopwords')

durma_kelimesi = set(stopwords.words('turkish'))
kelimeler_filtreli = [kelime for kelime in kelimeler if kelime.lower() not in durma_kelimesi]
print(kelimeler_filtreli)
```

### Adım 4: Stemming

Kelimeleri köklerine indirgeme (stemming) işlemi:

```python
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
kökler = [stemmer.stem(kelime) for kelime in kelimeler_filtreli]
print(kökler)
```

Bu temel işlemleri yaparak metni ön işleme tabi tuttuk. Buradan sonra daha ileri konulara (TF-IDF, Word Embeddings, BERT gibi modeller) geçebiliriz.

### 5. İleri Konular ve NLP Uygulamaları

Temel işlemleri öğrendikten sonra, şu konulara yönelebiliriz:

- **TF-IDF ve Kelime Frekansı ile Özellik Çıkarımı**
- **Makine Öğrenmesi Modelleri ile Metin Sınıflandırma**
- **Duygu Analizi**
- **Word Embeddings (Word2Vec, GloVe)**
- **Derin Öğrenme Tabanlı Modeller (LSTM, BERT)**