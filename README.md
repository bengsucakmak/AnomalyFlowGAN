AnomalyFlowGAN

AnomalyFlowGAN, endüstriyel sensör verilerinde anormallik tespiti ve zaman serisi tahmini yapabilen, ileri düzey yapay zeka modellerini bir araya getiren güçlü ve kapsamlı bir projedir.
Bu sistem, gerçek ve sentetik verileri birleştirerek, endüstri uygulamalarında erken uyarı ve güvenilir tahminler sunar.

Proje Hakkında

Anormal durumların önceden tespiti, işletmelerin maliyetlerini azaltır ve arıza riskini minimize eder.
AnomalyFlowGAN, bu kritik ihtiyacı karşılamak için:

WGAN-GP ile sentetik ve zenginleştirilmiş veri üretimi

RealNVP Normalizing Flow ile yüksek doğrulukta anomali skorlaması

Transformer Forecast ile uzun dönem ve hassas zaman serisi tahmini

Critic Model ile GAN eğitiminde kaliteyi optimize eder

Projede yer alan modeller, farklı açılardan birbirini tamamlayarak güvenilir ve kullanışlı sonuçlar üretir.

Dosya Yapısı ve İçerik Açıklaması

Klasör / Dosya	Açıklama

data/	Ham ve ön işlenmiş sensör verileri. Büyük veri setleri burada saklanır.
grafikcs/	Eğitim ve test sırasında oluşturulan grafikler, analiz raporları.
npy/	Ara işleme sonucu kaydedilmiş numpy formatındaki veri dosyaları.
saved models/	Eğitilmiş yapay zeka modellerinin .pth formatındaki ağırlıkları.
src/	Model mimarileri, eğitim ve yardımcı Python scriptleri.
notebooks/	Projeyi interaktif biçimde çalıştırmak için hazırlanmış Jupyter notebook dosyaları.
README.md	Proje hakkında kapsamlı bilgi, kullanım talimatları ve rehber.
requirements.txt	Projenin çalışması için gerekli Python paketleri listesi.

Kurulum Adımları

Python 3.8+ sürümünün kurulu olduğundan emin olun.

Yeni bir sanal ortam oluşturup aktif edin:

python -m venv venv  
source venv/bin/activate    # Linux/Mac  
venv\Scripts\activate       # Windows 

Gerekli Python kütüphanelerini yükleyin:

pip install -r requirements.txt 

Dataset İndirme

Veri seti büyük olduğu için GitHub’a eklenmemiştir.

Aşağıdaki bağlantıdan indirip, açtıktan sonra data/ klasörüne yerleştirmeniz gerekmektedir:

https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip

Model Eğitimi ve Kullanımı

Eğitim süreci için: notebooks/colab_workflow.ipynb dosyasını kullanabilirsiniz.

Eğitilmiş modellerle hızlı tahmin ve anomali tespiti için: 

notebooks/AnomalyFlowGAN_Inference.ipynb ideal çözüm sunar.

Bu notebooklar, size hem modellerin nasıl eğitileceğini hem de eğitilmiş modellerle nasıl verimli çalışılacağını adım adım öğretir.

Projenin Faydaları

Veri sınırlaması olmadan zenginleştirilmiş sentetik ve gerçek veri kombinasyonu

Endüstriyel anomali tespitinde yüksek doğruluk ve güvenilirlik

Gelecek zaman serisi tahminlerinde ileri seviye transformer modeller

Modüler ve genişletilebilir yapı sayesinde ihtiyaçlarınıza göre kolayca adapte edilebilir

Hugging Face Model Hub entegrasyonu ile model paylaşımı ve kullanımı basitleştirilmiş

İletişim & Destek

Sorularınız, önerileriniz veya iş birliği tekliflerinizi bekliyorum.
Projemizi takip ettiğiniz ve destek verdiğiniz için teşekkür ederim! 

AnomalyFlowGAN — Geleceği öngör, anomaliyi önceden yakala! 
