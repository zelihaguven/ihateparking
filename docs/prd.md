# ihateparking — Ürün Gereksinimleri (PRD)

> Durum: **ONAYLANDI** (2026-10-20). Kabul kriterleri bağlayıcıdır.
> Mimari ve uzun vadeli yol haritası için: [real-system-architecture.md](real-system-architecture.md).

## 1. Problem

Sürücüye "şuraya park et" dendiğinde araç, daha önce hiç görmediği bir otoparkta boş yeri kendisi bulmalı ve oraya güvenle park etmelidir. Ezberlenmiş harita, önceden verilmiş yuva listesi ya da doluluk bilgisi kullanılmaz; araç yalnızca kendi sensörüne güvenir.

## 2. Hedef

Simülasyonda, rastgele üretilmiş ve hiç görülmemiş otoparklarda:

1. otoparkı kendi sensörüyle haritalamak,
2. gerçekten boş olan bir park yerini bulmak,
3. çarpmadan ve doğru yuvaya park etmek,
4. bulamıyorsa kontrollü biçimde durup bunu bildirmek.

## 3. Kapsam

**Bu sürümde var**
- 2-D simülasyon, rastgele otopark üreticisi (dik, açılı, paralel sıralar; değişken doluluk ve dönüş)
- Simüle LiDAR (gürültü ve ışın kaybı ile), doluluk haritası
- Haritadan park yeri tespiti, keşif, Hybrid A* planlama, takip kontrolü
- Görev sonu değerlendirmesi (yalnızca ölçüm için gerçek otopark verisi kullanılır)

**Bu sürümde yok** (sonraki sürümler)
- Kamera ve öğrenen algı modeli, CARLA, ROS2
- Hareketli engeller (yaya, araç), gece/yağmur, kalibrasyon hatası
- Gerçek araç entegrasyonu

## 4. Fonksiyonel gereksinimler

| No | Gereksinim |
|----|------------|
| F1 | Araç otoparka dair hiçbir ön bilgi almadan göreve başlar. |
| F2 | Harita yalnızca araç sensörünün ürettiği veriden oluşur. |
| F3 | Planlayıcı görülmemiş alanı park manevrasında engel sayar. |
| F4 | Her yeni taramada mevcut plan yeniden kontrol edilir; engel çıkarsa plan iptal edilir. |
| F5 | Boş yer bulunamazsa ya da süre dolarsa araç durur ve nedenini raporlar (`NO_SPACE_FOUND`, `TIMEOUT`, `GUARD_STOP`). |
| F6 | Her görev seed ile tekrar üretilebilir ve görsel olarak incelenebilir. |
| F7 | Dik (ileri/geri), açılı ve paralel park türlerinin hepsi desteklenir. |
| F8 | Araç sıkışırsa durup beklemez; çarpmadan oradan çıkmayı dener (kurtarma manevrası) ve aramaya devam eder. Kurtarma hamleleri de aynı çarpışma kontrolünden geçer. Yalnızca hiçbir güvenli hamle kalmadığında durur. |
| F9 | Araç kendi konumunu kusursuz bilmez; tekerlek/direksiyon odometrisi ve LiDAR'dan tahmin eder. Doğru park bu tahmin hatasına rağmen sağlanır. |

## 5. Kabul kriterleri

Ölçüm: geliştirmede hiç kullanılmamış en az 100 otoparklık bir seed kümesi (öneri: `5000:5100`).

| Metrik | Önerilen hedef | Son ölçüm (seed 3000–3039, 40 otopark) |
|--------|----------------|-----------------------------------------|
| Doğru boş yuvaya park oranı | ≥ %95 | %68 (27/40), son düzeltmelerden önce |
| Yanlış yere park | 0 | 0 |
| Çarpışma | 0 | 0 |
| Güvenli olmayan bitiş (hareket halinde kalma, kilitlenme) | 0 | 0 |
| Yuva örtüşmesi (doğru park sayılmak için) | ≥ %80 | eşik olarak uygulanıyor |
| Eksen hatası | ≤ 5° | eşik 15°; dağılım ölçülmedi |
| Son konum hatası (aracın merkezi ile yuvanın merkezi arasındaki mesafe) | ≤ 20 cm | ölçülmedi |
| Konum tahmini açıkken doğru park oranı | ≥ %95 | konum tahmini yok |
| Görev süresi (simüle, medyan / en kötü) | ≤ 90 sn / ≤ 240 sn | 66.7 sn medyan; en kötü ölçülmedi |
| Yuva tespiti kesinliği (tespit edilenlerin gerçekten boş olması) | %100 | %100 (37/37) |
| Otomatik test paketi | tamamı geçer | 76 geçti, 3 atlandı (CasADi yok) |

Bir sürüm ancak tüm kriterler aynı ölçüm kümesinde sağlanırsa "tamamlandı" sayılır.

## 6. Yol haritası

Her aşamada en üst öncelik doğru parktır: 0 yanlış park, 0 çarpışma, ≥ %95 başarı. Yeni bir özellik bu kriterleri bozuyorsa eklenmez.

1. **Başarı oranı:** Mevcut simülasyonda %68'den ≥ %95'e çıkmak.
2. **Konum tahmini:** Araca kusursuz konum vermeyi bırakmak. Gürültülü odometri ile LiDAR eşlemesinden konum tahmin edip yine ≥ %95 doğru park etmek.
3. **CARLA:** Aynı görevi 3-D, gerçekçi sensörlü simülatörde çalıştırmak. CARLA macOS'ta çalışmaz; NVIDIA GPU'lu bir Linux ya da Windows makinesi gerekir.

## 7. Bilinen açıklar

- Başarısızlıkların çoğu `NO_SPACE_FOUND`. Gerçek otoparkta boş yer var, ama araç onu ya görmüyor ya da kabul etmiyor. Detektör şimdilik yalnızca iki yanı dolu yuvaları kabul ediyor; tek komşulu yuvalara izin vermek kesinliği %55'e düşürdü.
- Kurtarma davranışı (F8), mimari belgedeki "sahne bilinmiyorsa yalnızca dur" ilkesinin yerine geçer. Bu bilinçli bir ürün kararıdır.
- Planlama süresi bazı durumlarda saniyeler sürüyor; mimari belgedeki 5–10 Hz yeniden planlama hedefi ölçülmedi.
