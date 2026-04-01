"""
font_kur.py — Türkçe PDF desteği için DejaVu fontlarını indirir.
"""
import urllib.request, os, shutil

def pip_ile_indir():
    try:
        import reportlab
        rl_path = os.path.dirname(reportlab.__file__)
        font_path = os.path.join(rl_path, "fonts")
        basari = 0
        for dosya in ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf"]:
            kaynak = os.path.join(font_path, dosya)
            if os.path.exists(kaynak):
                shutil.copy(kaynak, dosya)
                print(f"  ✅ {dosya} reportlab'den kopyalandı.")
                basari += 1
        return basari >= 2
    except Exception as e:
        print(f"  ℹ ReportLab fontları yok: {e}")
        return False

def mac_fontu_kopyala():
    aranacak = [
        ("/Library/Fonts/Arial Unicode.ttf",                      "Arial Unicode"),
        ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",  "Arial Unicode"),
        ("/System/Library/Fonts/Arial.ttf",                       "Arial"),
        ("/Library/Fonts/Arial.ttf",                              "Arial"),
    ]
    for yol, isim in aranacak:
        if os.path.exists(yol):
            for hedef in ["DejaVuSans.ttf","DejaVuSans-Bold.ttf","DejaVuSans-Oblique.ttf"]:
                if not os.path.exists(hedef):
                    shutil.copy(yol, hedef)
                    print(f"  ✅ {hedef} ← {isim} kullanıldı.")
            return True
    return False

def url_ile_indir():
    urls = {
        "DejaVuSans.ttf":         "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf":    "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique.ttf": "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans-Oblique.ttf",
    }
    basari = 0
    for dosya, url in urls.items():
        if os.path.exists(dosya):
            print(f"  ✅ {dosya} zaten var.")
            basari += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as r, open(dosya, "wb") as f:
                data = r.read()
            if len(data) > 10000:
                print(f"  ✅ {dosya} indirildi ({len(data)//1024} KB).")
                basari += 1
            else:
                os.remove(dosya)
        except Exception as e:
            if os.path.exists(dosya): os.remove(dosya)
            print(f"  ❌ {dosya}: {e}")
    return basari

print("🔤 Font kurulumu başlıyor...\n")

print("1️⃣  ReportLab paketinden aranıyor...")
if pip_ile_indir():
    print("\n✅ Hazır!\n"); exit()

print("\n2️⃣  Mac sistem fontları aranıyor...")
if mac_fontu_kopyala():
    print("\n✅ Hazır!\n"); exit()

print("\n3️⃣  İnternetten indiriliyor...")
n = url_ile_indir()
if n >= 2:
    print("\n✅ Hazır!\n"); exit()

print("\n❌ Otomatik kurulum başarısız.")
print("Manuel: https://dejavu-fonts.github.io/ adresinden")
print("DejaVuSans.ttf, DejaVuSans-Bold.ttf dosyalarını proje klasörüne koy.\n")
