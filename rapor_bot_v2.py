"""
Ulaşım Rapor Botu v3
─────────────────────────────────────────────────────────
  ✅ PDF tasarımı: başlık bandı, KPI kartları, renk göstergesi
  ✅ Telegram mesajı: bölüm ayraçları, uyarılar, en iyi/kötü firma
  ✅ Haftalık özet raporu: Pazartesi otomatik PDF + Telegram
  ✅ Firma uyarı sistemi: kırmızı/sarı eşikler
  ✅ ML — Haftalık doluluk trendi tahmini (7 gün ilerisi)
  ✅ ML — Düşük performanslı firma için kapasite önerisi
  ✅ ML — ARIMA + Prophet + Ensemble model desteği
  ✅ ML — Anomali tespiti (Z-score + IQR + Isolation Forest)
─────────────────────────────────────────────────────────
"""

import imaplib
import email
import sqlite3
import pandas as pd
import requests
import os, io, json, logging, time, traceback
from datetime import datetime, timedelta
from contextlib import contextmanager

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import IsolationForest

# Opsiyonel bağımlılıklar — yoksa graceful fallback
_ARIMA_VAR = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    import warnings as _warnings
    _warnings.filterwarnings("ignore", module="statsmodels")
    _ARIMA_VAR = True
except ImportError:
    pass

_PROPHET_VAR = False
try:
    from prophet import Prophet
    _PROPHET_VAR = True
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Matplotlib için DejaVu fontunu ayarla (Türkçe/Rusça karakter desteği)
_MPL_FONT_SET = False
for _mpl_font in ["DejaVu Sans", "Liberation Sans", "FreeSans", "Arial"]:
    try:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = [_mpl_font, "DejaVu Sans", "Arial"]
        _MPL_FONT_SET = True
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── Türkçe font kaydı ──────────────────────────────────────────────────────

def _font_bul(isim: str, dosya_adi: str) -> str | None:
    """Font dosyasını proje klasöründe ve sistem yollarında arar."""
    # 1. Proje klasörü
    if os.path.exists(dosya_adi):
        return dosya_adi

    # 2. Sistem font yolları (Linux / macOS / Windows)
    sistem_yollar = [
        # Linux (Ubuntu, Debian, GitHub Actions)
        f"/usr/share/fonts/truetype/dejavu/{dosya_adi}",
        f"/usr/share/fonts/TTF/{dosya_adi}",
        f"/usr/share/fonts/{dosya_adi}",
        f"/usr/local/share/fonts/{dosya_adi}",
        # macOS
        f"/Library/Fonts/{dosya_adi}",
        f"/System/Library/Fonts/Supplemental/{dosya_adi}",
        os.path.expanduser(f"~/Library/Fonts/{dosya_adi}"),
        # Windows
        f"C:\\Windows\\Fonts\\{dosya_adi}",
    ]
    for yol in sistem_yollar:
        if os.path.exists(yol):
            return yol

    # 3. reportlab paketi içindeki fontlar
    try:
        import reportlab
        rl_font = os.path.join(os.path.dirname(reportlab.__file__), "fonts", dosya_adi)
        if os.path.exists(rl_font):
            return rl_font
    except Exception:
        pass

    return None


def _fontlari_kaydet():
    """
    DejaVu fontlarını proje klasöründe, sistem fontlarında veya reportlab içinde arar.
    Bulunamazsa Helvetica'ya geri döner.
    """
    font_harita = {
        "DejaVuSans":          "DejaVuSans.ttf",
        "DejaVuSans-Bold":     "DejaVuSans-Bold.ttf",
        "DejaVuSans-Oblique":  "DejaVuSans-Oblique.ttf",
    }

    bulunanlar = {}
    for isim, dosya in font_harita.items():
        yol = _font_bul(isim, dosya)
        if yol:
            bulunanlar[isim] = yol

    if len(bulunanlar) >= 2:  # En az normal + bold
        for isim, yol in bulunanlar.items():
            try:
                pdfmetrics.registerFont(TTFont(isim, yol))
                print(f"  ✅ Font kaydedildi: {isim} ← {yol}")
            except Exception as e:
                print(f"  ⚠ Font kayıt hatası ({isim}): {e}")

        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily(
            "DejaVuSans",
            normal="DejaVuSans",
            bold=bulunanlar.get("DejaVuSans-Bold", "DejaVuSans"),
            italic=bulunanlar.get("DejaVuSans-Oblique", "DejaVuSans"),
            boldItalic=bulunanlar.get("DejaVuSans-Bold", "DejaVuSans"),
        )
        return "DejaVuSans", bulunanlar.get("DejaVuSans-Bold", "DejaVuSans")
    else:
        print("⚠ DejaVu fontları bulunamadı. Türkçe/Rusça karakterler bozuk çıkabilir.")
        print("  Düzeltmek için:  python3 font_kur.py")
        print("  Veya:  sudo apt install fonts-dejavu-core  (Linux)")
        return "Helvetica", "Helvetica-Bold"


_FONT_NORMAL, _FONT_BOLD = _fontlari_kaydet()
# ────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════
#  AYARLAR
# ═══════════════════════════════════════════════
GMAIL_USER     = os.environ.get("GMAIL_USER",     "tantuncer6@gmail.com")
GMAIL_PASS     = os.environ.get("GMAIL_PASS",     "anprwtcjcriqcbdz")
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "7978688244:AAGLtC_4_tqQwMlMubfaNDeGuDnJVsN1Yn4")
TELEGRAM_CHAT  = os.environ.get("TELEGRAM_CHAT",  "886597229")
GECMIS_DOSYA   = "gecmis_veriler.json"

# ── Veri Kaynağı Ayarları ──────────────────────
# Gönderenler: isim veya e-posta parçası — bunlardan biri From alanında geçerse ek indirilir
GONDERICILER = [
    "Ali", "Ruslan", "Ismanzhanov", "Ismatillaev",
    # Yeni gönderici eklemek için buraya eklemeniz yeterli:
    # "Mehmet", "info@firma.com",
]

# E-posta tarama: kaç gün geriye baksın
MAIL_LOOKBACK_GUN = 2

# Desteklenen dosya uzantıları
DESTEKLENEN_UZANTILAR = [".xlsx", ".xls", ".csv"]

# ── Şehir Servisi Excel Ayarları ───────────────
# Dosya adında bu kelimelerden biri geçerse şehir servisi olarak tanınır
SEHIR_DOSYA_PATTERN = ["SEHIR SERVIS", "ŞEHİR SERVİS", "CITY SERVICE"]

# Sayfa adı denemeleri (sırayla denenir, ilk bulunan kullanılır)
SEHIR_SAYFA_ADLARI = ["liste", "Liste", "LISTE", "Sayfa1", "Sheet1"]

# Sütun alias haritası: { standart_isim: [olası_isimler] }
SEHIR_SUTUN_ALIAS = {
    "TARIH":             ["TARIH", "TARİH", "Tarih", "tarih", "DATE", "Date"],
    "OTOBUS FIRMASI":    ["OTOBUS FIRMASI", "OTOBÜS FİRMASI", "Firma", "FIRMA",
                          "firma_adi", "Firma Adı", "BUS COMPANY", "Company"],
    "OTOBUS PLAKASI":    ["OTOBUS PLAKASI", "OTOBÜS PLAKASI", "Plaka", "PLAKA",
                          "plaka", "PLATE", "Plate"],
    "OTOBUS KAPASITESI": ["OTOBUS KAPASITESI", "OTOBÜS KAPASİTESİ", "Kapasite",
                          "KAPASİTE", "CAPACITY", "Capacity", "koltuk"],
    "TASIMA KAPASITESI": ["TASIMA KAPASITESI", "TAŞIMA KAPASİTESİ", "Yolcu",
                          "YOLCU", "yolcu_sayisi", "PASSENGER", "Passenger"],
}

# ── Şantiye Servisi Excel Ayarları ─────────────
# Dosya adında bu kelimelerden biri geçerse şantiye servisi olarak tanınır
SANTIYE_DOSYA_PATTERN = ["OTOBUS LISTESI", "OTOBÜS LİSTESİ", "BUS LIST"]

# Sayfa adı şablonu (tarih yerine {tarih} yazılır, sırayla denenir)
SANTIYE_SAYFA_SABLONLARI = [
    "{tarih}.SABAH SERVIS.",
    "{tarih}.SABAH SERVİS.",
    "{tarih} SABAH",
    "{tarih}",
]

# Şantiye sütun alias haritası
SANTIYE_SUTUN_ALIAS = {
    "организация":      ["организация", "Организация", "ОРГАНИЗАЦИЯ",
                          "Firma", "FIRMA", "Organization", "firma_adi"],
    "Гос-Номер":        ["Гос-Номер", "ГОС-НОМЕР", "Plaka", "PLAKA",
                          "Plate", "plaka"],
    "Кол-во мест":      ["Кол-во мест", "КОЛ-ВО МЕСТ", "Kapasite",
                          "KAPASİTE", "Capacity", "koltuk"],
    "кол-во поссажиров": ["кол-во поссажиров", "КОЛ-ВО ПОССАЖИРОВ",
                           "кол-во пассажиров", "Yolcu", "YOLCU",
                           "Passenger", "yolcu_sayisi"],
}

# ── Tarih Format Denemeleri ────────────────────
# verileri_isle tarih eşleştirmede bu formatlar sırayla denenir
TARIH_FORMATLARI = [
    "%Y-%m-%d",       # 2026-03-20
    "%d.%m.%Y",       # 20.03.2026
    "%d/%m/%Y",       # 20/03/2026
    "%d-%m-%Y",       # 20-03-2026
    "%Y/%m/%d",       # 2026/03/20
    "%m/%d/%Y",       # 03/20/2026
]

# Uyarı eşikleri
ESIK_KIRMIZI_DOLULUK = 60    # %60 altı → kırmızı uyarı
ESIK_SARI_DOLULUK    = 80    # %60–80 arası → sarı uyarı
ESIK_MIN_ARAC        = 3     # Bu sayının altındaki araç → uyarı

# Makine öğrenmesi ayarları
ML_MIN_GUN           = 7     # Tahmin için gereken minimum gün sayısı
ML_TAHMIN_GUN        = 7     # Kaç gün ilerisi tahmin edilsin
ML_DUSUK_ESIK        = 75   # Bu doluluk altındaki firmalar için öneri üret (%)

# Model seçimi: "auto", "ridge", "arima", "prophet", "ensemble"
# auto → veri miktarına göre en iyi modeli seçer
# ensemble → tüm mevcut modellerin ağırlıklı ortalaması
ML_MODEL_SECIMI      = "auto"

# ARIMA ayarları
ML_ARIMA_ORDER       = (1, 1, 1)   # (p, d, q) — auto ise otomatik seçilir
ML_ARIMA_AUTO_ORDER  = True        # True ise en iyi order'ı otomatik arar

# Prophet ayarları
ML_PROPHET_SEASONALITY = "auto"    # "auto", "weekly", "none"

# Anomali tespiti ayarları
ANOMALI_AKTIF        = True        # Anomali tespiti açık/kapalı
ANOMALI_MIN_GUN      = 5          # Anomali tespiti için gereken minimum gün
ANOMALI_ZSCORE_ESIK  = 2.0        # Z-score eşiği (2.0 = %95 güven)
ANOMALI_IQR_CARPAN   = 1.5        # IQR çarpanı (1.5 = standart, 3.0 = aşırı)
ANOMALI_IF_KONTAMINASYON = 0.1    # Isolation Forest kontaminasyon oranı

# Renk paleti
RENK_LACIVERT  = "#1A3A5C"
RENK_ACIK_MAVİ = "#4A90D9"
RENK_TURUNCU   = "#E8834A"
RENK_YESIL     = "#27AE60"
RENK_SARI      = "#F39C12"
RENK_KIRMIZI   = "#E74C3C"
RENK_ARK       = "#F7F9FC"
RENK_GRI       = "#95A5A6"

# ── Veritabanı Ayarları ────────────────────────
DB_DOSYA       = "rapor_bot.db"      # SQLite dosyası
DB_JSON_YEDEK  = True                # JSON yedek dosyalarını da güncelle

# ── Bildirim Ayarları ──────────────────────────
BILDIRIM_AKTIF         = True
BILDIRIM_COOLDOWN_SAAT = 6     # Aynı tip uyarı tekrar gönderilmeden önce bekleme
BILDIRIM_ESIKLER = {
    "doluluk_kritik":   50,    # Bu altına düşerse ANLIK uyarı
    "doluluk_dusuk":    65,    # Bu altına düşerse uyarı
    "arac_min":         3,     # Bu altına düşerse uyarı
    "yolcu_ani_dusus":  30,    # Önceki güne göre % düşüş — anlık uyarı
}

# ── Retry / Hata Yönetimi ─────────────────────
RETRY_MAX        = 3          # Maksimum tekrar deneme
RETRY_BEKLEME    = 30         # İlk bekleme (saniye) — exponential backoff
RETRY_CARPAN     = 2          # Bekleme çarpanı (30, 60, 120...)
# ═══════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rapor_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


# ═══════════════════════════════════════════════
#  SQLite VERİTABANI
# ═══════════════════════════════════════════════

@contextmanager
def _db_baglanti():
    """Thread-safe SQLite bağlantısı."""
    conn = sqlite3.connect(DB_DOSYA, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_tablolari_olustur():
    """Veritabanı tablolarını oluşturur (yoksa)."""
    with _db_baglanti() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS gunluk_gecmis (
            tarih      TEXT PRIMARY KEY,
            arac       INTEGER DEFAULT 0,
            kapasite   INTEGER DEFAULT 0,
            yolcu      INTEGER DEFAULT 0,
            doluluk    REAL    DEFAULT 0,
            created_at TEXT    DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS firma_gecmis (
            tarih      TEXT NOT NULL,
            firma      TEXT NOT NULL,
            kategori   TEXT,
            arac       INTEGER DEFAULT 0,
            sefer      INTEGER DEFAULT 0,
            kapasite   INTEGER DEFAULT 0,
            yolcu      INTEGER DEFAULT 0,
            doluluk    REAL    DEFAULT 0,
            created_at TEXT    DEFAULT (datetime('now','localtime')),
            PRIMARY KEY (tarih, firma)
        );

        CREATE TABLE IF NOT EXISTS bildirimler (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih      TEXT NOT NULL,
            tip        TEXT NOT NULL,
            seviye     TEXT NOT NULL DEFAULT 'bilgi',
            mesaj      TEXT NOT NULL,
            gonderildi INTEGER DEFAULT 0,
            created_at TEXT    DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS rapor_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            tarih      TEXT NOT NULL,
            durum      TEXT NOT NULL DEFAULT 'basarili',
            sure_sn    REAL DEFAULT 0,
            hata       TEXT,
            detay      TEXT,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE INDEX IF NOT EXISTS idx_firma_tarih ON firma_gecmis(tarih);
        CREATE INDEX IF NOT EXISTS idx_firma_firma ON firma_gecmis(firma);
        CREATE INDEX IF NOT EXISTS idx_bildirim_tarih ON bildirimler(tarih);
        CREATE INDEX IF NOT EXISTS idx_rapor_tarih ON rapor_log(tarih);
        """)
    log.info("📦 Veritabanı tabloları hazır.")


def db_json_migration():
    """Mevcut JSON dosyalarını SQLite'a aktarır (tek seferlik)."""
    with _db_baglanti() as conn:
        # Genel geçmiş
        if os.path.exists(GECMIS_DOSYA):
            try:
                with open(GECMIS_DOSYA, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for tarih, v in raw.items():
                    conn.execute("""
                        INSERT OR IGNORE INTO gunluk_gecmis (tarih, arac, kapasite, yolcu, doluluk)
                        VALUES (?, ?, ?, ?, ?)
                    """, (tarih, v.get("arac", 0), v.get("kapasite", 0),
                          v.get("yolcu", 0), v.get("doluluk", 0)))
                log.info(f"📦 JSON migration: {len(raw)} günlük kayıt aktarıldı.")
            except Exception as e:
                log.warning(f"JSON migration hatası (genel): {e}")

        # Firma geçmişi
        if os.path.exists(FIRMA_GECMIS_DOSYA):
            try:
                with open(FIRMA_GECMIS_DOSYA, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                sayac = 0
                for tarih, firmalar in raw.items():
                    for firma, v in firmalar.items():
                        conn.execute("""
                            INSERT OR IGNORE INTO firma_gecmis
                            (tarih, firma, kategori, arac, sefer, kapasite, yolcu, doluluk)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (tarih, firma, v.get("kategori", ""),
                              v.get("arac", 0), v.get("sefer", 0),
                              v.get("kapasite", 0), v.get("yolcu", 0),
                              v.get("doluluk", 0)))
                        sayac += 1
                log.info(f"📦 JSON migration: {sayac} firma kaydı aktarıldı.")
            except Exception as e:
                log.warning(f"JSON migration hatası (firma): {e}")


# DB başlat (tablolar hemen oluşturulur, migration ana fonksiyonda çalışır)
db_tablolari_olustur()
_DB_MIGRATION_YAPILDI = False


# ═══════════════════════════════════════════════
#  BİLDİRİM SİSTEMİ
# ═══════════════════════════════════════════════

def _bildirim_cooldown_kontrol(tip: str) -> bool:
    """Aynı tip bildirim son X saat içinde gönderilmiş mi?"""
    with _db_baglanti() as conn:
        row = conn.execute("""
            SELECT created_at FROM bildirimler
            WHERE tip = ? AND gonderildi = 1
            ORDER BY created_at DESC LIMIT 1
        """, (tip,)).fetchone()
        if not row:
            return True  # Daha önce gönderilmemiş
        son = datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S")
        fark = (datetime.now() - son).total_seconds() / 3600
        return fark >= BILDIRIM_COOLDOWN_SAAT


def bildirim_kaydet(tarih: str, tip: str, seviye: str, mesaj: str,
                     gonder: bool = True) -> bool:
    """Bildirimi kaydeder ve Telegram'a gönderir."""
    with _db_baglanti() as conn:
        conn.execute("""
            INSERT INTO bildirimler (tarih, tip, seviye, mesaj, gonderildi)
            VALUES (?, ?, ?, ?, ?)
        """, (tarih, tip, seviye, mesaj, 1 if gonder else 0))

    if gonder and BILDIRIM_AKTIF:
        emo = {"kritik": "🚨", "uyari": "⚠️", "bilgi": "ℹ️"}.get(seviye, "📢")
        tg = f"{emo} *ANLIK BİLDİRİM*\n"
        tg += f"_{seviye.upper()} — {tip}_\n"
        tg += f"━━━━━━━━━━━━━━━━━━━━━\n"
        tg += mesaj
        return telegram_mesaj_gonder(tg)
    return False


def esik_kontrolu(tarih_str: str, gecmis: dict,
                   sehir_df: pd.DataFrame, santiye_df: pd.DataFrame):
    """
    Eşik aşımlarını kontrol eder ve anlık bildirim gönderir.
    Cooldown mekanizmasıyla spam engellenir.
    """
    if not BILDIRIM_AKTIF:
        return

    esikler = BILDIRIM_ESIKLER

    # Tüm veriyi birleştir
    tum_dfs = [df for df in [sehir_df, santiye_df] if not df.empty]
    if not tum_dfs:
        return
    tum = pd.concat(tum_dfs)
    g_kap = int(tum["Toplam_Kapasite"].sum())
    g_yolcu = int(tum["Toplam_Yolcu"].sum())
    g_dol = g_yolcu / g_kap * 100 if g_kap > 0 else 0

    # 1. Genel doluluk kritik
    if g_dol < esikler["doluluk_kritik"]:
        tip = "doluluk_kritik"
        if _bildirim_cooldown_kontrol(tip):
            bildirim_kaydet(tarih_str, tip, "kritik",
                f"Genel doluluk *%{g_dol:.1f}* — kritik eşik (%{esikler['doluluk_kritik']}) altında!\n"
                f"Yolcu: {g_yolcu:,} / Kapasite: {g_kap:,}")
    elif g_dol < esikler["doluluk_dusuk"]:
        tip = "doluluk_dusuk"
        if _bildirim_cooldown_kontrol(tip):
            bildirim_kaydet(tarih_str, tip, "uyari",
                f"Genel doluluk *%{g_dol:.1f}* — düşük eşik (%{esikler['doluluk_dusuk']}) altında.\n"
                f"Yolcu: {g_yolcu:,} / Kapasite: {g_kap:,}")

    # 2. Firma bazlı kontrol
    for df_name, df, col in [("Şehir", sehir_df, "OTOBUS FIRMASI"),
                              ("Şantiye", santiye_df, "организация")]:
        if df.empty:
            continue
        for _, row in df.iterrows():
            firma = str(row[col])[:25]
            arac = int(row["Farkli_Otobus"])
            kap = int(row["Toplam_Kapasite"])
            yolcu = int(row["Toplam_Yolcu"])
            dol = yolcu / kap * 100 if kap > 0 else 0

            if arac < esikler["arac_min"]:
                tip = f"arac_min_{firma}"
                if _bildirim_cooldown_kontrol(tip):
                    bildirim_kaydet(tarih_str, tip, "uyari",
                        f"*{firma}* [{df_name}]: Sadece *{arac} araç* aktif "
                        f"(minimum: {esikler['arac_min']})")

            if dol < esikler["doluluk_kritik"]:
                tip = f"firma_kritik_{firma}"
                if _bildirim_cooldown_kontrol(tip):
                    bildirim_kaydet(tarih_str, tip, "kritik",
                        f"*{firma}* [{df_name}]: Doluluk *%{dol:.1f}* — kritik seviye!")

    # 3. Ani düşüş kontrolü (önceki güne göre)
    if len(gecmis) >= 2:
        tarihler = sorted(gecmis.keys())
        onceki = gecmis[tarihler[-2]]
        if onceki["yolcu"] > 0:
            dusus = (onceki["yolcu"] - g_yolcu) / onceki["yolcu"] * 100
            if dusus > esikler["yolcu_ani_dusus"]:
                tip = "yolcu_ani_dusus"
                if _bildirim_cooldown_kontrol(tip):
                    bildirim_kaydet(tarih_str, tip, "kritik",
                        f"Yolcu sayısında ani düşüş: *%{dusus:.0f}*\n"
                        f"Dün: {onceki['yolcu']:,} → Bugün: {g_yolcu:,}\n"
                        f"Fark: {onceki['yolcu'] - g_yolcu:,} yolcu")

    log.info("🔔 Eşik kontrolü tamamlandı.")


def rapor_log_kaydet(tarih: str, durum: str, sure: float, hata: str = "", detay: str = ""):
    """Rapor çalıştırma logunu veritabanına kaydeder."""
    try:
        with _db_baglanti() as conn:
            conn.execute("""
                INSERT INTO rapor_log (tarih, durum, sure_sn, hata, detay)
                VALUES (?, ?, ?, ?, ?)
            """, (tarih, durum, sure, hata, detay))
    except Exception as e:
        log.error(f"Rapor log kayıt hatası: {e}")


# ── Retry Decorator ───────────────────────────

def retry_ile_calistir(fonksiyon, *args, **kwargs):
    """Exponential backoff ile retry mekanizması."""
    son_hata = None
    for deneme in range(1, RETRY_MAX + 1):
        try:
            return fonksiyon(*args, **kwargs)
        except Exception as e:
            son_hata = e
            if deneme < RETRY_MAX:
                bekleme = RETRY_BEKLEME * (RETRY_CARPAN ** (deneme - 1))
                log.warning(f"⚠ Deneme {deneme}/{RETRY_MAX} başarısız: {e}")
                log.info(f"  ⏳ {bekleme}s sonra tekrar denenecek...")
                time.sleep(bekleme)
            else:
                log.error(f"❌ {RETRY_MAX} deneme başarısız: {e}")
                log.error(traceback.format_exc())
    raise son_hata


# ═══════════════════════════════════════════════
#  MAKİNE ÖĞRENMESİ MODÜLÜ
# ═══════════════════════════════════════════════

def _gecmis_df(gecmis: dict) -> pd.DataFrame:
    """
    JSON geçmiş verisini modele uygun DataFrame'e çevirir.
    Özellikler: gün indeksi, haftanın günü (0=Pzt), sin/cos çevrimi
    """
    rows = []
    for i, (tarih, v) in enumerate(sorted(gecmis.items())):
        try:
            dt = datetime.strptime(tarih, "%Y-%m-%d")
        except ValueError:
            continue
        haftanin_gunu = dt.weekday()           # 0=Pazartesi … 6=Pazar
        rows.append({
            "idx":          i,
            "hgun":         haftanin_gunu,
            "hgun_sin":     np.sin(2 * np.pi * haftanin_gunu / 7),
            "hgun_cos":     np.cos(2 * np.pi * haftanin_gunu / 7),
            "doluluk":      v["doluluk"],
            "arac":         v["arac"],
            "yolcu":        v.get("yolcu", 0),
            "kapasite":     v.get("kapasite", 1),
            "tarih":        tarih,
            "dt":           dt,
        })
    return pd.DataFrame(rows)


def ml_haftalik_tahmin(gecmis: dict) -> dict | None:
    """
    Polinom + Ridge regresyon ile önümüzdeki 7 günün doluluk
    oranını tahmin eder.

    Döndürür:
        {
          "tahminler": [(tarih_str, doluluk_pct), ...],  # 7 eleman
          "model_hata": float,   # MAE (eğitim seti üzerinde)
          "guven": str,          # "Yüksek" / "Orta" / "Düşük"
          "veri_gun": int,       # kaç günlük veri kullanıldı
        }
        ya da None (yetersiz veri)
    """
    if len(gecmis) < ML_MIN_GUN:
        return None

    df = _gecmis_df(gecmis)
    if df.empty or len(df) < ML_MIN_GUN:
        return None

    # Özellik matrisi: idx + haftanın günü sin/cos
    X = df[["idx", "hgun_sin", "hgun_cos"]].values
    y = df["doluluk"].values

    # Polinom derece 2, Ridge düzenleme
    model = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.fit(X, y)

    # MAE — basit bir güven göstergesi
    y_pred_train = model.predict(X)
    mae = mean_absolute_error(y, y_pred_train)
    guven = "Yüksek" if mae < 5 else "Orta" if mae < 10 else "Düşük"

    # Gelecek 7 gün
    son_idx = int(df["idx"].max())
    son_dt  = df["dt"].max()
    tahminler = []
    for gun in range(1, ML_TAHMIN_GUN + 1):
        gelecek_dt  = son_dt + timedelta(days=gun)
        gelecek_idx = son_idx + gun
        hgun        = gelecek_dt.weekday()
        X_pred = np.array([[
            gelecek_idx,
            np.sin(2 * np.pi * hgun / 7),
            np.cos(2 * np.pi * hgun / 7),
        ]])
        tahmin = float(np.clip(model.predict(X_pred)[0], 0, 100))
        tahminler.append((gelecek_dt.strftime("%Y-%m-%d"), round(tahmin, 1)))

    return {
        "tahminler": tahminler,
        "model_hata": round(mae, 2),
        "guven": guven,
        "veri_gun": len(df),
    }


def ml_firma_kapasite_onerisi(sehir_df: pd.DataFrame,
                               santiye_df: pd.DataFrame,
                               gecmis: dict) -> list[dict]:
    """
    Mevcut günün düşük doluluklu firmaları için kapasite önerisi üretir.

    Mantık:
      - Son 7 günlük genel doluluk ortalamasına bakarak
        "hedef yolcu" hesaplar
      - Hedef yolcuya ulaşmak için kaç araç gerektiğini önerir
      - Fazla kapasiteyi azalt veya eksik aracı artır önerisi verir

    Döndürür: [{"firma": str, "mevcut_arac": int, "mevcut_dol": float,
                "hedef_dol": float, "onerim": str, "delta_arac": int}, ...]
    """
    oneriler = []

    # Son 7 günün ortalama doluluk oranı → hedef eşik
    if gecmis:
        ort_doluluk = sum(v["doluluk"] for v in gecmis.values()) / len(gecmis)
        hedef_dol   = max(ort_doluluk, ML_DUSUK_ESIK)
    else:
        hedef_dol = ML_DUSUK_ESIK

    def _tara(df: pd.DataFrame, firma_col: str, kategori: str):
        if df.empty:
            return
        for _, row in df.iterrows():
            firma    = str(row[firma_col])
            kap      = int(row["Toplam_Kapasite"])
            yolcu    = int(row["Toplam_Yolcu"])
            arac     = int(row["Farkli_Otobus"])
            sefer    = int(row["Toplam_Sefer"])
            dol      = yolcu / kap * 100 if kap > 0 else 0

            if dol >= ML_DUSUK_ESIK:
                continue   # Zaten iyi performansta, öneri gerekmez

            # Araç başına ortalama kapasite
            kap_per_arac = kap / arac if arac > 0 else 50
            # Hedef dolulukta gereken yolcu
            hedef_yolcu  = kap * hedef_dol / 100
            # Mevcut yolcu / hedef yolcu oranından araç önerisi
            if yolcu > 0:
                hedef_kap   = yolcu / (hedef_dol / 100)
                onerilecek_arac = max(1, round(hedef_kap / kap_per_arac))
            else:
                onerilecek_arac = arac   # yolcu yoksa dokunma

            delta = onerilecek_arac - arac

            if delta < 0:
                onerim = (f"{abs(delta)} araç azaltılabilir "
                          f"(şu an {arac} araç var, {onerilecek_arac} yeterli). "
                          f"Günlük yaklaşık %{abs(delta)/arac*100:.0f} maliyet tasarrufu.")
            elif delta > 0:
                onerim = (f"{delta} araç daha gerekebilir "
                          f"(şu an {arac} araç var, hedef için {onerilecek_arac} önerilir). "
                          f"Ek sefer düzenlenebilir.")
            else:
                onerim = "Araç sayısı uygun, rota veya saat düzenlemesi denenebilir."

            oneriler.append({
                "firma":         firma[:30],
                "kategori":      kategori,
                "mevcut_arac":   arac,
                "mevcut_dol":    round(dol, 1),
                "hedef_dol":     round(hedef_dol, 1),
                "onerim":        onerim,
                "delta_arac":    delta,
                "sefer":         sefer,
            })

    _tara(sehir_df,   "OTOBUS FIRMASI", "Şehir")
    _tara(santiye_df, "организация",    "Şantiye")
    return oneriler


# ═══════════════════════════════════════════════
#  GELİŞMİŞ ML MODELLERİ
# ═══════════════════════════════════════════════

def _arima_en_iyi_order(seri: np.ndarray) -> tuple:
    """AIC bazlı basit ARIMA order arama (p,d,q kombinasyonları)."""
    en_iyi_aic = float("inf")
    en_iyi = (1, 1, 1)
    for p in range(0, 3):
        for d in range(0, 2):
            for q in range(0, 3):
                try:
                    model = ARIMA(seri, order=(p, d, q))
                    sonuc = model.fit()
                    if sonuc.aic < en_iyi_aic:
                        en_iyi_aic = sonuc.aic
                        en_iyi = (p, d, q)
                except Exception:
                    continue
    return en_iyi


def ml_arima_tahmin(gecmis: dict) -> dict | None:
    """
    ARIMA modeli ile doluluk tahmini.
    statsmodels yoksa None döner.
    """
    if not _ARIMA_VAR:
        log.debug("ARIMA: statsmodels yüklü değil, atlanıyor.")
        return None

    if len(gecmis) < ML_MIN_GUN:
        return None

    df = _gecmis_df(gecmis)
    if df.empty or len(df) < ML_MIN_GUN:
        return None

    seri = df["doluluk"].values

    try:
        if ML_ARIMA_AUTO_ORDER:
            order = _arima_en_iyi_order(seri)
            log.info(f"  ARIMA auto-order: {order}")
        else:
            order = ML_ARIMA_ORDER

        model = ARIMA(seri, order=order)
        sonuc = model.fit()

        tahmin_raw = sonuc.forecast(steps=ML_TAHMIN_GUN)
        mae = mean_absolute_error(seri, sonuc.fittedvalues[-len(seri):])
        guven = "Yüksek" if mae < 5 else "Orta" if mae < 10 else "Düşük"

        son_dt = df["dt"].max()
        tahminler = []
        for gun in range(1, ML_TAHMIN_GUN + 1):
            gelecek_dt = son_dt + timedelta(days=gun)
            dol = float(np.clip(tahmin_raw[gun - 1], 0, 100))
            tahminler.append((gelecek_dt.strftime("%Y-%m-%d"), round(dol, 1)))

        return {
            "tahminler":  tahminler,
            "model_hata": round(mae, 2),
            "guven":      guven,
            "veri_gun":   len(df),
            "model_adi":  "ARIMA",
            "model_detay": f"order={order}, AIC={sonuc.aic:.1f}",
        }
    except Exception as e:
        log.warning(f"  ARIMA tahmin hatası: {e}")
        return None


def ml_prophet_tahmin(gecmis: dict) -> dict | None:
    """
    Prophet modeli ile doluluk tahmini.
    prophet yoksa None döner.
    """
    if not _PROPHET_VAR:
        log.debug("Prophet: kütüphane yüklü değil, atlanıyor.")
        return None

    if len(gecmis) < ML_MIN_GUN:
        return None

    df = _gecmis_df(gecmis)
    if df.empty or len(df) < ML_MIN_GUN:
        return None

    try:
        # Prophet ds + y formatı
        pdf = pd.DataFrame({"ds": df["dt"], "y": df["doluluk"]})

        m = Prophet(
            daily_seasonality=False,
            yearly_seasonality=False,
            weekly_seasonality=(ML_PROPHET_SEASONALITY != "none"),
            changepoint_prior_scale=0.05,
        )
        m.fit(pdf)

        future = m.make_future_dataframe(periods=ML_TAHMIN_GUN)
        forecast = m.predict(future)

        # MAE hesapla (eğitim seti)
        egitim_tahmin = forecast.head(len(df))["yhat"].values
        mae = mean_absolute_error(df["doluluk"].values, egitim_tahmin)
        guven = "Yüksek" if mae < 5 else "Orta" if mae < 10 else "Düşük"

        tahminler = []
        gelecek = forecast.tail(ML_TAHMIN_GUN)
        for _, row in gelecek.iterrows():
            dol = float(np.clip(row["yhat"], 0, 100))
            tahminler.append((row["ds"].strftime("%Y-%m-%d"), round(dol, 1)))

        return {
            "tahminler":  tahminler,
            "model_hata": round(mae, 2),
            "guven":      guven,
            "veri_gun":   len(df),
            "model_adi":  "Prophet",
            "model_detay": f"weekly={ML_PROPHET_SEASONALITY}",
        }
    except Exception as e:
        log.warning(f"  Prophet tahmin hatası: {e}")
        return None


def ml_ensemble_tahmin(gecmis: dict) -> dict | None:
    """
    Mevcut tüm modelleri çalıştırır, ağırlıklı ortalama ile birleştirir.
    Her modelin ağırlığı 1/MAE ile orantılıdır (daha düşük hata → daha yüksek ağırlık).

    ML_MODEL_SECIMI'ne göre davranış:
      "auto"     → veri miktarına göre en iyi tekli modeli seçer
      "ensemble" → tüm modellerin ağırlıklı ortalaması
      "ridge"    → sadece Ridge
      "arima"    → sadece ARIMA
      "prophet"  → sadece Prophet
    """
    if len(gecmis) < ML_MIN_GUN:
        return None

    # Tek model seçimi
    if ML_MODEL_SECIMI == "ridge":
        sonuc = ml_haftalik_tahmin(gecmis)
        if sonuc:
            sonuc["model_adi"] = "Ridge"
            sonuc["model_detay"] = "Poly(2) + Ridge(alpha=1.0)"
        return sonuc
    elif ML_MODEL_SECIMI == "arima":
        return ml_arima_tahmin(gecmis)
    elif ML_MODEL_SECIMI == "prophet":
        return ml_prophet_tahmin(gecmis)

    # Tüm modelleri çalıştır
    ridge_sonuc   = ml_haftalik_tahmin(gecmis)
    if ridge_sonuc:
        ridge_sonuc["model_adi"] = "Ridge"
        ridge_sonuc["model_detay"] = "Poly(2) + Ridge(alpha=1.0)"
    arima_sonuc   = ml_arima_tahmin(gecmis)
    prophet_sonuc = ml_prophet_tahmin(gecmis)

    tum_sonuclar = [s for s in [ridge_sonuc, arima_sonuc, prophet_sonuc] if s is not None]

    if not tum_sonuclar:
        return None

    if len(tum_sonuclar) == 1:
        return tum_sonuclar[0]

    # "auto" → en düşük MAE'li modeli seç
    if ML_MODEL_SECIMI == "auto":
        en_iyi = min(tum_sonuclar, key=lambda s: s["model_hata"])
        diger_isimler = [s["model_adi"] for s in tum_sonuclar if s != en_iyi]
        en_iyi["model_detay"] += f" (otomatik seçildi, diğer: {', '.join(diger_isimler)})"
        log.info(f"  ML auto-seçim: {en_iyi['model_adi']} "
                 f"(MAE={en_iyi['model_hata']:.2f})")
        return en_iyi

    # "ensemble" → ağırlıklı ortalama
    agirliklar = []
    for s in tum_sonuclar:
        w = 1.0 / max(s["model_hata"], 0.1)
        agirliklar.append(w)
    toplam_w = sum(agirliklar)
    agirliklar = [w / toplam_w for w in agirliklar]

    # Tahmin birleştirme
    n_gun = ML_TAHMIN_GUN
    ensemble_tahminler = []
    for gun_idx in range(n_gun):
        tarih = tum_sonuclar[0]["tahminler"][gun_idx][0]
        dol_toplam = 0.0
        for s, w in zip(tum_sonuclar, agirliklar):
            dol_toplam += s["tahminler"][gun_idx][1] * w
        ensemble_tahminler.append((tarih, round(float(np.clip(dol_toplam, 0, 100)), 1)))

    # Ensemble MAE → ağırlıklı ortalama MAE
    ens_mae = sum(s["model_hata"] * w for s, w in zip(tum_sonuclar, agirliklar))
    guven = "Yüksek" if ens_mae < 5 else "Orta" if ens_mae < 10 else "Düşük"

    model_isimleri = [f"{s['model_adi']}(w={w:.0%})" for s, w in zip(tum_sonuclar, agirliklar)]

    return {
        "tahminler":  ensemble_tahminler,
        "model_hata": round(ens_mae, 2),
        "guven":      guven,
        "veri_gun":   tum_sonuclar[0]["veri_gun"],
        "model_adi":  "Ensemble",
        "model_detay": " + ".join(model_isimleri),
    }


# ═══════════════════════════════════════════════
#  ANOMALİ TESPİTİ
# ═══════════════════════════════════════════════

def anomali_tespit(gecmis: dict) -> list[dict]:
    """
    Üç yöntemle anomali tespiti:
      1. Z-Score  — normal dağılımdan sapma
      2. IQR     — çeyrekler arası uzaklık
      3. Isolation Forest — sklearn tabanlı

    Her anomali: {"tarih", "metrik", "deger", "yontem", "skor", "aciklama"}
    """
    if not ANOMALI_AKTIF or len(gecmis) < ANOMALI_MIN_GUN:
        return []

    df = _gecmis_df(gecmis)
    if df.empty or len(df) < ANOMALI_MIN_GUN:
        return []

    anomaliler = []
    metrikler = {
        "doluluk":  ("Doluluk (%)",  df["doluluk"].values),
        "yolcu":    ("Yolcu Sayısı", df["yolcu"].values),
        "arac":     ("Araç Sayısı",  df["arac"].values),
    }

    for metrik_key, (metrik_adi, degerler) in metrikler.items():
        if len(degerler) < ANOMALI_MIN_GUN:
            continue

        ortalama = np.mean(degerler)
        std = np.std(degerler)

        # ── 1. Z-Score ──
        if std > 0:
            z_skorlar = (degerler - ortalama) / std
            for i, z in enumerate(z_skorlar):
                if abs(z) > ANOMALI_ZSCORE_ESIK:
                    yon = "yüksek" if z > 0 else "düşük"
                    anomaliler.append({
                        "tarih":    df.iloc[i]["tarih"],
                        "metrik":   metrik_adi,
                        "deger":    round(float(degerler[i]), 1),
                        "yontem":   "Z-Score",
                        "skor":     round(float(abs(z)), 2),
                        "aciklama": f"Normalden {yon} (z={z:.2f}, ort={ortalama:.1f}±{std:.1f})",
                    })

        # ── 2. IQR ──
        q1 = np.percentile(degerler, 25)
        q3 = np.percentile(degerler, 75)
        iqr = q3 - q1
        alt_sinir = q1 - ANOMALI_IQR_CARPAN * iqr
        ust_sinir = q3 + ANOMALI_IQR_CARPAN * iqr

        for i, val in enumerate(degerler):
            if val < alt_sinir or val > ust_sinir:
                yon = "yüksek" if val > ust_sinir else "düşük"
                uzaklik = abs(val - ust_sinir) if val > ust_sinir else abs(alt_sinir - val)
                # Z-Score ile aynı gün+metriği tekrar ekleme
                zaten_var = any(
                    a["tarih"] == df.iloc[i]["tarih"] and a["metrik"] == metrik_adi
                    for a in anomaliler
                )
                if not zaten_var:
                    anomaliler.append({
                        "tarih":    df.iloc[i]["tarih"],
                        "metrik":   metrik_adi,
                        "deger":    round(float(val), 1),
                        "yontem":   "IQR",
                        "skor":     round(float(uzaklik / max(iqr, 0.1)), 2),
                        "aciklama": f"Normalden {yon} (Q1={q1:.1f}, Q3={q3:.1f}, "
                                    f"sınır=[{alt_sinir:.1f}, {ust_sinir:.1f}])",
                    })

        # ── 3. Isolation Forest ──
        if len(degerler) >= ANOMALI_MIN_GUN:
            try:
                X_if = degerler.reshape(-1, 1)
                iso = IsolationForest(
                    contamination=ANOMALI_IF_KONTAMINASYON,
                    random_state=42,
                    n_estimators=100,
                )
                etiketler = iso.fit_predict(X_if)
                skorlar = iso.decision_function(X_if)

                for i, (etiket, skor) in enumerate(zip(etiketler, skorlar)):
                    if etiket == -1:  # anomali
                        zaten_var = any(
                            a["tarih"] == df.iloc[i]["tarih"] and a["metrik"] == metrik_adi
                            for a in anomaliler
                        )
                        if not zaten_var:
                            anomaliler.append({
                                "tarih":    df.iloc[i]["tarih"],
                                "metrik":   metrik_adi,
                                "deger":    round(float(degerler[i]), 1),
                                "yontem":   "Isolation Forest",
                                "skor":     round(float(abs(skor)), 3),
                                "aciklama": f"Anormal pattern tespit edildi (skor={skor:.3f})",
                            })
            except Exception as e:
                log.debug(f"  IF anomali hatası ({metrik_key}): {e}")

    # Tarihe göre sırala
    anomaliler.sort(key=lambda a: a["tarih"], reverse=True)
    return anomaliler


# ── Anomali Grafikleri ────────────────────────────

def grafik_anomali(gecmis: dict, anomaliler: list[dict]) -> bytes:
    """
    Doluluk, yolcu ve araç serilerini çizer.
    Anomali noktalarını kırmızı daire ile işaretler.
    """
    if not anomaliler or len(gecmis) < 2:
        return b""

    df = _gecmis_df(gecmis)
    if df.empty:
        return b""

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor="white",
                              sharex=True)
    fig.suptitle("Anomali Tespiti — Zaman Serisi Analizi",
                 fontsize=13, fontweight="bold", color=RENK_LACIVERT, y=0.98)

    etiketler = [t[5:].replace("-", "/") for t in df["tarih"]]
    metrik_harita = {
        "Doluluk (%)":  ("doluluk", RENK_YESIL,     "%", 0),
        "Yolcu Sayısı": ("yolcu",   RENK_ACIK_MAVİ, "",  1),
        "Araç Sayısı":  ("arac",    RENK_TURUNCU,   "",  2),
    }

    for metrik_adi, (sutun, renk, birim, ax_idx) in metrik_harita.items():
        ax = axes[ax_idx]
        ax.set_facecolor(RENK_ARK)
        degerler = df[sutun].values

        # Ana çizgi
        ax.plot(etiketler, degerler, color=renk, marker="o",
                linewidth=2, markersize=5, label=metrik_adi, zorder=2)
        ax.fill_between(etiketler, degerler, alpha=0.08, color=renk)

        # Ortalama çizgisi
        ort = np.mean(degerler)
        ax.axhline(ort, color=RENK_GRI, linestyle="--", linewidth=1, alpha=0.7,
                   label=f"Ort: {ort:.1f}")

        # ±2σ bandı
        std = np.std(degerler)
        if std > 0:
            ax.fill_between(etiketler, ort - 2 * std, ort + 2 * std,
                           alpha=0.08, color=RENK_GRI, label="±2σ bandı")

        # Anomali noktaları
        anomali_idx = []
        anomali_val = []
        anomali_tip = []
        for a in anomaliler:
            if a["metrik"] == metrik_adi:
                try:
                    idx = df[df["tarih"] == a["tarih"]].index[0]
                    rel_idx = list(df.index).index(idx)
                    anomali_idx.append(rel_idx)
                    anomali_val.append(a["deger"])
                    anomali_tip.append(a["yontem"])
                except (IndexError, ValueError):
                    pass

        if anomali_idx:
            anomali_x = [etiketler[i] for i in anomali_idx]
            ax.scatter(anomali_x, anomali_val, color=RENK_KIRMIZI,
                      s=120, zorder=5, edgecolors="white", linewidths=2,
                      label=f"Anomali ({len(anomali_idx)})")
            for x, v, tip in zip(anomali_x, anomali_val, anomali_tip):
                ax.annotate(f"{tip}\n{birim}{v:.0f}", (x, v),
                           xytext=(8, 12), textcoords="offset points",
                           fontsize=7, color=RENK_KIRMIZI, fontweight="bold",
                           arrowprops=dict(arrowstyle="->", color=RENK_KIRMIZI,
                                          lw=1.2),
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                    edgecolor=RENK_KIRMIZI, alpha=0.9))

        ax.set_ylabel(metrik_adi, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _grafik_kapat_bytes()


# ── ML Grafikleri ──────────────────────────────

def grafik_ml_tahmin(gecmis: dict, tahmin_sonuc: dict) -> bytes:
    """
    Gerçek geçmiş doluluk + tahmin eğrisi.
    Güven aralığı (±MAE) gri bantla gösterilir.
    """
    if not tahmin_sonuc:
        return b""

    df = _gecmis_df(gecmis)
    if df.empty:
        return b""

    gercek_tarih  = df["tarih"].tolist()
    gercek_dol    = df["doluluk"].tolist()
    tahmin_tarih  = [t[0] for t in tahmin_sonuc["tahminler"]]
    tahmin_dol    = [t[1] for t in tahmin_sonuc["tahminler"]]
    mae           = tahmin_sonuc["model_hata"]

    tum_tarih = gercek_tarih + tahmin_tarih
    # kısa etiketler
    etiket = [t[5:].replace("-", "/") for t in tum_tarih]

    fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")
    ax.set_facecolor(RENK_ARK)
    n = len(gercek_dol)

    # Gerçek çizgi
    ax.plot(etiket[:n], gercek_dol, color=RENK_ACIK_MAVİ,
            marker="o", linewidth=2.5, markersize=7, label="Gerçek Doluluk")
    ax.fill_between(etiket[:n], gercek_dol, alpha=0.12, color=RENK_ACIK_MAVİ)

    # Bağlantı noktası (son gerçek → ilk tahmin)
    ax.plot([etiket[n - 1], etiket[n]],
            [gercek_dol[-1], tahmin_dol[0]],
            color=RENK_TURUNCU, linewidth=1.5, linestyle="--")

    # Tahmin çizgisi
    ax.plot(etiket[n:], tahmin_dol, color=RENK_TURUNCU,
            marker="s", linewidth=2.5, markersize=7,
            linestyle="--", label=f"Tahmin (±{mae:.1f}% hata)")

    # Güven bandı
    alt = [max(0,   v - mae) for v in tahmin_dol]
    ust = [min(100, v + mae) for v in tahmin_dol]
    ax.fill_between(etiket[n:], alt, ust, alpha=0.18, color=RENK_TURUNCU, label="Güven aralığı")

    # Referans çizgileri
    ax.axhline(80, color=RENK_GRI,    linestyle=":",  linewidth=1.2, alpha=0.8, label="Hedef %80")
    ax.axhline(60, color=RENK_KIRMIZI, linestyle=":", linewidth=1,   alpha=0.6, label="Kritik %60")

    # Bugün ayırıcı
    ax.axvline(etiket[n - 1], color=RENK_LACIVERT, linewidth=1.5,
               linestyle="-.", alpha=0.5, label="Bugün")
    ax.text(etiket[n - 1], 2, "  Bugün", color=RENK_LACIVERT, fontsize=8, va="bottom")

    # Değer etiketleri — tahmin noktaları
    for i, (e, v) in enumerate(zip(etiket[n:], tahmin_dol)):
        ax.annotate(f"%{v:.0f}", (e, v), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    fontsize=8, color=RENK_TURUNCU, fontweight="bold")

    ax.set_ylim(0, 115)
    ax.set_title(
        f"Doluluk Tahmini — Önümüzdeki {ML_TAHMIN_GUN} Gün  "
        f"(Güven: {tahmin_sonuc['guven']}, {tahmin_sonuc['veri_gun']} günlük veri)",
        fontsize=12, fontweight="bold", color=RENK_LACIVERT, pad=10,
    )
    ax.set_ylabel("Doluluk Oranı (%)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    return _grafik_kapat_bytes()


def grafik_ml_firma_onerisi(oneriler: list[dict]) -> bytes:
    """
    Öneri üretilen firmalar için mevcut vs önerilen araç sayısı,
    ve mevcut doluluk çubuk grafiği — yan yana 2 panel.
    """
    if not oneriler:
        return b""

    firmalar   = [o["firma"][:18]      for o in oneriler]
    mev_dol    = [o["mevcut_dol"]      for o in oneriler]
    mev_arac   = [o["mevcut_arac"]     for o in oneriler]
    hedef_dol  = oneriler[0]["hedef_dol"]  # hepsi için aynı

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, max(4, len(oneriler) * 0.9 + 2)),
                                    facecolor="white")
    fig.suptitle("Düşük Performanslı Firmalar — Kapasite Önerisi",
                 fontsize=13, fontweight="bold", color=RENK_LACIVERT)

    # Sol: Doluluk karşılaştırması
    renk_dol = [RENK_YESIL if d >= 80 else RENK_SARI if d >= 60 else RENK_KIRMIZI
                for d in mev_dol]
    bax = ax1.barh(firmalar, mev_dol, color=renk_dol, edgecolor="white", height=0.55)
    ax1.axvline(hedef_dol, color=RENK_ACIK_MAVİ, linewidth=2,
                linestyle="--", label=f"Hedef %{hedef_dol:.0f}")
    for bar, v in zip(bax, mev_dol):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"%{v:.1f}", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Doluluk Oranı (%)", fontsize=9)
    ax1.set_title("Mevcut Doluluk", fontsize=11, fontweight="bold",
                  color=RENK_LACIVERT, pad=8)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)

    # Sağ: Araç delta (mevcut mavi, delta kırmızı/yeşil)
    delta_renkler = [RENK_YESIL if o["delta_arac"] > 0
                     else RENK_KIRMIZI if o["delta_arac"] < 0
                     else RENK_GRI for o in oneriler]
    delta_vals = [o["delta_arac"] for o in oneriler]
    bax2 = ax2.barh(firmalar, delta_vals, color=delta_renkler, edgecolor="white", height=0.55)
    ax2.axvline(0, color=RENK_LACIVERT, linewidth=1.2, alpha=0.6)
    for bar, v in zip(bax2, delta_vals):
        offset = 0.15 if v >= 0 else -0.15
        ax2.text(v + offset, bar.get_y() + bar.get_height() / 2,
                 f"{'+' if v > 0 else ''}{v}", va="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Araç Değişim Önerisi (+artır / −azalt)", fontsize=9)
    ax2.set_title("Kapasite Ayarı Önerisi", fontsize=11, fontweight="bold",
                  color=RENK_LACIVERT, pad=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    handles = [
        mpatches.Patch(color=RENK_YESIL,   label="Araç artır / Doluluk iyi"),
        mpatches.Patch(color=RENK_KIRMIZI, label="Araç azalt / Doluluk kritik"),
        mpatches.Patch(color=RENK_SARI,    label="Doluluk düşük"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.00))
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    return _grafik_kapat_bytes()


# ── ML Telegram Mesajı ─────────────────────────

def telegram_ml_mesaj(tahmin: dict | None, oneriler: list[dict],
                      anomaliler: list[dict] | None = None) -> str:
    satir = "━━━━━━━━━━━━━━━━━━━━━\n"
    m = f"🤖 *MAKİNE ÖĞRENMESİ ANALİZİ*\n{satir}"

    # Haftalık doluluk tahmini
    if tahmin:
        model_adi = tahmin.get("model_adi", "Ridge")
        model_detay = tahmin.get("model_detay", "")
        m += f"📈 *7 GÜNLÜK DOLULUK TAHMİNİ*\n"
        m += f"_Model: {model_adi} | Güven: {tahmin['guven']} | ±{tahmin['model_hata']:.1f}% hata_\n"
        if model_detay:
            m += f"_Detay: {model_detay[:60]}_\n"
        m += "\n"
        for tarih, dol in tahmin["tahminler"]:
            dt    = datetime.strptime(tarih, "%Y-%m-%d")
            gun   = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"][dt.weekday()]
            emo   = "🟢" if dol >= 80 else "🟡" if dol >= 60 else "🔴"
            m += f"{emo} `{dt.strftime('%d.%m')} {gun}` → *%{dol:.1f}*\n"
        m += satir
    else:
        m += f"⏳ _Tahmin için en az {ML_MIN_GUN} günlük veri gerekiyor._\n{satir}"

    # Anomali tespiti
    if anomaliler:
        m += f"🔍 *ANOMALİ TESPİTİ*  ({len(anomaliler)} anomali)\n"
        for a in anomaliler[:8]:  # En fazla 8 anomali göster
            m += f"⚠️ `{a['tarih'][5:]}` *{a['metrik']}*: {a['deger']}\n"
            m += f"   _{a['yontem']}: {a['aciklama'][:50]}_\n"
        if len(anomaliler) > 8:
            m += f"   _...ve {len(anomaliler) - 8} anomali daha (PDF'te)_\n"
        m += satir
    elif ANOMALI_AKTIF:
        m += "✅ _Anomali tespit edilmedi — tüm metrikler normal aralıkta._\n"
        m += satir

    # Kapasite önerileri
    if oneriler:
        m += f"🔧 *KAPASİTE ÖNERİLERİ*  ({len(oneriler)} firma)\n"
        for o in oneriler:
            delta = o["delta_arac"]
            emo   = "📉" if delta < 0 else "📈" if delta > 0 else "➡️"
            m += f"\n{emo} *{o['firma']}* [{o['kategori']}]\n"
            m += f"   Doluluk: %{o['mevcut_dol']:.1f} (hedef: %{o['hedef_dol']:.1f})\n"
            m += f"   💡 {o['onerim']}\n"
        m += satir
    else:
        m += "✅ _Tüm firmalar hedef doluluk oranında, kapasite önerisi yok._\n"

    m += "_📊 Grafik ve detaylar aşağıda gönderilecektir._"
    return m


# ── ML PDF Bölümü (mevcut PDF'e eklenir) ───────

def _ml_pdf_bolumu(hikaye: list, st: dict,
                   tahmin: dict | None, oneriler: list[dict],
                   tahmin_grafik: bytes, oneri_grafik: bytes,
                   anomaliler: list[dict] | None = None,
                   anomali_grafik: bytes = b""):
    """
    Mevcut günlük PDF'e ML sayfası ekler.
    hikaye listesini in-place günceller.
    """
    hikaye.append(PageBreak())
    hikaye.append(Paragraph("🤖 Makine Öğrenmesi Analizi", st["rapor_baslik"]))
    hikaye.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor(RENK_LACIVERT)))
    hikaye.append(Spacer(1, 0.4*cm))

    # — Tahmin tablosu —
    model_adi = tahmin.get("model_adi", "Ridge") if tahmin else ""
    model_detay = tahmin.get("model_detay", "") if tahmin else ""
    hikaye.append(Paragraph(f"Haftalık Doluluk Tahmini — {model_adi}", st["bolum"]))
    if tahmin:
        hikaye.append(Paragraph(
            f"Model: <b>{model_adi}</b>  |  "
            f"Güven: <b>{tahmin['guven']}</b>  |  "
            f"Hata: <b>±{tahmin['model_hata']:.1f}%</b>  |  "
            f"Veri: <b>{tahmin['veri_gun']} gün</b>",
            st["normal"]
        ))
        if model_detay:
            hikaye.append(Paragraph(
                f"<i>Detay: {model_detay[:80]}</i>",
                st["normal"]
            ))
        hikaye.append(Spacer(1, 0.3*cm))

        baslik = [["TARİH", "GÜN", "TAHMİN DOLULUK", "DURUM"]]
        satirlar = baslik[:]
        for tarih, dol in tahmin["tahminler"]:
            dt   = datetime.strptime(tarih, "%Y-%m-%d")
            gun  = ["Pazartesi","Salı","Çarşamba","Perşembe","Cuma","Cumartesi","Pazar"][dt.weekday()]
            dur  = "✔ İyi" if dol >= 80 else "⚠ Düşük" if dol >= 60 else "✖ Kritik"
            satirlar.append([dt.strftime("%d.%m.%Y"), gun, f"%{dol:.1f}", dur])

        t = Table(satirlar, repeatRows=1, colWidths=[3.5*cm, 4*cm, 4.5*cm, 5*cm])
        style = [
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor(RENK_LACIVERT)),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  _FONT_BOLD),
            ("FONTNAME",      (0, 1), (-1, -1), _FONT_NORMAL),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF4FB")]),
            ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#C0CDD8")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        # Doluluk sütununu renklendir
        for i, (_, dol) in enumerate(tahmin["tahminler"], start=1):
            bg = (colors.HexColor("#D5F5E3") if dol >= 80
                  else colors.HexColor("#FEF9E7") if dol >= 60
                  else colors.HexColor("#FADBD8"))
            style.append(("BACKGROUND", (2, i), (2, i), bg))
        t.setStyle(TableStyle(style))
        hikaye.append(t)
        hikaye.append(Spacer(1, 0.4*cm))

        if tahmin_grafik:
            img_buf = io.BytesIO(tahmin_grafik)
            hikaye.append(RLImage(img_buf, width=17*cm, height=7*cm))
            hikaye.append(Spacer(1, 0.5*cm))
    else:
        hikaye.append(Paragraph(
            f"Tahmin için en az {ML_MIN_GUN} günlük veri gerekiyor. "
            "Sistem veri biriktirmeye devam ediyor.", st["normal"]
        ))
        hikaye.append(Spacer(1, 0.4*cm))

    # — Kapasite önerileri tablosu —
    hikaye.append(Paragraph("Kapasite Optimizasyon Önerileri", st["bolum"]))
    if oneriler:
        baslik2 = [["FİRMA", "KATEGORİ", "MEVCUT\nARAÇ", "MEVCUT\nDOLULUK",
                    "HEDEF\nDOLULUK", "DELTA\nARAÇ", "ÖNERİ"]]
        satirlar2 = baslik2[:]
        for o in oneriler:
            delta = o["delta_arac"]
            d_str = f"+{delta}" if delta > 0 else str(delta)
            satirlar2.append([
                o["firma"][:22], o["kategori"],
                str(o["mevcut_arac"]),
                f"%{o['mevcut_dol']:.1f}",
                f"%{o['hedef_dol']:.1f}",
                d_str,
                o["onerim"][:60],
            ])
        t2 = Table(satirlar2, repeatRows=1,
                   colWidths=[3.5*cm, 2.2*cm, 1.8*cm, 2*cm, 2*cm, 1.8*cm, 5.7*cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#C0392B")),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTNAME",      (0, 0), (-1, 0),  _FONT_BOLD),
            ("FONTNAME",      (0, 1), (-1, -1), _FONT_NORMAL),
            ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("ALIGN",         (6, 1), (6, -1),  "LEFT"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#FEF9F9"),
                                                  colors.HexColor("#FDF2F2")]),
            ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#E8BBBB")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (6, 1), (6, -1),  6),
        ]))
        hikaye.append(t2)
        hikaye.append(Spacer(1, 0.4*cm))

        if oneri_grafik:
            img_buf2 = io.BytesIO(oneri_grafik)
            hikaye.append(RLImage(img_buf2, width=17*cm,
                                  height=max(5*cm, len(oneriler) * 1.2*cm + 2*cm)))
    else:
        hikaye.append(Paragraph(
            "Tüm firmalar hedef doluluk oranında. Kapasite önerisi bulunmuyor.",
            st["normal"]
        ))

    # — Anomali Tespiti bölümü —
    if anomaliler is not None:
        hikaye.append(PageBreak())
        hikaye.append(Paragraph("🔍 Anomali Tespiti", st["bolum"]))
        if anomaliler:
            hikaye.append(Paragraph(
                f"<b>{len(anomaliler)} anomali</b> tespit edildi "
                f"(Z-Score, IQR ve Isolation Forest yöntemleriyle).",
                st["normal"]
            ))
            hikaye.append(Spacer(1, 0.3*cm))

            baslik_a = [["TARİH", "METRİK", "DEĞER", "YÖNTEM", "SKOR", "AÇIKLAMA"]]
            satirlar_a = baslik_a[:]
            for a in anomaliler[:15]:  # PDF'te en fazla 15 anomali
                satirlar_a.append([
                    a["tarih"],
                    a["metrik"],
                    str(a["deger"]),
                    a["yontem"],
                    str(a["skor"]),
                    a["aciklama"][:45],
                ])

            ta = Table(satirlar_a, repeatRows=1,
                       colWidths=[2.5*cm, 2.5*cm, 1.8*cm, 2.5*cm, 1.5*cm, 6.2*cm])
            ta.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#8E44AD")),
                ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
                ("FONTNAME",      (0, 0), (-1, 0),  _FONT_BOLD),
                ("FONTNAME",      (0, 1), (-1, -1), _FONT_NORMAL),
                ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
                ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
                ("ALIGN",         (5, 1), (5, -1),  "LEFT"),
                ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#F9F0FF"),
                                                      colors.HexColor("#F3E5F5")]),
                ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#D1A8E0")),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            hikaye.append(ta)
            hikaye.append(Spacer(1, 0.4*cm))

            if anomali_grafik:
                img_buf_a = io.BytesIO(anomali_grafik)
                hikaye.append(RLImage(img_buf_a, width=17*cm, height=10*cm))
        else:
            hikaye.append(Paragraph(
                "Tüm metrikler normal aralıkta. Anomali tespit edilmedi.",
                st["normal"]
            ))


# ───────────────────────────────────────────────
#  UYARI SİSTEMİ
# ───────────────────────────────────────────────

def uyarilari_hesapla(sehir_df: pd.DataFrame, santiye_df: pd.DataFrame) -> list[dict]:
    """
    Her firma için uyarı üretir.
    Döndürdüğü liste: [{"firma": str, "tip": "kirmizi"|"sari", "mesaj": str}, ...]
    """
    uyarilar = []

    def _tara(df: pd.DataFrame, firma_col: str, kategori: str):
        if df.empty:
            return
        for _, row in df.iterrows():
            firma   = str(row[firma_col])[:30]
            arac    = int(row["Farkli_Otobus"])
            kap     = int(row["Toplam_Kapasite"])
            yolcu   = int(row["Toplam_Yolcu"])
            doluluk = yolcu / kap * 100 if kap > 0 else 0

            if doluluk < ESIK_KIRMIZI_DOLULUK:
                uyarilar.append({
                    "firma": firma, "kategori": kategori,
                    "tip": "kirmizi",
                    "mesaj": f"Kritik doluluk: %{doluluk:.1f} ({yolcu}/{kap} yolcu)",
                })
            elif doluluk < ESIK_SARI_DOLULUK:
                uyarilar.append({
                    "firma": firma, "kategori": kategori,
                    "tip": "sari",
                    "mesaj": f"Düşük doluluk: %{doluluk:.1f} ({yolcu}/{kap} yolcu)",
                })

            if arac < ESIK_MIN_ARAC:
                uyarilar.append({
                    "firma": firma, "kategori": kategori,
                    "tip": "sari",
                    "mesaj": f"Az araç: sadece {arac} aktif araç sahadaki",
                })

    _tara(sehir_df,   "OTOBUS FIRMASI", "Şehir")
    _tara(santiye_df, "организация",    "Şantiye")
    return uyarilar


def en_iyi_en_kotu(sehir_df: pd.DataFrame, santiye_df: pd.DataFrame):
    """Doluluk bazında en iyi ve en kötü firmayı döndürür."""
    parcalar = []
    if not sehir_df.empty:
        s = sehir_df.rename(columns={"OTOBUS FIRMASI": "Firma"}).copy()
        parcalar.append(s)
    if not santiye_df.empty:
        r = santiye_df.rename(columns={"организация": "Firma"}).copy()
        parcalar.append(r)
    if not parcalar:
        return None, None

    df = pd.concat(parcalar, ignore_index=True)
    df["Doluluk"] = df.apply(
        lambda r: r["Toplam_Yolcu"] / r["Toplam_Kapasite"] * 100
        if r["Toplam_Kapasite"] > 0 else 0, axis=1
    )
    en_iyi  = df.loc[df["Doluluk"].idxmax()]
    en_kotu = df.loc[df["Doluluk"].idxmin()]
    return en_iyi, en_kotu


# ───────────────────────────────────────────────
#  TELEGRAM
# ───────────────────────────────────────────────

def _tg_post(endpoint: str, **kwargs) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/{endpoint}"
    try:
        r = requests.post(url, timeout=30, **kwargs)
        if r.status_code == 200:
            return True
        log.error(f"Telegram {endpoint} hata {r.status_code}: {r.text[:200]}")
    except Exception as e:
        log.error(f"Telegram bağlantı hatası ({endpoint}): {e}")
    return False


def telegram_mesaj_gonder(mesaj: str) -> bool:
    return _tg_post("sendMessage", json={
        "chat_id": TELEGRAM_CHAT, "text": mesaj, "parse_mode": "Markdown"
    })


def telegram_foto_gonder(foto_bytes: bytes, baslik: str = "") -> bool:
    return _tg_post("sendPhoto",
        data={"chat_id": TELEGRAM_CHAT, "caption": baslik},
        files={"photo": ("grafik.png", foto_bytes, "image/png")},
    )


def telegram_dosya_gonder(dosya_bytes: bytes, dosya_adi: str, baslik: str = "") -> bool:
    return _tg_post("sendDocument",
        data={"chat_id": TELEGRAM_CHAT, "caption": baslik},
        files={"document": (dosya_adi, dosya_bytes, "application/pdf")},
    )


# ───────────────────────────────────────────────
#  GEÇMİŞ VERİ (haftalık trend)
# ───────────────────────────────────────────────

FIRMA_GECMIS_DOSYA = "firma_gecmis.json"


def gecmis_kaydet(tarih_str: str, sehir_df: pd.DataFrame, santiye_df: pd.DataFrame):
    ta, tk, ty = 0, 0, 0
    for df in [sehir_df, santiye_df]:
        if not df.empty:
            ta += int(df["Farkli_Otobus"].sum())
            tk += int(df["Toplam_Kapasite"].sum())
            ty += int(df["Toplam_Yolcu"].sum())
    dol = round(ty / tk * 100, 1) if tk > 0 else 0

    # SQLite'a kaydet
    try:
        with _db_baglanti() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO gunluk_gecmis (tarih, arac, kapasite, yolcu, doluluk)
                VALUES (?, ?, ?, ?, ?)
            """, (tarih_str, ta, tk, ty, dol))
        log.info(f"📦 Geçmiş kaydedildi (DB): {tarih_str}")
    except Exception as e:
        log.error(f"DB geçmiş kayıt hatası: {e}")

    # Firma bazlı — SQLite
    firma_sayac = 0
    try:
        with _db_baglanti() as conn:
            if not sehir_df.empty:
                for _, row in sehir_df.iterrows():
                    firma = str(row["OTOBUS FIRMASI"])[:30]
                    kap = int(row["Toplam_Kapasite"])
                    yolcu = int(row["Toplam_Yolcu"])
                    conn.execute("""
                        INSERT OR REPLACE INTO firma_gecmis
                        (tarih, firma, kategori, arac, sefer, kapasite, yolcu, doluluk)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (tarih_str, firma, "Şehir",
                          int(row["Farkli_Otobus"]), int(row["Toplam_Sefer"]),
                          kap, yolcu, round(yolcu/kap*100, 1) if kap > 0 else 0))
                    firma_sayac += 1
            if not santiye_df.empty:
                for _, row in santiye_df.iterrows():
                    firma = str(row["организация"])[:30]
                    kap = int(row["Toplam_Kapasite"])
                    yolcu = int(row["Toplam_Yolcu"])
                    conn.execute("""
                        INSERT OR REPLACE INTO firma_gecmis
                        (tarih, firma, kategori, arac, sefer, kapasite, yolcu, doluluk)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (tarih_str, firma, "Şantiye",
                          int(row["Farkli_Otobus"]), int(row["Toplam_Sefer"]),
                          kap, yolcu, round(yolcu/kap*100, 1) if kap > 0 else 0))
                    firma_sayac += 1
        log.info(f"📦 Firma geçmişi kaydedildi (DB): {firma_sayac} firma")
    except Exception as e:
        log.error(f"DB firma kayıt hatası: {e}")

    # JSON yedek (opsiyonel — geriye uyumluluk)
    if DB_JSON_YEDEK:
        try:
            kayit = {}
            if os.path.exists(GECMIS_DOSYA):
                with open(GECMIS_DOSYA, "r", encoding="utf-8") as f:
                    kayit = json.load(f)
            kayit[tarih_str] = {"arac": ta, "kapasite": tk, "yolcu": ty, "doluluk": dol}
            son_60 = sorted(kayit)[-60:]
            kayit = {k: kayit[k] for k in son_60}
            with open(GECMIS_DOSYA, "w", encoding="utf-8") as f:
                json.dump(kayit, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def gecmis_son_n_gun(n: int = 7) -> dict:
    """Son N günün verisini SQLite'dan okur."""
    try:
        with _db_baglanti() as conn:
            rows = conn.execute("""
                SELECT tarih, arac, kapasite, yolcu, doluluk
                FROM gunluk_gecmis ORDER BY tarih DESC LIMIT ?
            """, (n,)).fetchall()
        if not rows:
            return {}
        sonuc = {}
        for r in reversed(rows):
            sonuc[r["tarih"]] = {
                "arac": r["arac"], "kapasite": r["kapasite"],
                "yolcu": r["yolcu"], "doluluk": r["doluluk"],
            }
        return sonuc
    except Exception as e:
        log.error(f"DB geçmiş okuma hatası: {e}")
        return {}


def gecmis_tum() -> dict:
    """Tüm geçmiş verisini SQLite'dan okur."""
    try:
        with _db_baglanti() as conn:
            rows = conn.execute("""
                SELECT tarih, arac, kapasite, yolcu, doluluk
                FROM gunluk_gecmis ORDER BY tarih
            """).fetchall()
        return {r["tarih"]: {"arac": r["arac"], "kapasite": r["kapasite"],
                             "yolcu": r["yolcu"], "doluluk": r["doluluk"]}
                for r in rows}
    except Exception:
        return {}


def firma_gecmis_oku() -> dict:
    """Firma bazlı geçmiş verisini SQLite'dan okur. Eski JSON formatında döndürür."""
    try:
        with _db_baglanti() as conn:
            rows = conn.execute("""
                SELECT tarih, firma, kategori, arac, sefer, kapasite, yolcu, doluluk
                FROM firma_gecmis ORDER BY tarih
            """).fetchall()
        sonuc = {}
        for r in rows:
            if r["tarih"] not in sonuc:
                sonuc[r["tarih"]] = {}
            sonuc[r["tarih"]][r["firma"]] = {
                "kategori": r["kategori"], "arac": r["arac"],
                "sefer": r["sefer"], "kapasite": r["kapasite"],
                "yolcu": r["yolcu"], "doluluk": r["doluluk"],
            }
        return sonuc
    except Exception:
        return {}


# ───────────────────────────────────────────────
#  GRAFİKLER
# ───────────────────────────────────────────────

def _grafik_kapat_bytes() -> bytes:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf.read()


def _ax_temizle(ax, baslik: str):
    ax.set_facecolor(RENK_ARK)
    ax.set_title(baslik, fontsize=11, fontweight="bold", pad=8, color=RENK_LACIVERT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CBD5E0")


def grafik_gunluk_ozet(sehir_df: pd.DataFrame, santiye_df: pd.DataFrame,
                        tarih_str: str, uyarilar: list[dict]) -> bytes:
    """3 panelli günlük grafik: araç | doluluk | uyarı sayacı."""
    parcalar = []
    if not sehir_df.empty:
        s = sehir_df.rename(columns={"OTOBUS FIRMASI": "Firma"})[
            ["Firma", "Farkli_Otobus", "Toplam_Kapasite", "Toplam_Yolcu"]].copy()
        s["Tip"] = "Şehir"
        parcalar.append(s)
    if not santiye_df.empty:
        r = santiye_df.rename(columns={"организация": "Firma"})[
            ["Firma", "Farkli_Otobus", "Toplam_Kapasite", "Toplam_Yolcu"]].copy()
        r["Tip"] = "Şantiye"
        parcalar.append(r)
    if not parcalar:
        return b""

    df = pd.concat(parcalar, ignore_index=True)
    df["Doluluk"] = (df["Toplam_Yolcu"] / df["Toplam_Kapasite"] * 100).fillna(0).round(1)
    df = df.sort_values("Farkli_Otobus", ascending=True)

    renkler_tip = [RENK_ACIK_MAVİ if t == "Şehir" else RENK_TURUNCU for t in df["Tip"]]
    renkler_dol = [
        RENK_YESIL if d >= 80 else RENK_SARI if d >= 60 else RENK_KIRMIZI
        for d in df["Doluluk"]
    ]

    fig = plt.figure(figsize=(16, max(6, len(df) * 0.8 + 2)), facecolor="white")
    fig.suptitle(f"Günlük Ulaşım Özeti  —  {tarih_str}",
                 fontsize=14, fontweight="bold", color=RENK_LACIVERT, y=0.98)
    gs = GridSpec(1, 2, figure=fig, wspace=0.4)

    # Sol: Araç Sayısı
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.barh(df["Firma"], df["Farkli_Otobus"], color=renkler_tip,
                    edgecolor="white", linewidth=0.8, height=0.6)
    for bar, val in zip(bars, df["Farkli_Otobus"]):
        ax1.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                 str(int(val)), va="center", fontsize=9, fontweight="bold", color=RENK_LACIVERT)
    _ax_temizle(ax1, "Firma Bazlı Aktif Araç Sayısı")
    ax1.set_xlabel("Araç Sayısı", fontsize=9)

    # Sağ: Doluluk
    ax2 = fig.add_subplot(gs[1])
    bars2 = ax2.barh(df["Firma"], df["Doluluk"], color=renkler_dol,
                     edgecolor="white", linewidth=0.8, height=0.6)
    ax2.axvline(80, color=RENK_GRI, linestyle="--", linewidth=1.2, alpha=0.8, label="Hedef %80")
    ax2.axvline(60, color=RENK_KIRMIZI, linestyle=":", linewidth=1, alpha=0.6, label="Kritik %60")
    for bar, val in zip(bars2, df["Doluluk"]):
        ax2.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                 f"%{val:.1f}", va="center", fontsize=9, fontweight="bold", color=RENK_LACIVERT)
    _ax_temizle(ax2, "Firma Bazlı Doluluk Oranı")
    ax2.set_xlabel("Doluluk (%)", fontsize=9)
    ax2.set_xlim(0, 115)
    ax2.legend(fontsize=8, framealpha=0.7)

    # Lejant
    handles = [
        mpatches.Patch(color=RENK_ACIK_MAVİ,  label="Şehir Servisi"),
        mpatches.Patch(color=RENK_TURUNCU,     label="Şantiye Servisi"),
        mpatches.Patch(color=RENK_YESIL,       label="≥%80 Doluluk"),
        mpatches.Patch(color=RENK_SARI,        label="%60–80 Doluluk"),
        mpatches.Patch(color=RENK_KIRMIZI,     label="<%60 Doluluk"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.00))
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    return _grafik_kapat_bytes()


def grafik_haftalik_trend(gecmis: dict) -> bytes:
    if len(gecmis) < 2:
        return b""

    tarihler   = list(gecmis.keys())
    araclar    = [gecmis[t]["arac"]    for t in tarihler]
    doluluklar = [gecmis[t]["doluluk"] for t in tarihler]
    yolcular   = [gecmis[t]["yolcu"]   for t in tarihler]
    etiketler  = [t[5:].replace("-", "/") for t in tarihler]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle("Haftalık Operasyon Trendi — Son 7 Gün",
                 fontsize=13, fontweight="bold", color=RENK_LACIVERT)

    # Sol: Araç + Doluluk (çift eksen)
    ax1.set_facecolor(RENK_ARK)
    ax1.plot(etiketler, araclar, color=RENK_ACIK_MAVİ, marker="o",
             linewidth=2.5, markersize=8, label="Araç Sayısı")
    ax1.fill_between(etiketler, araclar, alpha=0.12, color=RENK_ACIK_MAVİ)
    ax1.set_ylabel("Araç Sayısı", color=RENK_ACIK_MAVİ, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=RENK_ACIK_MAVİ)
    ax1.set_ylim(0, max(araclar) * 1.35 if araclar else 10)
    for i, v in enumerate(araclar):
        ax1.annotate(str(v), (etiketler[i], v),
                     xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=8, color=RENK_ACIK_MAVİ, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(etiketler, doluluklar, color=RENK_YESIL, marker="s",
             linewidth=2.5, markersize=8, linestyle="--", label="Doluluk %")
    ax2.axhline(80, color=RENK_GRI, linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_ylabel("Doluluk (%)", color=RENK_YESIL, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=RENK_YESIL)
    ax2.set_ylim(0, 115)
    for i, v in enumerate(doluluklar):
        ax2.annotate(f"%{v:.0f}", (etiketler[i], v),
                     xytext=(0, -14), textcoords="offset points",
                     ha="center", fontsize=8, color=RENK_YESIL, fontweight="bold")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.grid(axis="x", linestyle="--", alpha=0.4)
    ax1.set_title("Araç & Doluluk Trendi", fontsize=11, fontweight="bold",
                  color=RENK_LACIVERT, pad=8)

    # Sağ: Yolcu çubuk
    ax3.set_facecolor(RENK_ARK)
    bar_renkler = [RENK_YESIL if d >= 80 else RENK_SARI if d >= 60 else RENK_KIRMIZI
                   for d in doluluklar]
    bars = ax3.bar(etiketler, yolcular, color=bar_renkler, edgecolor="white",
                   linewidth=0.8, width=0.55)
    for bar, val in zip(bars, yolcular):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(yolcular) * 0.02,
                 f"{val:,}", ha="center", fontsize=8, fontweight="bold", color=RENK_LACIVERT)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.grid(axis="y", linestyle="--", alpha=0.4)
    ax3.set_title("Günlük Taşınan Yolcu Sayısı", fontsize=11, fontweight="bold",
                  color=RENK_LACIVERT, pad=8)
    ax3.set_ylabel("Yolcu Sayısı", fontsize=9)

    plt.tight_layout()
    return _grafik_kapat_bytes()


# ───────────────────────────────────────────────
#  PDF SAYFA ŞABLONU (header / footer)
# ───────────────────────────────────────────────

class RaporSayfa(BaseDocTemplate):
    def __init__(self, buf, tarih_str: str, rapor_turu: str = "GÜNLÜK", **kwargs):
        super().__init__(buf, **kwargs)
        self.tarih_str  = tarih_str
        self.rapor_turu = rapor_turu
        frame = Frame(
            self.leftMargin, self.bottomMargin,
            self.width, self.height - 1.5*cm,
            id="normal",
        )
        tpl = PageTemplate(id="ana", frames=[frame], onPage=self._sayfa_ciz)
        self.addPageTemplates([tpl])

    def _sayfa_ciz(self, canv: rl_canvas.Canvas, doc):
        canv.saveState()
        w, h = A4

        # ── Üst bant ──
        canv.setFillColor(colors.HexColor(RENK_LACIVERT))
        canv.rect(0, h - 1.3*cm, w, 1.3*cm, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont(_FONT_BOLD, 11)
        canv.drawString(1.8*cm, h - 0.85*cm, f"ULAŞIM OPERASYON RAPORU  —  {self.rapor_turu}")
        canv.setFont(_FONT_NORMAL, 9)
        canv.drawRightString(w - 1.8*cm, h - 0.85*cm, self.tarih_str)

        # ── Alt bant ──
        canv.setFillColor(colors.HexColor("#EEF4FB"))
        canv.rect(0, 0, w, 0.8*cm, fill=1, stroke=0)
        canv.setFillColor(colors.HexColor(RENK_GRI))
        canv.setFont(_FONT_NORMAL, 7.5)
        canv.drawString(1.8*cm, 0.28*cm,
                        f"Oluşturma: {datetime.now().strftime('%d.%m.%Y %H:%M')}  •  Otomatik rapor")
        canv.drawRightString(w - 1.8*cm, 0.28*cm, f"Sayfa {doc.page}")

        canv.restoreState()


# ───────────────────────────────────────────────
#  PDF YARDIMCI STİLLER
# ───────────────────────────────────────────────

def _stiller():
    base = getSampleStyleSheet()
    return {
        "rapor_baslik": ParagraphStyle(
            "RaporBaslik", parent=base["Title"],
            fontSize=22, textColor=colors.HexColor(RENK_LACIVERT),
            spaceAfter=2, alignment=TA_CENTER, fontName=_FONT_BOLD,
        ),
        "tarih": ParagraphStyle(
            "Tarih", parent=base["Normal"],
            fontSize=11, textColor=colors.HexColor(RENK_GRI),
            spaceAfter=14, alignment=TA_CENTER,
        ),
        "bolum": ParagraphStyle(
            "Bolum", parent=base["Heading2"],
            fontSize=12, textColor=colors.HexColor(RENK_LACIVERT),
            spaceBefore=16, spaceAfter=6, fontName=_FONT_BOLD,
        ),
        "normal": ParagraphStyle(
            "Normal2", parent=base["Normal"], fontSize=9, leading=13,
        ),
        "uyari_k": ParagraphStyle(
            "UyariK", parent=base["Normal"],
            fontSize=9, leading=13,
            textColor=colors.HexColor(RENK_KIRMIZI), fontName=_FONT_BOLD,
        ),
        "uyari_s": ParagraphStyle(
            "UyariS", parent=base["Normal"],
            fontSize=9, leading=13,
            textColor=colors.HexColor(RENK_SARI), fontName=_FONT_BOLD,
        ),
        "kucuk": ParagraphStyle(
            "Kucuk", parent=base["Normal"],
            fontSize=7.5, textColor=colors.HexColor(RENK_GRI), alignment=TA_CENTER,
        ),
        "kpi_deger": ParagraphStyle(
            "KpiDeger", parent=base["Normal"],
            fontSize=18, fontName=_FONT_BOLD,
            textColor=colors.HexColor(RENK_LACIVERT), alignment=TA_CENTER,
        ),
        "kpi_etiket": ParagraphStyle(
            "KpiEtiket", parent=base["Normal"],
            fontSize=8, textColor=colors.HexColor(RENK_GRI), alignment=TA_CENTER,
        ),
    }


def _kpi_tablo(toplam_arac, toplam_sefer, toplam_kap, toplam_yolcu, doluluk):
    """4 kutucuklu KPI satırı."""
    st = _stiller()
    doluluk_renk = (RENK_YESIL if doluluk >= 80
                    else RENK_SARI if doluluk >= 60 else RENK_KIRMIZI)

    def kutu(deger, etiket, renk=RENK_LACIVERT):
        return [
            Paragraph(f'<font color="{renk}">{deger}</font>', st["kpi_deger"]),
            Paragraph(etiket, st["kpi_etiket"]),
        ]

    veri = [[
        kutu(str(toplam_arac),           "Aktif Araç"),
        kutu(str(toplam_sefer),          "Toplam Sefer"),
        kutu(f"{toplam_yolcu:,}",        "Toplam Yolcu"),
        kutu(f"%{doluluk:.1f}",          "Genel Doluluk", doluluk_renk),
    ]]
    tablo = Table(veri, colWidths=[4.2*cm] * 4)
    tablo.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#EEF4FB")),
        ("BOX",           (0, 0), (-1, -1), 0.8, colors.HexColor("#C8D8E8")),
        ("INNERGRID",     (0, 0), (-1, -1), 0.4, colors.HexColor("#C8D8E8")),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return tablo


def _veri_tablo(df_in: pd.DataFrame, firma_sutun: str) -> Table | None:
    """Tek bir kategori tablosu."""
    if df_in.empty:
        return None

    HEADER_BG  = colors.HexColor(RENK_LACIVERT)
    TOPLAM_BG  = colors.HexColor("#D5E8F7")
    SATIR_ALT  = [colors.white, colors.HexColor("#EEF4FB")]

    baslik = ["FİRMA", "AKTİF\nARAÇ", "TOPLAM\nSEFER", "EK\nSEFER",
              "KAPASİTE", "YOLCU", "DOLULUK\n%", "DURUM"]
    satirlar = [baslik]

    for _, row in df_in.iterrows():
        kap     = int(row["Toplam_Kapasite"])
        yolcu   = int(row["Toplam_Yolcu"])
        doluluk = yolcu / kap * 100 if kap > 0 else 0
        durum   = ("✔ İyi" if doluluk >= 80
                   else "⚠ Düşük" if doluluk >= 60 else "✖ Kritik")
        satirlar.append([
            str(row[firma_sutun])[:26],
            str(int(row["Farkli_Otobus"])),
            str(int(row["Toplam_Sefer"])),
            str(int(row["Ek_Sefer"])),
            str(kap),
            str(yolcu),
            f"%{doluluk:.1f}",
            durum,
        ])

    # Toplam satırı
    g_arac  = int(df_in["Farkli_Otobus"].sum())
    g_sefer = int(df_in["Toplam_Sefer"].sum())
    g_ek    = int(df_in["Ek_Sefer"].sum())
    g_kap   = int(df_in["Toplam_Kapasite"].sum())
    g_yolcu = int(df_in["Toplam_Yolcu"].sum())
    g_dol   = g_yolcu / g_kap * 100 if g_kap > 0 else 0
    satirlar.append(["GENEL TOPLAM", str(g_arac), str(g_sefer), str(g_ek),
                     str(g_kap), str(g_yolcu), f"%{g_dol:.1f}", ""])

    tablo = Table(satirlar, repeatRows=1,
                  colWidths=[4.8*cm, 1.6*cm, 1.8*cm, 1.5*cm, 1.8*cm, 1.7*cm, 1.8*cm, 2*cm])
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),   HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),   colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),   _FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0),   7.5),
        ("ALIGN",         (0, 0), (-1, -1),  "CENTER"),
        ("ALIGN",         (0, 1), (0, -1),   "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1),  "MIDDLE"),
        ("FONTNAME",      (0, 1), (-1, -2),  _FONT_NORMAL),
        ("FONTSIZE",      (0, 1), (-1, -2),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -2),  SATIR_ALT),
        ("BACKGROUND",    (0, -1), (-1, -1), TOPLAM_BG),
        ("FONTNAME",      (0, -1), (-1, -1), _FONT_BOLD),
        ("FONTSIZE",      (0, -1), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1),  0.35, colors.HexColor("#C0CDD8")),
        ("TOPPADDING",    (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING", (0, 0), (-1, -1),  4),
        ("LEFTPADDING",   (0, 1), (0, -1),   6),
    ]
    # Doluluk sütunu renklendirme (index 6, satır 1..n-1)
    for i, row in enumerate(satirlar[1:-1], start=1):
        try:
            pct = float(row[6].replace("%", ""))
            renk = (colors.HexColor("#D5F5E3") if pct >= 80
                    else colors.HexColor("#FEF9E7") if pct >= 60
                    else colors.HexColor("#FADBD8"))
            style.append(("BACKGROUND", (6, i), (6, i), renk))
        except Exception:
            pass

    tablo.setStyle(TableStyle(style))
    return tablo


def _uyari_tablo(uyarilar: list[dict]) -> Table | None:
    if not uyarilar:
        return None
    baslik = [["#", "KATEGORİ", "FİRMA", "UYARI DETAYI", "SEVİYE"]]
    satirlar = baslik[:]
    for i, u in enumerate(uyarilar, 1):
        seviye = "🔴 KRİTİK" if u["tip"] == "kirmizi" else "🟡 DİKKAT"
        satirlar.append([
            str(i), u["kategori"], u["firma"], u["mesaj"], seviye
        ])

    tablo = Table(satirlar, repeatRows=1,
                  colWidths=[0.8*cm, 2.2*cm, 4.5*cm, 7*cm, 2.5*cm])
    tablo.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),   colors.HexColor("#C0392B")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),   colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),   _FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, 0),   8),
        ("FONTNAME",      (0, 1), (-1, -1),  _FONT_NORMAL),
        ("FONTSIZE",      (0, 1), (-1, -1),  8),
        ("ALIGN",         (0, 0), (-1, -1),  "CENTER"),
        ("ALIGN",         (3, 1), (3, -1),   "LEFT"),
        ("ALIGN",         (2, 1), (2, -1),   "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1),  "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),  [colors.HexColor("#FEF9F9"), colors.HexColor("#FDF2F2")]),
        ("GRID",          (0, 0), (-1, -1),  0.35, colors.HexColor("#E8BBBB")),
        ("TOPPADDING",    (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING", (0, 0), (-1, -1),  4),
    ]))
    return tablo


def _haftalik_trend_tablo(gecmis: dict) -> Table | None:
    if len(gecmis) < 2:
        return None
    tarihler = list(gecmis.keys())
    baslik   = [["TARİH", "ARAÇ", "KAPASİTE", "YOLCU", "DOLULUK %", "DEĞİŞİM"]]
    satirlar = baslik[:]
    for i, tarih in enumerate(tarihler):
        v     = gecmis[tarih]
        onceki_dol = gecmis[tarihler[i - 1]]["doluluk"] if i > 0 else None
        degisim    = ""
        if onceki_dol is not None:
            fark     = v["doluluk"] - onceki_dol
            degisim  = f"{'▲' if fark > 0 else '▼' if fark < 0 else '—'} {abs(fark):.1f}%"
        satirlar.append([
            tarih, str(v["arac"]), str(v.get("kapasite", "-")),
            f"{v.get('yolcu', 0):,}", f"%{v['doluluk']:.1f}", degisim,
        ])

    tablo = Table(satirlar, repeatRows=1,
                  colWidths=[3.5*cm, 2.5*cm, 3*cm, 3*cm, 3*cm, 3*cm])
    tablo.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),   colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),   colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),   _FONT_BOLD),
        ("FONTSIZE",      (0, 0), (-1, -1),  8),
        ("ALIGN",         (0, 0), (-1, -1),  "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1),  "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),  [colors.white, colors.HexColor("#F0F4F8")]),
        ("GRID",          (0, 0), (-1, -1),  0.35, colors.HexColor("#C0C0C0")),
        ("TOPPADDING",    (0, 0), (-1, -1),  4),
        ("BOTTOMPADDING", (0, 0), (-1, -1),  4),
    ]))
    return tablo


# ───────────────────────────────────────────────
#  PDF RAPORU — GÜNLÜK
# ───────────────────────────────────────────────

def pdf_gunluk_olustur(tarih_str: str, sehir_df: pd.DataFrame, santiye_df: pd.DataFrame,
                        grafik_bytes: bytes, uyarilar: list[dict], gecmis: dict,
                        ml_tahmin: dict | None = None,
                        ml_oneriler: list[dict] | None = None,
                        ml_tahmin_grafik: bytes = b"",
                        ml_oneri_grafik: bytes = b"",
                        ml_anomaliler: list[dict] | None = None,
                        ml_anomali_grafik: bytes = b"") -> bytes:
    buf = io.BytesIO()
    doc = RaporSayfa(buf, tarih_str, rapor_turu="GÜNLÜK RAPOR",
                     pagesize=A4,
                     leftMargin=1.8*cm, rightMargin=1.8*cm,
                     topMargin=2*cm,    bottomMargin=1.5*cm)
    st = _stiller()
    h  = []

    # Başlık
    h.append(Spacer(1, 0.3*cm))
    h.append(Paragraph("GÜNLÜK ULAŞIM RAPORU", st["rapor_baslik"]))
    h.append(Paragraph(tarih_str, st["tarih"]))
    h.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(RENK_LACIVERT)))
    h.append(Spacer(1, 0.5*cm))

    # KPI kartları
    tum_dfs = [df for df in [sehir_df, santiye_df] if not df.empty]
    if tum_dfs:
        tum = pd.concat(tum_dfs)
        g_arac  = int(tum["Farkli_Otobus"].sum())
        g_sefer = int(tum["Toplam_Sefer"].sum())
        g_kap   = int(tum["Toplam_Kapasite"].sum())
        g_yolcu = int(tum["Toplam_Yolcu"].sum())
        g_dol   = g_yolcu / g_kap * 100 if g_kap > 0 else 0
        h.append(KeepTogether([
            Paragraph("Operasyon Özeti", st["bolum"]),
            _kpi_tablo(g_arac, g_sefer, g_kap, g_yolcu, g_dol),
        ]))
        h.append(Spacer(1, 0.5*cm))

    # Şehir Servisleri tablosu
    h.append(Paragraph("1. Şehir Servisleri", st["bolum"]))
    t1 = _veri_tablo(sehir_df, "OTOBUS FIRMASI")
    h.append(t1 if t1 else Paragraph("Veri bulunamadı.", st["normal"]))
    h.append(Spacer(1, 0.4*cm))

    # Şantiye Servisleri tablosu
    h.append(Paragraph("2. Şantiye Servisleri", st["bolum"]))
    t2 = _veri_tablo(santiye_df, "организация")
    h.append(t2 if t2 else Paragraph("Veri bulunamadı.", st["normal"]))
    h.append(Spacer(1, 0.4*cm))

    # Uyarılar bölümü
    if uyarilar:
        h.append(Paragraph(
            f"⚠ Dikkat Gerektiren Durumlar  ({len(uyarilar)} uyarı)", st["bolum"]
        ))
        ut = _uyari_tablo(uyarilar)
        if ut:
            h.append(ut)
        h.append(Spacer(1, 0.4*cm))

    # Grafik
    if grafik_bytes:
        h.append(PageBreak())
        h.append(Paragraph("Görsel Özet", st["bolum"]))
        img_buf = io.BytesIO(grafik_bytes)
        h.append(RLImage(img_buf, width=17*cm, height=max(7*cm, min(11*cm, len(sehir_df) * 0.6*cm + 4*cm))))
        h.append(Spacer(1, 0.5*cm))

    # Haftalık trend tablosu
    if len(gecmis) >= 2:
        h.append(Paragraph("Son 7 Günlük Trend", st["bolum"]))
        tt = _haftalik_trend_tablo(gecmis)
        if tt:
            h.append(tt)

    # ── ML Analiz Sayfası ──
    if ml_tahmin is not None or ml_oneriler or ml_anomaliler:
        _ml_pdf_bolumu(h, st,
                       ml_tahmin,
                       ml_oneriler or [],
                       ml_tahmin_grafik,
                       ml_oneri_grafik,
                       anomaliler=ml_anomaliler,
                       anomali_grafik=ml_anomali_grafik)

    doc.build(h)
    buf.seek(0)
    return buf.read()


# ───────────────────────────────────────────────
#  PDF RAPORU — HAFTALIK ÖZET
# ───────────────────────────────────────────────

def pdf_haftalik_olustur(gecmis: dict, trend_grafik: bytes) -> bytes:
    if len(gecmis) < 2:
        return b""

    tarih_aralik = f"{list(gecmis.keys())[0]}  →  {list(gecmis.keys())[-1]}"
    buf = io.BytesIO()
    doc = RaporSayfa(buf, tarih_aralik, rapor_turu="HAFTALIK ÖZET",
                     pagesize=A4,
                     leftMargin=1.8*cm, rightMargin=1.8*cm,
                     topMargin=2*cm,    bottomMargin=1.5*cm)
    st = _stiller()
    h  = []

    # Başlık
    h.append(Spacer(1, 0.3*cm))
    h.append(Paragraph("HAFTALIK ÖZET RAPORU", st["rapor_baslik"]))
    h.append(Paragraph(tarih_aralik, st["tarih"]))
    h.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(RENK_LACIVERT)))
    h.append(Spacer(1, 0.5*cm))

    # Haftalık KPI'lar (7 günün ortalamaları / toplamları)
    hafta_arac   = [v["arac"]    for v in gecmis.values()]
    hafta_yolcu  = [v["yolcu"]   for v in gecmis.values()]
    hafta_kap    = [v["kapasite"]for v in gecmis.values()]
    hafta_dol    = [v["doluluk"] for v in gecmis.values()]

    ort_arac   = round(sum(hafta_arac)  / len(hafta_arac),  1)
    top_yolcu  = sum(hafta_yolcu)
    ort_dol    = round(sum(hafta_dol)   / len(hafta_dol),   1)
    en_yuksek  = max(hafta_dol)
    en_dusuk   = min(hafta_dol)

    h.append(Paragraph("Haftalık Performans Özeti", st["bolum"]))
    ozet_veri = [
        ["GÖSTERGE",                  "DEĞER"],
        ["Günlük Ortalama Araç",      f"{ort_arac:.1f} araç"],
        ["Toplam Taşınan Yolcu",      f"{top_yolcu:,} kişi"],
        ["Ortalama Doluluk Oranı",    f"%{ort_dol:.1f}"],
        ["En Yüksek Günlük Doluluk",  f"%{en_yuksek:.1f}"],
        ["En Düşük Günlük Doluluk",   f"%{en_dusuk:.1f}"],
        ["Raporlanan Gün Sayısı",     str(len(gecmis))],
    ]
    oz_tablo = Table(ozet_veri, colWidths=[8*cm, 5*cm])
    oz_tablo.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),   colors.HexColor(RENK_LACIVERT)),
        ("TEXTCOLOR",     (0, 0), (-1, 0),   colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),   _FONT_BOLD),
        ("FONTNAME",      (0, 1), (0, -1),   _FONT_BOLD),
        ("FONTNAME",      (1, 1), (1, -1),   _FONT_NORMAL),
        ("FONTSIZE",      (0, 0), (-1, -1),  9),
        ("ALIGN",         (0, 0), (-1, -1),  "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1),  "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),  [colors.white, colors.HexColor("#EEF4FB")]),
        ("GRID",          (0, 0), (-1, -1),  0.35, colors.HexColor("#C0CDD8")),
        ("TOPPADDING",    (0, 0), (-1, -1),  6),
        ("BOTTOMPADDING", (0, 0), (-1, -1),  6),
        ("LEFTPADDING",   (0, 0), (-1, -1),  10),
    ]))
    h.append(oz_tablo)
    h.append(Spacer(1, 0.5*cm))

    # Günlük detay tablosu
    h.append(Paragraph("Günlük Veri Tablosu", st["bolum"]))
    tt = _haftalik_trend_tablo(gecmis)
    if tt:
        h.append(tt)
    h.append(Spacer(1, 0.5*cm))

    # Trend grafiği
    if trend_grafik:
        h.append(Paragraph("Haftalık Trend Grafiği", st["bolum"]))
        img_buf = io.BytesIO(trend_grafik)
        h.append(RLImage(img_buf, width=17*cm, height=8*cm))

    doc.build(h)
    buf.seek(0)
    return buf.read()


# ───────────────────────────────────────────────
#  TELEGRAM MESAJI  (gelişmiş)
# ───────────────────────────────────────────────

def telegram_gunluk_mesaj(tarih_str: str, sehir_df: pd.DataFrame,
                           santiye_df: pd.DataFrame, uyarilar: list[dict],
                           gecmis: dict) -> str:
    en_iyi, en_kotu = en_iyi_en_kotu(sehir_df, santiye_df)
    satir = "━━━━━━━━━━━━━━━━━━━━━\n"

    m  = f"🚌 *GÜNLÜK ULAŞIM RAPORU*\n"
    m += f"📅 _{tarih_str}_\n"
    m += satir

    # ── KPI Özeti ──
    tum_dfs = [df for df in [sehir_df, santiye_df] if not df.empty]
    if tum_dfs:
        tum     = pd.concat(tum_dfs)
        g_arac  = int(tum["Farkli_Otobus"].sum())
        g_sefer = int(tum["Toplam_Sefer"].sum())
        g_kap   = int(tum["Toplam_Kapasite"].sum())
        g_yolcu = int(tum["Toplam_Yolcu"].sum())
        g_dol   = g_yolcu / g_kap * 100 if g_kap > 0 else 0
        dol_emo = "🟢" if g_dol >= 80 else "🟡" if g_dol >= 60 else "🔴"

        m += f"📊 *OPERASYON ÖZETİ*\n"
        m += f"🚍 `{g_arac} araç  |  {g_sefer} sefer`\n"
        m += f"👥 `{g_yolcu:,} yolcu / {g_kap:,} koltuk`\n"
        m += f"{dol_emo} `Genel doluluk: %{g_dol:.1f}`\n"
        m += satir

    # ── Şehir Servisleri ──
    if not sehir_df.empty:
        m += "🏙️ *ŞEHİR SERVİSLERİ*\n"
        for _, row in sehir_df.iterrows():
            kap     = int(row["Toplam_Kapasite"])
            yolcu   = int(row["Toplam_Yolcu"])
            dol     = yolcu / kap * 100 if kap > 0 else 0
            emo     = "🟢" if dol >= 80 else "🟡" if dol >= 60 else "🔴"
            firma   = str(row["OTOBUS FIRMASI"])[:22]
            m += f"{emo} *{firma}*\n"
            m += f"   🚍 {int(row['Farkli_Otobus'])} araç  •  {int(row['Toplam_Sefer'])} sefer  •  %{dol:.1f}\n"
        m += satir

    # ── Şantiye Servisleri ──
    if not santiye_df.empty:
        m += "🏗️ *ŞANTİYE SERVİSLERİ*\n"
        for _, row in santiye_df.iterrows():
            kap     = int(row["Toplam_Kapasite"])
            yolcu   = int(row["Toplam_Yolcu"])
            dol     = yolcu / kap * 100 if kap > 0 else 0
            emo     = "🟢" if dol >= 80 else "🟡" if dol >= 60 else "🔴"
            firma   = str(row["организация"])[:22]
            m += f"{emo} *{firma}*\n"
            m += f"   🚍 {int(row['Farkli_Otobus'])} araç  •  {int(row['Toplam_Sefer'])} sefer  •  %{dol:.1f}\n"
        m += satir

    # ── En iyi / En kötü ──
    if en_iyi is not None and en_kotu is not None:
        m += "🏆 *PERFORMANS*\n"
        m += f"🥇 En iyi: *{str(en_iyi['Firma'])[:20]}*  (%{en_iyi['Doluluk']:.1f})\n"
        m += f"⚠️ En düşük: *{str(en_kotu['Firma'])[:20]}*  (%{en_kotu['Doluluk']:.1f})\n"
        m += satir

    # ── Uyarılar ──
    kirmizi = [u for u in uyarilar if u["tip"] == "kirmizi"]
    sarı    = [u for u in uyarilar if u["tip"] == "sari"]
    if uyarilar:
        m += f"🚨 *UYARILAR  ({len(kirmizi)} kritik / {len(sarı)} dikkat)*\n"
        for u in kirmizi:
            m += f"🔴 [{u['kategori']}] *{u['firma']}*: {u['mesaj']}\n"
        for u in sarı:
            m += f"🟡 [{u['kategori']}] *{u['firma']}*: {u['mesaj']}\n"
        m += satir

    # ── Haftalık trend kısa özeti ──
    if len(gecmis) >= 2:
        tarihler = list(gecmis.keys())
        onceki   = gecmis[tarihler[-2]]["doluluk"]
        bugun    = gecmis[tarihler[-1]]["doluluk"]
        fark     = bugun - onceki
        ok       = "📈" if fark > 0 else "📉" if fark < 0 else "➡️"
        m += f"{ok} Dünkü fark: `{'+' if fark >= 0 else ''}{fark:.1f}%`\n"

    m += "\n📄 _Detaylı PDF ve grafikler aşağıda gönderilecektir._"
    return m


def telegram_haftalik_mesaj(gecmis: dict) -> str:
    tarihler   = list(gecmis.keys())
    hafta_dol  = [v["doluluk"] for v in gecmis.values()]
    hafta_arac = [v["arac"]    for v in gecmis.values()]
    hafta_yol  = [v["yolcu"]   for v in gecmis.values()]

    ort_dol   = sum(hafta_dol)  / len(hafta_dol)
    ort_arac  = sum(hafta_arac) / len(hafta_arac)
    top_yolcu = sum(hafta_yol)

    en_y_gun  = tarihler[hafta_dol.index(max(hafta_dol))]
    en_d_gun  = tarihler[hafta_dol.index(min(hafta_dol))]

    dol_emo = "🟢" if ort_dol >= 80 else "🟡" if ort_dol >= 60 else "🔴"

    m  = "📅 *HAFTALIK ÖZET RAPORU*\n"
    m += f"🗓️ _{tarihler[0]}  →  {tarihler[-1]}_\n"
    m += "━━━━━━━━━━━━━━━━━━━━━\n"
    m += f"{dol_emo} *Ortalama Doluluk:* %{ort_dol:.1f}\n"
    m += f"🚍 *Ortalama Araç:* {ort_arac:.1f} araç/gün\n"
    m += f"👥 *Toplam Yolcu:* {top_yolcu:,} kişi\n"
    m += "━━━━━━━━━━━━━━━━━━━━━\n"
    m += f"🏆 En verimli gün: *{en_y_gun}*  (%{max(hafta_dol):.1f})\n"
    m += f"⚠️ En düşük gün: *{en_d_gun}*  (%{min(hafta_dol):.1f})\n"
    m += "━━━━━━━━━━━━━━━━━━━━━\n"
    m += "\n📄 _Haftalık PDF raporu aşağıda gönderilecektir._"
    return m


# ═══════════════════════════════════════════════
#  FİRMA BAZLI DETAY RAPORU
# ═══════════════════════════════════════════════

def _firma_trend_df(firma_gecmis: dict, firma_adi: str) -> pd.DataFrame:
    """Belirli bir firmanın günlük verisini DataFrame olarak döndürür."""
    rows = []
    for tarih in sorted(firma_gecmis.keys()):
        firmalar = firma_gecmis[tarih]
        if firma_adi in firmalar:
            v = firmalar[firma_adi]
            rows.append({"tarih": tarih, **v})
    return pd.DataFrame(rows)


def grafik_firma_detay(firma_gecmis: dict, firma_adi: str) -> bytes:
    """Tek firma için doluluk + araç trendi grafiği."""
    fdf = _firma_trend_df(firma_gecmis, firma_adi)
    if fdf.empty or len(fdf) < 2:
        return b""

    etiketler = [t[5:].replace("-", "/") for t in fdf["tarih"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    fig.suptitle(f"Firma Detay — {firma_adi}", fontsize=13, fontweight="bold",
                 color=RENK_LACIVERT)

    # Sol: Doluluk trendi
    dol = fdf["doluluk"].values
    dol_renk = [RENK_YESIL if d >= 80 else RENK_SARI if d >= 60 else RENK_KIRMIZI for d in dol]
    ax1.bar(etiketler, dol, color=dol_renk, edgecolor="white", width=0.6)
    ax1.axhline(80, color=RENK_GRI, linestyle="--", linewidth=1, alpha=0.7, label="Hedef %80")
    ax1.set_ylabel("Doluluk (%)", fontsize=9)
    ax1.set_ylim(0, 110)
    ax1.set_title("Doluluk Trendi", fontsize=11, fontweight="bold", color=RENK_LACIVERT)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(dol):
        ax1.text(i, v + 1.5, f"%{v:.0f}", ha="center", fontsize=8, fontweight="bold")

    # Sağ: Araç + Yolcu
    ax2.set_facecolor(RENK_ARK)
    ax2.bar(etiketler, fdf["yolcu"].values, color=RENK_ACIK_MAVİ, alpha=0.7,
            label="Yolcu", width=0.6)
    ax2b = ax2.twinx()
    ax2b.plot(etiketler, fdf["arac"].values, color=RENK_TURUNCU, marker="o",
              linewidth=2, markersize=6, label="Araç")
    ax2.set_ylabel("Yolcu", fontsize=9, color=RENK_ACIK_MAVİ)
    ax2b.set_ylabel("Araç", fontsize=9, color=RENK_TURUNCU)
    ax2.set_title("Yolcu & Araç", fontsize=11, fontweight="bold", color=RENK_LACIVERT)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax2.spines["top"].set_visible(False)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    return _grafik_kapat_bytes()


def _firma_verimlilik(sehir_df: pd.DataFrame, santiye_df: pd.DataFrame) -> list[dict]:
    """Her firma için verimlilik metrikleri hesaplar."""
    sonuc = []
    def _isle(df, firma_col, kategori):
        if df.empty:
            return
        for _, row in df.iterrows():
            firma = str(row[firma_col])[:30]
            arac = int(row["Farkli_Otobus"])
            sefer = int(row["Toplam_Sefer"])
            kap = int(row["Toplam_Kapasite"])
            yolcu = int(row["Toplam_Yolcu"])
            doluluk = yolcu / kap * 100 if kap > 0 else 0
            sefer_per_arac = sefer / arac if arac > 0 else 0
            yolcu_per_sefer = yolcu / sefer if sefer > 0 else 0
            yolcu_per_arac = yolcu / arac if arac > 0 else 0
            bos_koltuk = kap - yolcu
            verimlilik_skoru = min(100, doluluk * 0.5 + min(sefer_per_arac, 3) / 3 * 30
                                  + min(yolcu_per_arac, 100) / 100 * 20)
            sonuc.append({
                "firma": firma, "kategori": kategori,
                "arac": arac, "sefer": sefer, "kapasite": kap,
                "yolcu": yolcu, "doluluk": round(doluluk, 1),
                "sefer_per_arac": round(sefer_per_arac, 2),
                "yolcu_per_sefer": round(yolcu_per_sefer, 1),
                "yolcu_per_arac": round(yolcu_per_arac, 1),
                "bos_koltuk": bos_koltuk,
                "verimlilik": round(verimlilik_skoru, 1),
            })
    _isle(sehir_df, "OTOBUS FIRMASI", "Şehir")
    _isle(santiye_df, "организация", "Şantiye")
    sonuc.sort(key=lambda x: x["verimlilik"], reverse=True)
    return sonuc


def grafik_verimlilik(verimlilik: list[dict]) -> bytes:
    """Firma verimlilik karşılaştırma grafiği."""
    if not verimlilik:
        return b""

    firmalar = [v["firma"][:18] for v in verimlilik]
    skorlar = [v["verimlilik"] for v in verimlilik]
    renkler = [RENK_YESIL if s >= 70 else RENK_SARI if s >= 50 else RENK_KIRMIZI for s in skorlar]

    fig, ax = plt.subplots(figsize=(12, max(4, len(firmalar) * 0.7 + 2)), facecolor="white")
    bars = ax.barh(firmalar, skorlar, color=renkler, edgecolor="white", height=0.55)
    ax.axvline(70, color=RENK_GRI, linestyle="--", linewidth=1.2, label="Hedef %70")
    for bar, v in zip(bars, skorlar):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}", va="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("Verimlilik Skoru", fontsize=10)
    ax.set_title("Firma Verimlilik Karşılaştırması", fontsize=12,
                 fontweight="bold", color=RENK_LACIVERT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return _grafik_kapat_bytes()


def _verimlilik_pdf_tablo(verimlilik: list[dict]) -> Table | None:
    if not verimlilik:
        return None
    baslik = [["SIRA", "FİRMA", "KAT.", "ARAÇ", "SEFER",
               "SEFER/\nARAÇ", "YOLCU/\nSEFER", "DOLULUK\n%",
               "BOŞ\nKOLTUK", "VERİMLİLİK\nSKORU"]]
    satirlar = baslik[:]
    for i, v in enumerate(verimlilik, 1):
        satirlar.append([
            str(i), v["firma"][:22], v["kategori"],
            str(v["arac"]), str(v["sefer"]),
            f"{v['sefer_per_arac']:.1f}", f"{v['yolcu_per_sefer']:.0f}",
            f"%{v['doluluk']:.1f}", str(v["bos_koltuk"]),
            f"{v['verimlilik']:.0f}",
        ])
    t = Table(satirlar, repeatRows=1,
              colWidths=[1*cm, 3.5*cm, 1.5*cm, 1.3*cm, 1.3*cm,
                         1.5*cm, 1.5*cm, 1.8*cm, 1.5*cm, 2.1*cm])
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  _FONT_BOLD),
        ("FONTNAME",      (0, 1), (-1, -1), _FONT_NORMAL),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",         (1, 1), (1, -1),  "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F4F8")]),
        ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#C0CDD8")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    # Verimlilik sütunu renklendirme
    for i, v in enumerate(verimlilik, 1):
        bg = (colors.HexColor("#D5F5E3") if v["verimlilik"] >= 70
              else colors.HexColor("#FEF9E7") if v["verimlilik"] >= 50
              else colors.HexColor("#FADBD8"))
        style.append(("BACKGROUND", (9, i), (9, i), bg))
    t.setStyle(TableStyle(style))
    return t


def telegram_firma_mesaj(verimlilik: list[dict], firma_gecmis: dict) -> str:
    """Firma bazlı Telegram mesajı."""
    satir = "━━━━━━━━━━━━━━━━━━━━━\n"
    m = f"🏢 *FİRMA DETAY RAPORU*\n{satir}"

    if not verimlilik:
        m += "ℹ Firma verisi bulunmuyor.\n"
        return m

    m += f"📊 *VERİMLİLİK SIRALAMASI*  ({len(verimlilik)} firma)\n\n"
    for i, v in enumerate(verimlilik[:10], 1):
        emo = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        skor_emo = "🟢" if v["verimlilik"] >= 70 else "🟡" if v["verimlilik"] >= 50 else "🔴"
        m += f"{emo} {skor_emo} *{v['firma']}* [{v['kategori']}]\n"
        m += (f"   Skor: *{v['verimlilik']:.0f}* | "
              f"%{v['doluluk']:.1f} dol. | "
              f"{v['sefer_per_arac']:.1f} sef/araç | "
              f"{v['bos_koltuk']} boş koltuk\n")
    m += satir

    # Firma trend kısa özet
    firma_sayilari = {}
    for tarih, firmalar in sorted(firma_gecmis.items()):
        firma_sayilari[tarih] = len(firmalar)
    if len(firma_sayilari) >= 2:
        tarihler = list(firma_sayilari.keys())
        m += f"📅 _Takip edilen dönem: {tarihler[0]} → {tarihler[-1]}_\n"

    m += "\n📄 _Firma detay PDF'i aşağıda gönderilecektir._"
    return m


def pdf_firma_detay_olustur(tarih_str: str, sehir_df: pd.DataFrame,
                             santiye_df: pd.DataFrame,
                             verimlilik: list[dict],
                             verimlilik_grafik: bytes,
                             firma_gecmis: dict) -> bytes:
    """Firma bazlı detay PDF raporu."""
    buf = io.BytesIO()
    doc = RaporSayfa(buf, tarih_str, rapor_turu="FİRMA DETAY",
                     pagesize=A4,
                     leftMargin=1.5*cm, rightMargin=1.5*cm,
                     topMargin=2*cm, bottomMargin=1.5*cm)
    st = _stiller()
    h = []

    h.append(Spacer(1, 0.3*cm))
    h.append(Paragraph("FİRMA DETAY RAPORU", st["rapor_baslik"]))
    h.append(Paragraph(tarih_str, st["tarih"]))
    h.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(RENK_LACIVERT)))
    h.append(Spacer(1, 0.5*cm))

    # Verimlilik tablosu
    h.append(Paragraph("Firma Verimlilik Sıralaması", st["bolum"]))
    h.append(Paragraph(
        "Verimlilik skoru: Doluluk (%50 ağırlık) + Sefer/Araç oranı (%30) + "
        "Yolcu/Araç oranı (%20) bileşiminden hesaplanır.",
        st["normal"]
    ))
    h.append(Spacer(1, 0.3*cm))
    vt = _verimlilik_pdf_tablo(verimlilik)
    if vt:
        h.append(vt)
    h.append(Spacer(1, 0.4*cm))

    # Verimlilik grafiği
    if verimlilik_grafik:
        h.append(RLImage(io.BytesIO(verimlilik_grafik), width=17*cm,
                         height=max(5*cm, len(verimlilik) * 0.9*cm + 2*cm)))

    # Her firma için detay sayfası (ilk 5 firma)
    for v in verimlilik[:5]:
        firma = v["firma"]
        fdf = _firma_trend_df(firma_gecmis, firma)
        if fdf.empty or len(fdf) < 2:
            continue

        h.append(PageBreak())
        h.append(Paragraph(f"Firma Detay — {firma}", st["bolum"]))
        h.append(Paragraph(f"Kategori: {v['kategori']}  |  Güncel skor: {v['verimlilik']:.0f}",
                           st["normal"]))
        h.append(Spacer(1, 0.3*cm))

        # Firma trend tablosu
        baslik_f = [["TARİH", "ARAÇ", "SEFER", "KAPASİTE", "YOLCU", "DOLULUK %"]]
        satirlar_f = baslik_f[:]
        for _, row in fdf.iterrows():
            satirlar_f.append([
                row["tarih"], str(row["arac"]), str(row["sefer"]),
                str(row["kapasite"]), str(row["yolcu"]), f"%{row['doluluk']:.1f}",
            ])
        ft = Table(satirlar_f, repeatRows=1,
                   colWidths=[3.5*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor(RENK_LACIVERT)),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), _FONT_BOLD),
            ("FONTNAME",      (0, 1), (-1, -1), _FONT_NORMAL),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF4FB")]),
            ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#C0CDD8")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        h.append(ft)
        h.append(Spacer(1, 0.4*cm))

        # Firma grafiği
        fg = grafik_firma_detay(firma_gecmis, firma)
        if fg:
            h.append(RLImage(io.BytesIO(fg), width=17*cm, height=7*cm))

    doc.build(h)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════
#  AYLIK ÖZET RAPORU
# ═══════════════════════════════════════════════

def _aylik_veri(gecmis: dict, ay: int = None, yil: int = None) -> dict:
    """Belirli ay/yıl için geçmiş verisini filtreler."""
    bugun = datetime.now()
    if ay is None:
        # Geçen ay
        gecen = bugun.replace(day=1) - timedelta(days=1)
        ay = gecen.month
        yil = gecen.year
    if yil is None:
        yil = bugun.year

    ay_str = f"{yil}-{ay:02d}"
    return {k: v for k, v in gecmis.items() if k.startswith(ay_str)}


def grafik_aylik_ozet(aylik_gecmis: dict, ay_adi: str) -> bytes:
    """Aylık doluluk + yolcu trend grafiği."""
    if len(aylik_gecmis) < 2:
        return b""

    tarihler = list(aylik_gecmis.keys())
    doluluklar = [aylik_gecmis[t]["doluluk"] for t in tarihler]
    yolcular = [aylik_gecmis[t]["yolcu"] for t in tarihler]
    araclar = [aylik_gecmis[t]["arac"] for t in tarihler]
    etiketler = [t[8:] for t in tarihler]  # sadece gün

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
    fig.suptitle(f"Aylık Özet — {ay_adi}", fontsize=14, fontweight="bold",
                 color=RENK_LACIVERT, y=0.98)

    # Sol üst: Doluluk trendi
    ax = axes[0, 0]
    dol_renk = [RENK_YESIL if d >= 80 else RENK_SARI if d >= 60 else RENK_KIRMIZI
                for d in doluluklar]
    ax.bar(etiketler, doluluklar, color=dol_renk, edgecolor="white", width=0.7)
    ax.axhline(80, color=RENK_GRI, linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylabel("Doluluk (%)")
    ax.set_title("Günlük Doluluk", fontsize=11, fontweight="bold", color=RENK_LACIVERT)
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Sağ üst: Yolcu trendi
    ax = axes[0, 1]
    ax.fill_between(etiketler, yolcular, alpha=0.15, color=RENK_ACIK_MAVİ)
    ax.plot(etiketler, yolcular, color=RENK_ACIK_MAVİ, marker="o", linewidth=2, markersize=5)
    ax.set_ylabel("Yolcu")
    ax.set_title("Günlük Yolcu", fontsize=11, fontweight="bold", color=RENK_LACIVERT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Sol alt: Araç trendi
    ax = axes[1, 0]
    ax.bar(etiketler, araclar, color=RENK_TURUNCU, alpha=0.7, edgecolor="white", width=0.7)
    ax.set_ylabel("Araç")
    ax.set_title("Günlük Araç", fontsize=11, fontweight="bold", color=RENK_LACIVERT)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Sağ alt: Haftalık ortalama doluluk
    ax = axes[1, 1]
    import numpy as _np
    hafta_ort = []
    hafta_etiket = []
    for i in range(0, len(doluluklar), 7):
        dilim = doluluklar[i:i + 7]
        hafta_ort.append(_np.mean(dilim))
        hafta_etiket.append(f"H{len(hafta_etiket)+1}")
    h_renk = [RENK_YESIL if d >= 80 else RENK_SARI if d >= 60 else RENK_KIRMIZI
              for d in hafta_ort]
    ax.bar(hafta_etiket, hafta_ort, color=h_renk, edgecolor="white", width=0.5)
    ax.axhline(80, color=RENK_GRI, linestyle="--", linewidth=1)
    ax.set_ylabel("Ort. Doluluk (%)")
    ax.set_title("Haftalık Ortalama", fontsize=11, fontweight="bold", color=RENK_LACIVERT)
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(hafta_ort):
        ax.text(i, v + 1.5, f"%{v:.0f}", ha="center", fontsize=9, fontweight="bold")

    plt.xticks(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return _grafik_kapat_bytes()


def telegram_aylik_mesaj(aylik_gecmis: dict, ay_adi: str,
                          firma_gecmis: dict = None) -> str:
    """Aylık özet Telegram mesajı."""
    if not aylik_gecmis:
        return f"📅 *AYLIK ÖZET — {ay_adi}*\n⚠ Veri bulunamadı."

    satir = "━━━━━━━━━━━━━━━━━━━━━\n"
    tarihler = list(aylik_gecmis.keys())
    dol = [v["doluluk"] for v in aylik_gecmis.values()]
    arac = [v["arac"] for v in aylik_gecmis.values()]
    yolcu = [v["yolcu"] for v in aylik_gecmis.values()]
    kap = [v["kapasite"] for v in aylik_gecmis.values()]

    ort_dol = sum(dol) / len(dol)
    top_yolcu = sum(yolcu)
    top_kap = sum(kap)
    ort_arac = sum(arac) / len(arac)
    bos_koltuk = top_kap - top_yolcu
    dol_emo = "🟢" if ort_dol >= 80 else "🟡" if ort_dol >= 60 else "🔴"

    m = f"📅 *AYLIK ÖZET RAPORU — {ay_adi}*\n"
    m += f"🗓️ _{tarihler[0]} → {tarihler[-1]}_\n"
    m += satir

    m += f"{dol_emo} *Ortalama Doluluk:* %{ort_dol:.1f}\n"
    m += f"🚍 *Ort. Araç/Gün:* {ort_arac:.1f}\n"
    m += f"👥 *Toplam Yolcu:* {top_yolcu:,}\n"
    m += f"💺 *Toplam Kapasite:* {top_kap:,}\n"
    m += f"🪑 *Boş Koltuk:* {bos_koltuk:,} ({bos_koltuk/top_kap*100:.1f}%)\n"
    m += f"📊 *Raporlanan Gün:* {len(tarihler)}\n"
    m += satir

    m += f"🏆 *En İyi Gün:* {tarihler[dol.index(max(dol))]}  (%{max(dol):.1f})\n"
    m += f"⚠️ *En Kötü Gün:* {tarihler[dol.index(min(dol))]}  (%{min(dol):.1f})\n"
    m += f"📈 *En Yüksek Yolcu:* {max(yolcu):,} ({tarihler[yolcu.index(max(yolcu))]})\n"
    m += satir

    # Haftalık ortalamalar
    m += "📊 *HAFTALIK ORTALAMALAR*\n"
    for i in range(0, len(dol), 7):
        dilim = dol[i:i + 7]
        if dilim:
            ort = sum(dilim) / len(dilim)
            emo = "🟢" if ort >= 80 else "🟡" if ort >= 60 else "🔴"
            m += f"{emo} Hafta {i//7+1}: %{ort:.1f} ({len(dilim)} gün)\n"
    m += satir

    # En iyi/kötü firmalar (firma geçmişinden)
    if firma_gecmis:
        son_tarih = sorted(firma_gecmis.keys())[-1] if firma_gecmis else None
        if son_tarih and son_tarih in firma_gecmis:
            firmalar = firma_gecmis[son_tarih]
            if firmalar:
                firma_dol = [(f, v.get("doluluk", 0)) for f, v in firmalar.items()]
                firma_dol.sort(key=lambda x: x[1], reverse=True)
                m += "🏢 *FİRMA PERFORMANSI (son gün)*\n"
                for f, d in firma_dol[:5]:
                    emo = "🟢" if d >= 80 else "🟡" if d >= 60 else "🔴"
                    m += f"{emo} {f[:20]}: %{d:.1f}\n"
                m += satir

    m += "\n📄 _Aylık PDF raporu aşağıda gönderilecektir._"
    return m


def pdf_aylik_olustur(aylik_gecmis: dict, ay_adi: str,
                       aylik_grafik: bytes,
                       firma_gecmis: dict = None) -> bytes:
    """Aylık özet PDF raporu."""
    if len(aylik_gecmis) < 2:
        return b""

    tarih_aralik = f"{list(aylik_gecmis.keys())[0]}  →  {list(aylik_gecmis.keys())[-1]}"
    buf = io.BytesIO()
    doc = RaporSayfa(buf, tarih_aralik, rapor_turu=f"AYLIK ÖZET — {ay_adi}",
                     pagesize=A4,
                     leftMargin=1.8*cm, rightMargin=1.8*cm,
                     topMargin=2*cm, bottomMargin=1.5*cm)
    st = _stiller()
    h = []

    h.append(Spacer(1, 0.3*cm))
    h.append(Paragraph(f"AYLIK ÖZET RAPORU — {ay_adi}", st["rapor_baslik"]))
    h.append(Paragraph(tarih_aralik, st["tarih"]))
    h.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor(RENK_LACIVERT)))
    h.append(Spacer(1, 0.5*cm))

    # Aylık KPI
    dol = [v["doluluk"] for v in aylik_gecmis.values()]
    arac = [v["arac"] for v in aylik_gecmis.values()]
    yolcu_l = [v["yolcu"] for v in aylik_gecmis.values()]
    kap_l = [v["kapasite"] for v in aylik_gecmis.values()]

    ort_dol = sum(dol) / len(dol)
    top_yolcu = sum(yolcu_l)
    top_kap = sum(kap_l)
    bos = top_kap - top_yolcu

    h.append(Paragraph("Aylık Performans Özeti", st["bolum"]))
    ozet = [
        ["GÖSTERGE", "DEĞER"],
        ["Raporlanan Gün Sayısı", str(len(aylik_gecmis))],
        ["Ortalama Doluluk", f"%{ort_dol:.1f}"],
        ["Günlük Ortalama Araç", f"{sum(arac)/len(arac):.1f}"],
        ["Toplam Taşınan Yolcu", f"{top_yolcu:,}"],
        ["Toplam Kapasite", f"{top_kap:,}"],
        ["Boş Koltuk (toplam)", f"{bos:,} (%{bos/top_kap*100:.1f})"],
        ["En Yüksek Doluluk", f"%{max(dol):.1f}"],
        ["En Düşük Doluluk", f"%{min(dol):.1f}"],
        ["Std Sapma (Doluluk)", f"±{(sum((d-ort_dol)**2 for d in dol)/len(dol))**0.5:.1f}"],
    ]
    ot = Table(ozet, colWidths=[8*cm, 5*cm])
    ot.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor(RENK_LACIVERT)),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), _FONT_BOLD),
        ("FONTNAME",      (0, 1), (0, -1), _FONT_BOLD),
        ("FONTNAME",      (1, 1), (1, -1), _FONT_NORMAL),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF4FB")]),
        ("GRID",          (0, 0), (-1, -1), 0.35, colors.HexColor("#C0CDD8")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    h.append(ot)
    h.append(Spacer(1, 0.5*cm))

    # Günlük detay tablosu
    h.append(Paragraph("Günlük Veri Tablosu", st["bolum"]))
    tt = _haftalik_trend_tablo(aylik_gecmis)
    if tt:
        h.append(tt)
    h.append(Spacer(1, 0.5*cm))

    # Aylık grafik
    if aylik_grafik:
        h.append(PageBreak())
        h.append(Paragraph("Aylık Trend Grafikleri", st["bolum"]))
        h.append(RLImage(io.BytesIO(aylik_grafik), width=17*cm, height=12*cm))

    doc.build(h)
    buf.seek(0)
    return buf.read()


# ───────────────────────────────────────────────
#  MAİL & VERİ OKUMA (gelişmiş — çoklu format)
# ───────────────────────────────────────────────

# ── Yardımcı: Sütun eşleme ───────────────────────────────

def _sutun_eslestir(df: pd.DataFrame, alias_harita: dict) -> pd.DataFrame:
    """
    DataFrame sütun isimlerini standart isimlere çevirir.
    alias_harita: { standart_isim: [olası_isimler] }
    Eşleşmeyen sütunlar olduğu gibi kalır.
    """
    yeni_isimler = {}
    mevcut = set(df.columns)
    for standart, adaylar in alias_harita.items():
        if standart in mevcut:
            continue  # zaten doğru isimde
        for aday in adaylar:
            if aday in mevcut:
                yeni_isimler[aday] = standart
                break
    if yeni_isimler:
        log.info(f"  🔄 Sütun eşleme: {yeni_isimler}")
        df = df.rename(columns=yeni_isimler)
    return df


def _sayfa_bul(dosya_yolu: str, aday_isimler: list[str]) -> str | None:
    """Excel dosyasında aday sayfa isimlerini sırayla dener, ilk eşleşeni döndürür."""
    try:
        xl = pd.ExcelFile(dosya_yolu)
        mevcut_sayfalar = xl.sheet_names
        xl.close()
    except Exception:
        return None

    for aday in aday_isimler:
        if aday in mevcut_sayfalar:
            return aday
    # Fuzzy: büyük/küçük harf duyarsız eşleşme
    for aday in aday_isimler:
        for mevcut in mevcut_sayfalar:
            if aday.lower() == mevcut.lower():
                return mevcut
    # Kısmi eşleşme: aday mevcut sayfanın içinde geçiyor mu?
    for aday in aday_isimler:
        for mevcut in mevcut_sayfalar:
            if aday.lower() in mevcut.lower():
                return mevcut
    log.warning(f"  ⚠ Sayfa bulunamadı. Mevcut sayfalar: {mevcut_sayfalar}")
    return None


def _tarih_formatla(dt: datetime, formatlar: list[str] | None = None) -> list[str]:
    """Bir datetime'ı tüm desteklenen formatlarda string listesine çevirir."""
    if formatlar is None:
        formatlar = TARIH_FORMATLARI
    sonuclar = []
    for fmt in formatlar:
        try:
            sonuclar.append(dt.strftime(fmt))
        except Exception:
            pass
    return sonuclar


def _dosya_oku(dosya_yolu: str, sayfa: str = None, header: int = 0) -> pd.DataFrame:
    """Excel veya CSV dosyasını okur."""
    uzanti = os.path.splitext(dosya_yolu)[1].lower()
    if uzanti == ".csv":
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1254", "cp1251"]:
            try:
                df = pd.read_csv(dosya_yolu, encoding=enc, header=header)
                log.info(f"  📄 CSV okundu ({enc}): {dosya_yolu}")
                return df
            except (UnicodeDecodeError, UnicodeError):
                continue
        log.error(f"  ❌ CSV encoding hatası: {dosya_yolu}")
        return pd.DataFrame()
    else:
        # Excel
        df = pd.read_excel(dosya_yolu, sheet_name=sayfa, header=header)
        return df


def _dosya_pattern_eslesir(dosya_adi: str, patternler: list[str]) -> bool:
    """Dosya adının pattern listesindeki herhangi biriyle eşleşip eşleşmediğini kontrol eder."""
    dosya_upper = dosya_adi.upper()
    return any(p.upper() in dosya_upper for p in patternler)


# ── Mail & Veri Okuma ─────────────────────────────────────

def maillerden_ekleri_indir() -> list:
    log.info("📥 Gmail'e bağlanılıyor...")
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(GMAIL_USER, GMAIL_PASS)
        mail.select("inbox")
        tarih = (datetime.now() - timedelta(days=MAIL_LOOKBACK_GUN)).strftime("%d-%b-%Y")
        _, messages = mail.search(None, f'(SINCE "{tarih}")')
        mail_id_list = messages[0].split()
        log.info(f"📧 {len(mail_id_list)} mail taranıyor (son {MAIL_LOOKBACK_GUN} gün)...")

        indirilenler = []
        atlanan_gondericiler = set()

        for num in mail_id_list:
            _, data = mail.fetch(num, "(RFC822)")
            msg     = email.message_from_bytes(data[0][1])
            gonderen = str(msg.get("From"))

            if not any(isim in gonderen for isim in GONDERICILER):
                atlanan_gondericiler.add(gonderen[:40])
                continue

            for part in msg.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                if part.get("Content-Disposition") is None:
                    continue
                dosya_adi = part.get_filename()
                if dosya_adi:
                    decoded, charset = email.header.decode_header(dosya_adi)[0]
                    if isinstance(decoded, bytes):
                        dosya_adi = decoded.decode(charset or "utf-8")
                    # Desteklenen uzantı kontrolü
                    uzanti = os.path.splitext(dosya_adi)[1].lower()
                    if uzanti in DESTEKLENEN_UZANTILAR:
                        path = os.path.join(os.getcwd(), dosya_adi)
                        with open(path, "wb") as f:
                            f.write(part.get_payload(decode=True))
                        indirilenler.append(dosya_adi)
                        log.info(f"  ✅ İndirildi: {dosya_adi} (gönderen: {gonderen[:30]})")
                    else:
                        log.debug(f"  ⏭ Atlandı (uzantı {uzanti}): {dosya_adi}")

        if atlanan_gondericiler:
            log.debug(f"  ℹ {len(atlanan_gondericiler)} gönderici eşleşmedi")

        mail.logout()
        log.info(f"📥 Toplam {len(indirilenler)} dosya indirildi.")
        return indirilenler
    except Exception as e:
        log.error(f"❌ Mail hatası: {e}")
        return []


def verileri_isle(dosyalar: list, dun_ali: str, dun_ruslan: str):
    sehir_df = santiye_df = pd.DataFrame()
    dun_dt = datetime.now() - timedelta(1)
    tum_tarih_strler = _tarih_formatla(dun_dt)
    islem_ozeti = {"sehir_denenen": 0, "sehir_basarili": False,
                   "santiye_denenen": 0, "santiye_basarili": False}

    # ── Şehir Servisleri ──────────────────────────
    sehir_dosyalar = [f for f in dosyalar if _dosya_pattern_eslesir(f, SEHIR_DOSYA_PATTERN)]
    # Pattern'e uymayan dosyaları da CSV ise dene
    csv_dosyalar = [f for f in dosyalar if f.lower().endswith(".csv")
                    and f not in sehir_dosyalar]

    for dosya in sehir_dosyalar + csv_dosyalar:
        islem_ozeti["sehir_denenen"] += 1
        try:
            uzanti = os.path.splitext(dosya)[1].lower()

            if uzanti == ".csv":
                df = _dosya_oku(dosya)
            else:
                sayfa = _sayfa_bul(dosya, SEHIR_SAYFA_ADLARI)
                if sayfa is None:
                    log.warning(f"  ⚠ Şehir sayfası bulunamadı: {dosya}")
                    continue
                df = _dosya_oku(dosya, sayfa=sayfa)
                log.info(f"  📑 Sayfa '{sayfa}' okundu: {dosya}")

            df.columns = df.columns.str.strip()
            df = _sutun_eslestir(df, SEHIR_SUTUN_ALIAS)

            # Tarih sütunu kontrolü
            if "TARIH" in df.columns:
                df["TARIH"] = df["TARIH"].ffill()
                # Tüm tarih formatlarıyla eşleşmeyi dene
                df_dun = pd.DataFrame()
                for tarih_str in tum_tarih_strler:
                    df_filtre = df[df["TARIH"].astype(str).str.contains(tarih_str, na=False)]
                    if not df_filtre.empty:
                        df_dun = df_filtre
                        log.info(f"  📅 Tarih eşleşti ('{tarih_str}'): {len(df_dun)} satır")
                        break
                if df_dun.empty:
                    log.warning(f"  ⚠ Tarih eşleşmedi: {dosya} "
                                f"(aranan: {tum_tarih_strler[:3]}...)")
                    continue
            else:
                # Tarih sütunu yok — tüm veriyi kullan
                log.info(f"  ℹ Tarih sütunu yok, tüm veri kullanılıyor: {dosya}")
                df_dun = df

            if df_dun.empty:
                continue

            sehir_df = df_dun.groupby("OTOBUS FIRMASI").agg(
                Farkli_Otobus=("OTOBUS PLAKASI", "nunique"),
                Toplam_Sefer=("OTOBUS PLAKASI", "count"),
                Toplam_Kapasite=("OTOBUS KAPASITESI", "sum"),
                Toplam_Yolcu=("TASIMA KAPASITESI", "sum"),
            ).reset_index()
            sehir_df["Ek_Sefer"] = sehir_df["Toplam_Sefer"] - sehir_df["Farkli_Otobus"]
            islem_ozeti["sehir_basarili"] = True
            log.info(f"✅ Şehir verisi: {dosya} ({len(sehir_df)} firma)")
            break
        except Exception as e:
            log.warning(f"  ⚠ Şehir hatası ({dosya}): {e}")

    # ── Şantiye Servisleri ────────────────────────
    santiye_dosyalar = [f for f in dosyalar if _dosya_pattern_eslesir(f, SANTIYE_DOSYA_PATTERN)]

    for dosya in santiye_dosyalar:
        islem_ozeti["santiye_denenen"] += 1
        uzanti = os.path.splitext(dosya)[1].lower()

        if uzanti == ".csv":
            try:
                df = _dosya_oku(dosya, header=1)
                df.columns = df.columns.str.strip()
                df = _sutun_eslestir(df, SANTIYE_SUTUN_ALIAS)
                santiye_df = df.groupby("организация").agg(
                    Toplam_Sefer=("Гос-Номер", "count"),
                    Farkli_Otobus=("Гос-Номер", "nunique"),
                    Toplam_Kapasite=("Кол-во мест", "sum"),
                    Toplam_Yolcu=("кол-во поссажиров", "sum"),
                ).reset_index()
                santiye_df["Ek_Sefer"] = santiye_df["Toplam_Sefer"] - santiye_df["Farkli_Otobus"]
                islem_ozeti["santiye_basarili"] = True
                log.info(f"✅ Şantiye verisi (CSV): {dosya}")
                break
            except Exception as e:
                log.warning(f"  ⚠ Şantiye CSV hatası ({dosya}): {e}")
            continue

        # Excel: sayfa adı şablonlarını dene
        sayfa_bulundu = False
        for sablon in SANTIYE_SAYFA_SABLONLARI:
            sayfa_adi = sablon.format(tarih=dun_ruslan)
            try:
                df = _dosya_oku(dosya, sayfa=sayfa_adi, header=1)
                df.columns = df.columns.str.strip()
                df = _sutun_eslestir(df, SANTIYE_SUTUN_ALIAS)
                santiye_df = df.groupby("организация").agg(
                    Toplam_Sefer=("Гос-Номер", "count"),
                    Farkli_Otobus=("Гос-Номер", "nunique"),
                    Toplam_Kapasite=("Кол-во мест", "sum"),
                    Toplam_Yolcu=("кол-во поссажиров", "sum"),
                ).reset_index()
                santiye_df["Ek_Sefer"] = santiye_df["Toplam_Sefer"] - santiye_df["Farkli_Otobus"]
                islem_ozeti["santiye_basarili"] = True
                sayfa_bulundu = True
                log.info(f"✅ Şantiye verisi: {dosya} / '{sayfa_adi}' ({len(santiye_df)} firma)")
                break
            except Exception as e:
                log.debug(f"  ⏭ '{sayfa_adi}' denemesi başarısız: {e}")

        if sayfa_bulundu:
            break
        else:
            log.warning(f"  ⚠ Şantiye sayfası bulunamadı: {dosya} "
                        f"(denenen şablonlar: {len(SANTIYE_SAYFA_SABLONLARI)})")

    # ── İşlem Özeti ───────────────────────────────
    log.info(f"📊 Veri işleme özeti: "
             f"Şehir {'✅' if islem_ozeti['sehir_basarili'] else '❌'} "
             f"({islem_ozeti['sehir_denenen']} dosya denendi) | "
             f"Şantiye {'✅' if islem_ozeti['santiye_basarili'] else '❌'} "
             f"({islem_ozeti['santiye_denenen']} dosya denendi)")

    return sehir_df, santiye_df


# ───────────────────────────────────────────────
#  ANA FONKSİYON
# ───────────────────────────────────────────────

def _rapor_ic_surec():
    """Asıl rapor süreci — retry_ile_calistir ile sarılır."""
    global _DB_MIGRATION_YAPILDI
    if not _DB_MIGRATION_YAPILDI:
        if os.path.exists(GECMIS_DOSYA) or os.path.exists(FIRMA_GECMIS_DOSYA):
            db_json_migration()
        _DB_MIGRATION_YAPILDI = True

    log.info("=" * 55)
    log.info("🚀 RAPOR SÜRECİ BAŞLADI")
    log.info("=" * 55)

    dosyalar = retry_ile_calistir(maillerden_ekleri_indir)
    dosyalar.reverse()

    if not dosyalar:
        telegram_mesaj_gonder("⚠️ Yeni Excel raporu bulunamadı (Ali / Ruslan).")
        return

    dun         = datetime.now() - timedelta(1)
    dun_ruslan  = dun.strftime("%d.%m.%Y")
    dun_ali     = dun.strftime("%Y-%m-%d")
    dun_goster  = dun.strftime("%d.%m.%Y")

    sehir_df, santiye_df = verileri_isle(dosyalar, dun_ali, dun_ruslan)

    if sehir_df.empty and santiye_df.empty:
        telegram_mesaj_gonder(f"⚠️ {dun_goster} için hiç veri okunamadı.")
        return

    # Uyarıları hesapla
    uyarilar = uyarilari_hesapla(sehir_df, santiye_df)
    log.info(f"🔔 {len(uyarilar)} uyarı oluşturuldu.")

    # Geçmişe kaydet
    gecmis_kaydet(dun_ali, sehir_df, santiye_df)
    gecmis = gecmis_son_n_gun(7)

    # Anlık bildirimler — eşik kontrolü
    esik_kontrolu(dun_ali, gecmis, sehir_df, santiye_df)

    # 1. Telegram metin
    mesaj = telegram_gunluk_mesaj(dun_goster, sehir_df, santiye_df, uyarilar, gecmis)
    telegram_mesaj_gonder(mesaj)

    # 2. Günlük grafik → foto
    grafik_bytes = grafik_gunluk_ozet(sehir_df, santiye_df, dun_goster, uyarilar)
    if grafik_bytes:
        telegram_foto_gonder(grafik_bytes, f"📊 Günlük Grafik — {dun_goster}")

    # 3. Haftalık trend grafiği → foto
    trend_bytes = b""
    if len(gecmis) >= 2:
        trend_bytes = grafik_haftalik_trend(gecmis)
        if trend_bytes:
            telegram_foto_gonder(trend_bytes, "📅 Haftalık Trend")

    # 4. Günlük PDF (ML sayfasıyla birlikte)
    log.info("🤖 ML analizi çalıştırılıyor...")
    log.info(f"  Mevcut modeller: Ridge=✅, "
             f"ARIMA={'✅' if _ARIMA_VAR else '❌'}, "
             f"Prophet={'✅' if _PROPHET_VAR else '❌'} "
             f"| Seçim: {ML_MODEL_SECIMI}")

    # Gelişmiş tahmin — ensemble/auto/tekli model
    ml_tahmin        = ml_ensemble_tahmin(gecmis)
    ml_oneriler      = ml_firma_kapasite_onerisi(sehir_df, santiye_df, gecmis)
    ml_tahmin_grafik = grafik_ml_tahmin(gecmis, ml_tahmin) if ml_tahmin else b""
    ml_oneri_grafik  = grafik_ml_firma_onerisi(ml_oneriler)

    if ml_tahmin:
        log.info(f"📈 Tahmin üretildi — model: {ml_tahmin.get('model_adi', '?')}, "
                 f"güven: {ml_tahmin['guven']}, hata: ±{ml_tahmin['model_hata']:.1f}%")
    if ml_oneriler:
        log.info(f"🔧 {len(ml_oneriler)} firma için kapasite önerisi oluşturuldu.")

    # Anomali tespiti
    ml_anomaliler     = anomali_tespit(gecmis)
    ml_anomali_grafik = grafik_anomali(gecmis, ml_anomaliler) if ml_anomaliler else b""
    if ml_anomaliler:
        log.info(f"🔍 {len(ml_anomaliler)} anomali tespit edildi.")
    else:
        log.info("🔍 Anomali tespit edilmedi.")

    # ML Telegram mesajı (anomali dahil)
    ml_mesaj = telegram_ml_mesaj(ml_tahmin, ml_oneriler, anomaliler=ml_anomaliler)
    telegram_mesaj_gonder(ml_mesaj)

    # ML grafikleri → foto
    if ml_tahmin_grafik:
        telegram_foto_gonder(ml_tahmin_grafik,
                             f"📈 7 Günlük Tahmin ({ml_tahmin.get('model_adi', '')})")
    if ml_oneri_grafik:
        telegram_foto_gonder(ml_oneri_grafik, "🔧 Kapasite Optimizasyon Önerisi")
    if ml_anomali_grafik:
        telegram_foto_gonder(ml_anomali_grafik, "🔍 Anomali Tespiti Grafiği")

    pdf_gun = pdf_gunluk_olustur(dun_goster, sehir_df, santiye_df,
                                  grafik_bytes, uyarilar, gecmis,
                                  ml_tahmin=ml_tahmin,
                                  ml_oneriler=ml_oneriler,
                                  ml_tahmin_grafik=ml_tahmin_grafik,
                                  ml_oneri_grafik=ml_oneri_grafik,
                                  ml_anomaliler=ml_anomaliler,
                                  ml_anomali_grafik=ml_anomali_grafik)
    telegram_dosya_gonder(pdf_gun, f"Ulasim_Raporu_{dun_ali}.pdf",
                          f"📄 Günlük PDF — {dun_goster}")

    # 5. Firma Detay Raporu (verimlilik analizi)
    log.info("🏢 Firma detay raporu hazırlanıyor...")
    verimlilik = _firma_verimlilik(sehir_df, santiye_df)
    firma_gecmis = firma_gecmis_oku()

    if verimlilik:
        v_grafik = grafik_verimlilik(verimlilik)
        firma_mesaj = telegram_firma_mesaj(verimlilik, firma_gecmis)
        telegram_mesaj_gonder(firma_mesaj)
        if v_grafik:
            telegram_foto_gonder(v_grafik, "🏢 Firma Verimlilik Sıralaması")

        pdf_firma = pdf_firma_detay_olustur(
            dun_goster, sehir_df, santiye_df,
            verimlilik, v_grafik, firma_gecmis)
        telegram_dosya_gonder(pdf_firma, f"Firma_Detay_{dun_ali}.pdf",
                              "🏢 Firma Detay PDF")
        log.info(f"🏢 {len(verimlilik)} firma için detay raporu gönderildi.")

    # 6. Haftalık özet (her Pazartesi)
    bugun_gun = datetime.now().weekday()   # 0=Pazartesi
    if bugun_gun == 0 and len(gecmis) >= 5:
        log.info("📅 Haftalık özet raporu hazırlanıyor...")
        mesaj_h = telegram_haftalik_mesaj(gecmis)
        telegram_mesaj_gonder(mesaj_h)

        pdf_hafta = pdf_haftalik_olustur(gecmis, trend_bytes)
        if pdf_hafta:
            telegram_dosya_gonder(pdf_hafta,
                                  f"Haftalik_Rapor_{dun_ali}.pdf",
                                  "📅 Haftalık Özet PDF")

    # 7. Aylık özet (her ayın 1'inde — geçen ay için)
    bugun = datetime.now()
    if bugun.day == 1:
        log.info("📅 Aylık özet raporu hazırlanıyor...")
        tum_gecmis = gecmis_tum()
        gecen_ay = (bugun.replace(day=1) - timedelta(days=1))
        ay_adi_map = {
            1: "Ocak", 2: "Şubat", 3: "Mart", 4: "Nisan",
            5: "Mayıs", 6: "Haziran", 7: "Temmuz", 8: "Ağustos",
            9: "Eylül", 10: "Ekim", 11: "Kasım", 12: "Aralık",
        }
        ay_adi = f"{ay_adi_map[gecen_ay.month]} {gecen_ay.year}"
        aylik_gecmis = _aylik_veri(tum_gecmis, gecen_ay.month, gecen_ay.year)

        if aylik_gecmis:
            aylik_grafik = grafik_aylik_ozet(aylik_gecmis, ay_adi)
            aylik_mesaj = telegram_aylik_mesaj(aylik_gecmis, ay_adi, firma_gecmis)
            telegram_mesaj_gonder(aylik_mesaj)
            if aylik_grafik:
                telegram_foto_gonder(aylik_grafik, f"📅 Aylık Trend — {ay_adi}")

            pdf_ay = pdf_aylik_olustur(aylik_gecmis, ay_adi, aylik_grafik, firma_gecmis)
            if pdf_ay:
                telegram_dosya_gonder(pdf_ay,
                                      f"Aylik_Rapor_{gecen_ay.strftime('%Y_%m')}.pdf",
                                      f"📅 Aylık Özet PDF — {ay_adi}")
            log.info(f"📅 Aylık rapor gönderildi: {ay_adi} ({len(aylik_gecmis)} gün)")
        else:
            log.info(f"📅 Aylık rapor: {ay_adi} için yeterli veri yok.")

    log.info("✅ Tüm çıktılar gönderildi.")


def raporu_hazirla_ve_gonder():
    """Ana giriş noktası — retry + hata yönetimi + loglama."""
    baslangic = time.time()
    tarih_str = datetime.now().strftime("%Y-%m-%d")
    try:
        _rapor_ic_surec()
        sure = time.time() - baslangic
        rapor_log_kaydet(tarih_str, "basarili", sure)
        log.info(f"⏱ Toplam süre: {sure:.1f}s")
    except Exception as e:
        sure = time.time() - baslangic
        hata_mesaj = f"{type(e).__name__}: {e}"
        rapor_log_kaydet(tarih_str, "hatali", sure, hata=hata_mesaj,
                         detay=traceback.format_exc()[-500:])
        log.error(f"❌ Rapor süreci başarısız ({sure:.1f}s): {hata_mesaj}")
        # Hata bildirimini Telegram'a gönder
        try:
            telegram_mesaj_gonder(
                f"🚨 *RAPOR HATASI*\n"
                f"━━━━━━━━━━━━━━━━━━━━━\n"
                f"Tarih: {tarih_str}\n"
                f"Hata: `{hata_mesaj[:200]}`\n"
                f"Süre: {sure:.1f}s\n"
                f"_Loglara bakın: rapor\\_bot.log_"
            )
        except Exception:
            pass


if __name__ == "__main__":
    raporu_hazirla_ve_gonder()
