"""
Zamanlayıcı v2 — zamanlayici.py
─────────────────────────────────────────────────────────
Gelişmiş zamanlama sistemi:
  ✅ Exponential backoff retry
  ✅ Health check HTTP endpoint
  ✅ Watchdog — takılmış süreci algıla
  ✅ Konfigürasyon dosyası desteği
  ✅ Graceful shutdown
  ✅ Rapor geçmişi / durum takibi
Çalıştır:  python zamanlayici.py
─────────────────────────────────────────────────────────
"""

import time
import logging
import threading
import signal
import sys
import json
import os
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

# ═══════════════════════════════════════════════
#  AYARLAR (veya zamanlayici_config.json'dan)
# ═══════════════════════════════════════════════

VARSAYILAN_AYARLAR = {
    "gunluk_saatler":     ["08:00", "18:00"],
    "haftalik_pazartesi":  True,
    "haftalik_saat":       "09:00",
    "haftasonu_atla":      False,
    "healthcheck_port":    9090,
    "healthcheck_aktif":   True,
    "watchdog_timeout_dk": 30,
    "retry_max":           3,
    "retry_bekleme_sn":    30,
    "retry_carpan":        2,
}

CONFIG_DOSYA = "zamanlayici_config.json"


def _ayarlar_yukle() -> dict:
    """Config dosyasından ayarları yükler, yoksa varsayılanları kullanır."""
    ayarlar = VARSAYILAN_AYARLAR.copy()
    if os.path.exists(CONFIG_DOSYA):
        try:
            with open(CONFIG_DOSYA, "r", encoding="utf-8") as f:
                ozel = json.load(f)
            ayarlar.update(ozel)
            print(f"  ✅ Config yüklendi: {CONFIG_DOSYA}")
        except Exception as e:
            print(f"  ⚠ Config hatası, varsayılan kullanılıyor: {e}")
    else:
        # Varsayılan config dosyasını oluştur
        try:
            with open(CONFIG_DOSYA, "w", encoding="utf-8") as f:
                json.dump(VARSAYILAN_AYARLAR, f, ensure_ascii=False, indent=2)
            print(f"  ℹ Varsayılan config oluşturuldu: {CONFIG_DOSYA}")
        except Exception:
            pass
    return ayarlar


AYARLAR = _ayarlar_yukle()

# ═══════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ZAM] %(message)s",
    handlers=[
        logging.FileHandler("zamanlayici.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("zamanlayici")


# ═══════════════════════════════════════════════
#  DURUM TAKİBİ
# ═══════════════════════════════════════════════

class BotDurum:
    """Thread-safe durum takibi."""
    def __init__(self):
        self._lock = threading.Lock()
        self.son_calistirma = None
        self.son_basari = None
        self.son_hata = None
        self.toplam_calistirma = 0
        self.toplam_basari = 0
        self.toplam_hata = 0
        self.aktif_calisiyor = False
        self.aktif_baslangic = None
        self.baslangic_zamani = datetime.now()
        self.kapatiliyor = False

    def calistirma_basladi(self):
        with self._lock:
            self.aktif_calisiyor = True
            self.aktif_baslangic = datetime.now()
            self.son_calistirma = datetime.now()
            self.toplam_calistirma += 1

    def calistirma_bitti(self, basarili: bool, hata: str = ""):
        with self._lock:
            self.aktif_calisiyor = False
            self.aktif_baslangic = None
            if basarili:
                self.son_basari = datetime.now()
                self.toplam_basari += 1
            else:
                self.son_hata = hata
                self.toplam_hata += 1

    def durum_dict(self) -> dict:
        with self._lock:
            uptime = (datetime.now() - self.baslangic_zamani).total_seconds()
            return {
                "durum": "kapatiliyor" if self.kapatiliyor
                         else "çalışıyor" if self.aktif_calisiyor
                         else "bekliyor",
                "uptime_saat": round(uptime / 3600, 1),
                "son_calistirma": str(self.son_calistirma) if self.son_calistirma else None,
                "son_basari": str(self.son_basari) if self.son_basari else None,
                "son_hata": self.son_hata,
                "toplam_calistirma": self.toplam_calistirma,
                "toplam_basari": self.toplam_basari,
                "toplam_hata": self.toplam_hata,
                "aktif_sure_dk": round((datetime.now() - self.aktif_baslangic).total_seconds() / 60, 1)
                    if self.aktif_baslangic else 0,
            }


DURUM = BotDurum()


# ═══════════════════════════════════════════════
#  HEALTH CHECK HTTP SERVER
# ═══════════════════════════════════════════════

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            d = DURUM.durum_dict()
            code = 200 if d["toplam_hata"] < d["toplam_calistirma"] or d["toplam_calistirma"] == 0 else 503
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(d, ensure_ascii=False).encode())
        elif self.path == "/status":
            d = DURUM.durum_dict()
            html = f"""<html><body style="font-family:monospace;background:#111;color:#eee;padding:20px">
            <h2>🚌 Zamanlayıcı Durum</h2>
            <pre>{json.dumps(d, ensure_ascii=False, indent=2)}</pre>
            </body></html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # HTTP loglarını bastır


def _health_server_baslat():
    port = AYARLAR["healthcheck_port"]
    try:
        server = HTTPServer(("0.0.0.0", port), HealthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True, name="healthcheck")
        t.start()
        log.info(f"🏥 Health check: http://localhost:{port}/health")
        return server
    except Exception as e:
        log.warning(f"Health check başlatılamadı: {e}")
        return None


# ═══════════════════════════════════════════════
#  RETRY MEKANİZMASI
# ═══════════════════════════════════════════════

def _retry_calistir(fonksiyon, etiket: str = ""):
    """Exponential backoff ile retry."""
    max_deneme = AYARLAR["retry_max"]
    bekleme = AYARLAR["retry_bekleme_sn"]
    carpan = AYARLAR["retry_carpan"]

    for deneme in range(1, max_deneme + 1):
        DURUM.calistirma_basladi()
        try:
            fonksiyon()
            DURUM.calistirma_bitti(True)
            log.info(f"✅ {etiket} tamamlandı (deneme {deneme}).")
            return True
        except Exception as e:
            hata_str = f"{type(e).__name__}: {e}"
            DURUM.calistirma_bitti(False, hata_str)

            if deneme < max_deneme:
                bekle = bekleme * (carpan ** (deneme - 1))
                log.warning(f"⚠ {etiket} başarısız (deneme {deneme}/{max_deneme}): {e}")
                log.info(f"  ⏳ {bekle}s sonra tekrar denenecek...")
                time.sleep(bekle)
            else:
                log.error(f"❌ {etiket} {max_deneme} denemede başarısız: {e}")
                return False
    return False


# ═══════════════════════════════════════════════
#  WATCHDOG
# ═══════════════════════════════════════════════

def _watchdog():
    """Takılmış rapor sürecini algılar."""
    timeout_dk = AYARLAR["watchdog_timeout_dk"]
    log.info(f"🐕 Watchdog aktif (timeout: {timeout_dk} dk)")
    while not DURUM.kapatiliyor:
        time.sleep(60)
        d = DURUM.durum_dict()
        if d["durum"] == "çalışıyor" and d["aktif_sure_dk"] > timeout_dk:
            log.error(f"🐕 WATCHDOG: Rapor süreci {d['aktif_sure_dk']:.0f} dk'dır çalışıyor! "
                      f"(timeout: {timeout_dk} dk)")
            DURUM.calistirma_bitti(False, f"Watchdog timeout ({d['aktif_sure_dk']:.0f} dk)")


# ═══════════════════════════════════════════════
#  ZAMANLAMA FONKSİYONLARI
# ═══════════════════════════════════════════════

def _simdi() -> str:
    return datetime.now().strftime("%H:%M")


def _bugun_gun() -> int:
    return datetime.now().weekday()


def _haftasonu() -> bool:
    return _bugun_gun() >= 5


def _sonraki_sure(hedef_saat: str) -> float:
    simdi = datetime.now()
    h, m = map(int, hedef_saat.split(":"))
    hedef = simdi.replace(hour=h, minute=m, second=0, microsecond=0)
    if hedef <= simdi:
        hedef += timedelta(days=1)
    return (hedef - simdi).total_seconds()


def _raporu_calistir(etiket: str = "GÜNLÜK"):
    """rapor_bot modülünü güvenli şekilde çağırır."""
    def _ic():
        try:
            from rapor_bot_v2 import raporu_hazirla_ve_gonder
            raporu_hazirla_ve_gonder()
        except ImportError as e:
            log.error(f"❌ Bot import hatası: {e}")
            raise

    _retry_calistir(_ic, etiket)


# ── Zamanlayıcı döngüleri ────────────────────────

def gunluk_dongu(saat: str):
    log.info(f"⏰ Günlük zamanlayıcı kuruldu: {saat}")
    while not DURUM.kapatiliyor:
        bekleme = _sonraki_sure(saat)
        log.info(f"   [{saat}] Sonraki çalışmaya {bekleme/3600:.1f} saat kaldı.")

        # Bekleme sırasında kapatma kontrolü (60s aralıklarla)
        beklenecek = bekleme
        while beklenecek > 0 and not DURUM.kapatiliyor:
            uyku = min(beklenecek, 60)
            time.sleep(uyku)
            beklenecek -= uyku

        if DURUM.kapatiliyor:
            break

        if AYARLAR["haftasonu_atla"] and _haftasonu():
            log.info(f"   [{saat}] Haftasonu — atlandı.")
            continue

        _raporu_calistir(f"GÜNLÜK {saat}")


def haftalik_dongu():
    if not AYARLAR["haftalik_pazartesi"]:
        return
    saat = AYARLAR["haftalik_saat"]
    log.info(f"📅 Haftalık zamanlayıcı kuruldu: Pazartesi {saat}")
    while not DURUM.kapatiliyor:
        bekleme = _sonraki_sure(saat)

        beklenecek = bekleme
        while beklenecek > 0 and not DURUM.kapatiliyor:
            uyku = min(beklenecek, 60)
            time.sleep(uyku)
            beklenecek -= uyku

        if DURUM.kapatiliyor:
            break

        if _bugun_gun() == 0:
            _raporu_calistir("HAFTALIK")


# ═══════════════════════════════════════════════
#  GRACEFUL SHUTDOWN
# ═══════════════════════════════════════════════

def _sinyal_yakala(sig, frame):
    log.info(f"🛑 Kapatma sinyali alındı ({sig})...")
    DURUM.kapatiliyor = True


signal.signal(signal.SIGINT, _sinyal_yakala)
signal.signal(signal.SIGTERM, _sinyal_yakala)


# ═══════════════════════════════════════════════
#  DURUM EKRANI
# ═══════════════════════════════════════════════

def durum_yazdir():
    gun_isimleri = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
    d = DURUM.durum_dict()

    print("\n" + "═" * 56)
    print("  🚌 ULAŞIM RAPOR BOTU — ZAMANLAYİCİ v2")
    print("═" * 56)
    print(f"  Şu an    : {datetime.now().strftime('%d.%m.%Y %H:%M')}  ({gun_isimleri[_bugun_gun()]})")
    print(f"  Durum    : {d['durum']}")
    print(f"  Uptime   : {d['uptime_saat']:.1f} saat")
    print(f"  Çalışma  : {d['toplam_basari']}/{d['toplam_calistirma']} başarılı"
          f"  ({d['toplam_hata']} hata)")
    print(f"  Günlük saatler:")
    for s in AYARLAR["gunluk_saatler"]:
        kalan = _sonraki_sure(s)
        sa, dk = divmod(int(kalan), 3600)
        dk //= 60
        print(f"    • {s}  →  {sa}s {dk}dk sonra")
    if AYARLAR["haftalik_pazartesi"]:
        print(f"  Haftalık : Her Pazartesi {AYARLAR['haftalik_saat']}")
    print(f"  Haftasonu: {'⛔ Atlanıyor' if AYARLAR['haftasonu_atla'] else '✅ Çalışıyor'}")
    print(f"  Retry    : max {AYARLAR['retry_max']}, "
          f"backoff {AYARLAR['retry_bekleme_sn']}s×{AYARLAR['retry_carpan']}")
    if AYARLAR["healthcheck_aktif"]:
        print(f"  Health   : http://localhost:{AYARLAR['healthcheck_port']}/health")
    print(f"  Watchdog : {AYARLAR['watchdog_timeout_dk']} dk timeout")
    print("═" * 56)
    print("  Durdurmak için Ctrl+C\n")


# ═══════════════════════════════════════════════
#  ANA GİRİŞ
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    durum_yazdir()

    threads = []

    # Health check
    health_server = None
    if AYARLAR["healthcheck_aktif"]:
        health_server = _health_server_baslat()

    # Watchdog
    t = threading.Thread(target=_watchdog, daemon=True, name="watchdog")
    t.start()
    threads.append(t)

    # Günlük zamanlayıcılar
    for saat in AYARLAR["gunluk_saatler"]:
        t = threading.Thread(target=gunluk_dongu, args=(saat,), daemon=True, name=f"gun-{saat}")
        t.start()
        threads.append(t)

    # Haftalık
    if AYARLAR["haftalik_pazartesi"]:
        t = threading.Thread(target=haftalik_dongu, daemon=True, name="haftalik")
        t.start()
        threads.append(t)

    log.info(f"✅ {len(threads)} thread aktif.")

    # Ana thread — durum göster + graceful shutdown bekle
    try:
        while not DURUM.kapatiliyor:
            time.sleep(60)
            # Her saat durum yaz
            if datetime.now().minute == 0:
                durum_yazdir()
    except KeyboardInterrupt:
        pass

    DURUM.kapatiliyor = True
    log.info("🛑 Zamanlayıcı kapatılıyor...")

    if health_server:
        health_server.shutdown()

    # Thread'lerin bitmesini kısa süre bekle
    for t in threads:
        t.join(timeout=5)

    log.info("👋 Zamanlayıcı durduruldu.")
