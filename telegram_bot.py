"""
Telegram Interaktif Bot — telegram_bot.py
─────────────────────────────────────────────
Komutlar:
  /durum        — Anlık doluluk + araç özeti
  /rapor        — Son günün tam raporunu gönder
  /firma        — Firma listesi veya detay (/firma METRO)
  /tahmin       — ML doluluk tahmini
  /karsilastir  — Dün vs önceki gün
  /verimlilik   — Firma verimlilik sıralaması
  /anomali      — Anomali tespiti
  /yardim       — Komut listesi

  Rusça:
  /статус       — Durum özeti
  /отчет        — Rapor gönder
  /помощь       — Yardım

Çalıştır:  python telegram_bot.py
GitHub Actions:  workflow'a eklendi (polling modu)
─────────────────────────────────────────────
"""

import os
import sys
import json
import time
import logging
import requests
import threading
from datetime import datetime, timedelta

# Aynı dizindeki rapor modülünü import et
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOT] %(message)s",
    handlers=[
        logging.FileHandler("telegram_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("telegram_bot")

# ═══════════════════════════════════════════════
#  AYARLAR
# ═══════════════════════════════════════════════

TOKEN = os.environ.get("TELEGRAM_TOKEN", "7978688244:AAGLtC_4_tqQwMlMubfaNDeGuDnJVsN1Yn4")
IZINLI_CHATLER = os.environ.get("TELEGRAM_CHAT", "886597229").split(",")
API = f"https://api.telegram.org/bot{TOKEN}"
POLLING_ARALIK = 2  # saniye


# ═══════════════════════════════════════════════
#  TELEGRAM API YARDIMCILARI
# ═══════════════════════════════════════════════

def _tg(method: str, **kwargs) -> dict:
    try:
        r = requests.post(f"{API}/{method}", timeout=30, **kwargs)
        return r.json()
    except Exception as e:
        log.error(f"TG API hatası ({method}): {e}")
        return {}


def mesaj_gonder(chat_id: str, text: str, parse_mode: str = "Markdown"):
    # Telegram 4096 karakter limiti
    if len(text) > 4000:
        parcalar = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for p in parcalar:
            _tg("sendMessage", json={"chat_id": chat_id, "text": p, "parse_mode": parse_mode})
    else:
        _tg("sendMessage", json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode})


def foto_gonder(chat_id: str, foto_bytes: bytes, baslik: str = ""):
    _tg("sendPhoto",
        data={"chat_id": chat_id, "caption": baslik},
        files={"photo": ("grafik.png", foto_bytes, "image/png")})


def dosya_gonder(chat_id: str, dosya_bytes: bytes, dosya_adi: str, baslik: str = ""):
    _tg("sendDocument",
        data={"chat_id": chat_id, "caption": baslik},
        files={"document": (dosya_adi, dosya_bytes, "application/pdf")})


# ═══════════════════════════════════════════════
#  VERİ OKUMA
# ═══════════════════════════════════════════════

def _db_oku_gunluk(n: int = 7) -> dict:
    """SQLite'dan son N günü oku."""
    try:
        from rapor_bot_v2 import gecmis_son_n_gun
        return gecmis_son_n_gun(n)
    except Exception:
        pass
    # Fallback: JSON
    dosya = "gecmis_veriler.json"
    if not os.path.exists(dosya):
        return {}
    with open(dosya, "r", encoding="utf-8") as f:
        tum = json.load(f)
    return {k: tum[k] for k in sorted(tum)[-n:]}


def _db_oku_firma() -> dict:
    """Firma geçmişini oku."""
    try:
        from rapor_bot_v2 import firma_gecmis_oku
        return firma_gecmis_oku()
    except Exception:
        pass
    dosya = "firma_gecmis.json"
    if not os.path.exists(dosya):
        return {}
    with open(dosya, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════
#  KOMUT İŞLEYİCİLER
# ═══════════════════════════════════════════════

def cmd_durum(chat_id: str):
    """Son günün anlık özeti."""
    gecmis = _db_oku_gunluk(2)
    if not gecmis:
        mesaj_gonder(chat_id, "⚠ Henüz veri yok.")
        return

    tarihler = sorted(gecmis.keys())
    son = gecmis[tarihler[-1]]
    onceki = gecmis[tarihler[-2]] if len(tarihler) > 1 else None

    dol = son["doluluk"]
    emo = "🟢" if dol >= 80 else "🟡" if dol >= 60 else "🔴"

    m = f"{emo} *ANLIK DURUM*\n"
    m += f"━━━━━━━━━━━━━━━━━━━━━\n"
    m += f"📅 Tarih: *{tarihler[-1]}*\n"
    m += f"🚍 Araç: *{son['arac']}*\n"
    m += f"👥 Yolcu: *{son['yolcu']:,}* / {son['kapasite']:,} koltuk\n"
    m += f"{emo} Doluluk: *%{dol:.1f}*\n"
    m += f"🪑 Boş koltuk: *{son['kapasite'] - son['yolcu']:,}*\n"

    if onceki:
        fark_dol = dol - onceki["doluluk"]
        fark_yolcu = son["yolcu"] - onceki["yolcu"]
        ok = "📈" if fark_dol > 0 else "📉" if fark_dol < 0 else "➡️"
        m += f"━━━━━━━━━━━━━━━━━━━━━\n"
        m += f"{ok} Önceki güne göre: *{'+' if fark_dol >= 0 else ''}{fark_dol:.1f}%* doluluk\n"
        m += f"   Yolcu farkı: *{'+' if fark_yolcu >= 0 else ''}{fark_yolcu:,}*\n"

    mesaj_gonder(chat_id, m)


def cmd_rapor(chat_id: str):
    """Son günün tam raporunu tetikler."""
    mesaj_gonder(chat_id, "⏳ Rapor hazırlanıyor...")
    try:
        from rapor_bot_v2 import raporu_hazirla_ve_gonder
        raporu_hazirla_ve_gonder()
        mesaj_gonder(chat_id, "✅ Rapor gönderildi!")
    except Exception as e:
        mesaj_gonder(chat_id, f"❌ Rapor hatası: `{e}`")


def cmd_firma(chat_id: str, args: str = ""):
    """Firma listesi veya detayı."""
    firma_gecmis = _db_oku_firma()
    if not firma_gecmis:
        mesaj_gonder(chat_id, "⚠ Firma verisi bulunamadı.")
        return

    son_tarih = sorted(firma_gecmis.keys())[-1]
    firmalar = firma_gecmis[son_tarih]

    if not args.strip():
        # Firma listesi
        m = f"🏢 *FİRMA LİSTESİ* ({son_tarih})\n"
        m += f"━━━━━━━━━━━━━━━━━━━━━\n"
        for firma, v in sorted(firmalar.items(), key=lambda x: x[1].get("doluluk", 0), reverse=True):
            dol = v.get("doluluk", 0)
            emo = "🟢" if dol >= 80 else "🟡" if dol >= 60 else "🔴"
            m += f"{emo} *{firma}* [{v.get('kategori', '?')}]\n"
            m += f"   %{dol:.1f} | {v.get('arac', 0)} araç | {v.get('yolcu', 0):,} yolcu\n"
        m += f"\n💡 _Detay için: /firma FIRMA\\_ADI_"
        mesaj_gonder(chat_id, m)
        return

    # Belirli firma detayı
    aranan = args.strip().upper()
    bulunan = None
    for firma in firmalar:
        if aranan in firma.upper():
            bulunan = firma
            break

    if not bulunan:
        mesaj_gonder(chat_id, f"⚠ '{args.strip()}' bulunamadı.\n/firma yazarak listeye bakabilirsin.")
        return

    # Firma trend (son 7 gün)
    m = f"🏢 *{bulunan}*\n━━━━━━━━━━━━━━━━━━━━━\n"
    tarihler = sorted(firma_gecmis.keys())[-7:]
    for tarih in tarihler:
        if bulunan in firma_gecmis[tarih]:
            v = firma_gecmis[tarih][bulunan]
            dol = v.get("doluluk", 0)
            emo = "🟢" if dol >= 80 else "🟡" if dol >= 60 else "🔴"
            m += f"{emo} `{tarih[5:]}` %{dol:.1f} | {v.get('arac', 0)} araç | {v.get('yolcu', 0):,} yolcu\n"
        else:
            m += f"⚪ `{tarih[5:]}` veri yok\n"

    # Grafik
    try:
        from rapor_bot_v2 import grafik_firma_detay
        grafik = grafik_firma_detay(firma_gecmis, bulunan)
        if grafik:
            foto_gonder(chat_id, grafik, f"📊 {bulunan} — Trend")
    except Exception:
        pass

    mesaj_gonder(chat_id, m)


def cmd_tahmin(chat_id: str):
    """ML doluluk tahmini."""
    gecmis = _db_oku_gunluk(14)
    if len(gecmis) < 7:
        mesaj_gonder(chat_id, f"⚠ Tahmin için en az 7 günlük veri gerekiyor (mevcut: {len(gecmis)}).")
        return

    mesaj_gonder(chat_id, "🤖 Tahmin hesaplanıyor...")
    try:
        from rapor_bot_v2 import ml_ensemble_tahmin, grafik_ml_tahmin
        sonuc = ml_ensemble_tahmin(gecmis)
        if not sonuc:
            mesaj_gonder(chat_id, "⚠ Tahmin üretilemedi.")
            return

        m = f"🤖 *{sonuc.get('model_adi', 'ML')} TAHMİNİ*\n"
        m += f"━━━━━━━━━━━━━━━━━━━━━\n"
        m += f"_Güven: {sonuc['guven']} | ±{sonuc['model_hata']:.1f}% hata_\n\n"
        for tarih, dol in sonuc["tahminler"]:
            dt = datetime.strptime(tarih, "%Y-%m-%d")
            gun = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"][dt.weekday()]
            emo = "🟢" if dol >= 80 else "🟡" if dol >= 60 else "🔴"
            m += f"{emo} `{dt.strftime('%d.%m')} {gun}` → *%{dol:.1f}*\n"

        grafik = grafik_ml_tahmin(gecmis, sonuc)
        if grafik:
            foto_gonder(chat_id, grafik, f"📈 {sonuc.get('model_adi', '')} Tahmini")

        mesaj_gonder(chat_id, m)
    except Exception as e:
        mesaj_gonder(chat_id, f"❌ Tahmin hatası: `{e}`")


def cmd_karsilastir(chat_id: str):
    """Son 2 günü karşılaştırır."""
    gecmis = _db_oku_gunluk(2)
    if len(gecmis) < 2:
        mesaj_gonder(chat_id, "⚠ Karşılaştırma için en az 2 günlük veri gerekiyor.")
        return

    tarihler = sorted(gecmis.keys())
    t1, t2 = tarihler[-2], tarihler[-1]
    v1, v2 = gecmis[t1], gecmis[t2]

    def _fark(a, b, birim=""):
        f = b - a
        ok = "📈" if f > 0 else "📉" if f < 0 else "➡️"
        return f"{ok} {'+' if f >= 0 else ''}{f:,.1f}{birim}"

    m = f"📊 *KARŞILAŞTIRMA*\n"
    m += f"━━━━━━━━━━━━━━━━━━━━━\n"
    m += f"📅 *{t1}* vs *{t2}*\n\n"

    m += f"*Doluluk:*\n"
    m += f"  Önceki: %{v1['doluluk']:.1f}  →  Son: %{v2['doluluk']:.1f}\n"
    m += f"  {_fark(v1['doluluk'], v2['doluluk'], '%')}\n\n"

    m += f"*Yolcu:*\n"
    m += f"  Önceki: {v1['yolcu']:,}  →  Son: {v2['yolcu']:,}\n"
    m += f"  {_fark(v1['yolcu'], v2['yolcu'])}\n\n"

    m += f"*Araç:*\n"
    m += f"  Önceki: {v1['arac']}  →  Son: {v2['arac']}\n"
    m += f"  {_fark(v1['arac'], v2['arac'])}\n\n"

    m += f"*Kapasite:*\n"
    m += f"  Önceki: {v1['kapasite']:,}  →  Son: {v2['kapasite']:,}\n"
    m += f"  {_fark(v1['kapasite'], v2['kapasite'])}\n"

    mesaj_gonder(chat_id, m)


def cmd_verimlilik(chat_id: str):
    """Firma verimlilik sıralaması."""
    firma_gecmis = _db_oku_firma()
    if not firma_gecmis:
        mesaj_gonder(chat_id, "⚠ Firma verisi bulunamadı.")
        return

    son_tarih = sorted(firma_gecmis.keys())[-1]
    firmalar = firma_gecmis[son_tarih]

    skorlar = []
    for firma, v in firmalar.items():
        dol = v.get("doluluk", 0)
        arac = v.get("arac", 1)
        sefer = v.get("sefer", 0)
        yolcu = v.get("yolcu", 0)
        s_a = sefer / arac if arac > 0 else 0
        y_a = yolcu / arac if arac > 0 else 0
        skor = min(100, dol * 0.5 + min(s_a, 3) / 3 * 30 + min(y_a, 100) / 100 * 20)
        skorlar.append((firma, v.get("kategori", "?"), skor, dol))

    skorlar.sort(key=lambda x: x[2], reverse=True)

    m = f"🏆 *VERİMLİLİK SIRALAMASI* ({son_tarih})\n"
    m += f"━━━━━━━━━━━━━━━━━━━━━\n"
    for i, (firma, kat, skor, dol) in enumerate(skorlar, 1):
        emo = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        s_emo = "🟢" if skor >= 70 else "🟡" if skor >= 50 else "🔴"
        m += f"{emo} {s_emo} *{firma}* [{kat}]\n"
        m += f"   Skor: *{skor:.0f}* | Doluluk: %{dol:.1f}\n"

    mesaj_gonder(chat_id, m)


def cmd_anomali(chat_id: str):
    """Anomali tespiti."""
    gecmis = _db_oku_gunluk(14)
    if len(gecmis) < 5:
        mesaj_gonder(chat_id, "⚠ Anomali tespiti için en az 5 günlük veri gerekiyor.")
        return

    try:
        from rapor_bot_v2 import anomali_tespit, grafik_anomali
        anomaliler = anomali_tespit(gecmis)
        if not anomaliler:
            mesaj_gonder(chat_id, "✅ Anomali tespit edilmedi — tüm metrikler normal aralıkta.")
            return

        m = f"🔍 *ANOMALİ TESPİTİ* ({len(anomaliler)} anomali)\n"
        m += f"━━━━━━━━━━━━━━━━━━━━━\n"
        for a in anomaliler[:10]:
            m += f"⚠️ `{a['tarih'][5:]}` *{a['metrik']}*: {a['deger']}\n"
            m += f"   _{a['yontem']}: {a['aciklama'][:50]}_\n"

        grafik = grafik_anomali(gecmis, anomaliler)
        if grafik:
            foto_gonder(chat_id, grafik, f"🔍 Anomali — {len(anomaliler)} tespit")

        mesaj_gonder(chat_id, m)
    except Exception as e:
        mesaj_gonder(chat_id, f"❌ Anomali hatası: `{e}`")


def cmd_yardim(chat_id: str, dil: str = "tr"):
    """Komut listesi."""
    if dil == "ru":
        m = "🤖 *КОМАНДЫ БОТА*\n"
        m += "━━━━━━━━━━━━━━━━━━━━━\n"
        m += "/статус — Текущий статус\n"
        m += "/отчет — Отправить отчет\n"
        m += "/фирма — Список фирм\n"
        m += "/прогноз — ML прогноз\n"
        m += "/сравнить — Сравнение дней\n"
        m += "/помощь — Эта помощь\n"
    else:
        m = "🤖 *BOT KOMUTLARI*\n"
        m += "━━━━━━━━━━━━━━━━━━━━━\n"
        m += "/durum — Anlık doluluk özeti\n"
        m += "/rapor — Tam raporu gönder\n"
        m += "/firma — Firma listesi\n"
        m += "/firma METRO — Firma detay\n"
        m += "/tahmin — ML doluluk tahmini\n"
        m += "/karsilastir — Dün vs bugün\n"
        m += "/verimlilik — Firma sıralaması\n"
        m += "/anomali — Anomali tespiti\n"
        m += "/yardim — Bu yardım\n"
        m += "━━━━━━━━━━━━━━━━━━━━━\n"
        m += "🇷🇺 /помощь — Русский\n"

    mesaj_gonder(chat_id, m)


# Rusça alias'lar
RU_KOMUTLAR = {
    "/статус":   lambda cid, _: cmd_durum(cid),
    "/отчет":    lambda cid, _: cmd_rapor(cid),
    "/фирма":    lambda cid, a: cmd_firma(cid, a),
    "/прогноз":  lambda cid, _: cmd_tahmin(cid),
    "/сравнить": lambda cid, _: cmd_karsilastir(cid),
    "/помощь":   lambda cid, _: cmd_yardim(cid, "ru"),
}


# ═══════════════════════════════════════════════
#  KOMUT YÖNLENDİRİCİ
# ═══════════════════════════════════════════════

KOMUTLAR = {
    "/durum":       lambda cid, _: cmd_durum(cid),
    "/rapor":       lambda cid, _: cmd_rapor(cid),
    "/firma":       lambda cid, a: cmd_firma(cid, a),
    "/tahmin":      lambda cid, _: cmd_tahmin(cid),
    "/karsilastir": lambda cid, _: cmd_karsilastir(cid),
    "/verimlilik":  lambda cid, _: cmd_verimlilik(cid),
    "/anomali":     lambda cid, _: cmd_anomali(cid),
    "/yardim":      lambda cid, _: cmd_yardim(cid),
    "/help":        lambda cid, _: cmd_yardim(cid),
    "/start":       lambda cid, _: cmd_yardim(cid),
    **RU_KOMUTLAR,
}


def mesaj_isle(update: dict):
    """Gelen mesajı işler."""
    msg = update.get("message", {})
    chat_id = str(msg.get("chat", {}).get("id", ""))
    text = msg.get("text", "").strip()

    if not chat_id or not text:
        return

    # Yetki kontrolü
    if chat_id not in IZINLI_CHATLER:
        log.warning(f"⛔ Yetkisiz erişim: chat_id={chat_id}")
        mesaj_gonder(chat_id, "⛔ Bu bot sadece yetkili kullanıcılar içindir.")
        return

    # Komutu parse et
    parcalar = text.split(maxsplit=1)
    komut = parcalar[0].lower().split("@")[0]  # /durum@botname → /durum
    args = parcalar[1] if len(parcalar) > 1 else ""

    if komut in KOMUTLAR:
        log.info(f"📩 Komut: {komut} (chat={chat_id}, args='{args}')")
        try:
            KOMUTLAR[komut](chat_id, args)
        except Exception as e:
            log.error(f"Komut hatası ({komut}): {e}")
            mesaj_gonder(chat_id, f"❌ Komut hatası: `{e}`")
    else:
        mesaj_gonder(chat_id, f"❓ Bilinmeyen komut: `{komut}`\n/yardim yazarak komutları görebilirsin.")


# ═══════════════════════════════════════════════
#  POLLING DÖNGÜSÜ
# ═══════════════════════════════════════════════

def polling_baslat():
    """Long polling ile mesajları dinler."""
    log.info("🤖 Telegram bot başlatılıyor (polling)...")
    offset = 0

    while True:
        try:
            r = requests.get(f"{API}/getUpdates", params={
                "offset": offset, "timeout": 30, "allowed_updates": ["message"],
            }, timeout=35)
            data = r.json()

            if not data.get("ok"):
                log.error(f"getUpdates hatası: {data}")
                time.sleep(5)
                continue

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                threading.Thread(target=mesaj_isle, args=(update,), daemon=True).start()

        except requests.exceptions.Timeout:
            continue
        except Exception as e:
            log.error(f"Polling hatası: {e}")
            time.sleep(5)


# ═══════════════════════════════════════════════
#  GİRİŞ
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 Telegram Interaktif Bot")
    print(f"   İzinli chat'ler: {IZINLI_CHATLER}")
    print("   Durdurmak için: Ctrl+C")
    print("=" * 50)

    # Bot bilgilerini göster
    me = _tg("getMe")
    if me.get("ok"):
        bot = me["result"]
        print(f"   Bot: @{bot.get('username', '?')}")
    print("=" * 50)

    polling_baslat()
