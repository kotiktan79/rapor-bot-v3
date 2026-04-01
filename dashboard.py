"""
Ulaşım Operasyon Dashboard v2 — dashboard.py
─────────────────────────────────────────────
Gelişmiş Flask tabanlı interaktif web arayüzü.
  ✅ Tarih aralığı filtresi
  ✅ Metrik seçici (doluluk, yolcu, araç, kapasite)
  ✅ İstatistik paneli (min, max, ort, std, trend)
  ✅ Anomali işaretleri
  ✅ Karşılaştırma modu (dönem vs dönem)
  ✅ CSV export
  ✅ Auto-refresh toggle
  ✅ Responsive tasarım
Çalıştır:  python dashboard.py
Aç:        http://localhost:8080
─────────────────────────────────────────────
"""

from flask import Flask, jsonify, render_template_string, request, Response
import json, os, csv, io
from datetime import datetime, timedelta
import numpy as np

GECMIS_DOSYA = "gecmis_veriler.json"

app = Flask(__name__)


def _veri_oku() -> dict:
    if not os.path.exists(GECMIS_DOSYA):
        return {}
    with open(GECMIS_DOSYA, "r", encoding="utf-8") as f:
        return json.load(f)


def _gunler_listesi(raw: dict, baslangic: str = None, bitis: str = None) -> list:
    gunler = []
    for tarih in sorted(raw.keys()):
        if baslangic and tarih < baslangic:
            continue
        if bitis and tarih > bitis:
            continue
        v = raw[tarih]
        gunler.append({
            "tarih":    tarih,
            "arac":     v.get("arac", 0),
            "yolcu":    v.get("yolcu", 0),
            "kapasite": v.get("kapasite", 1),
            "doluluk":  v.get("doluluk", 0),
        })
    return gunler


def _istatistik(gunler: list, metrik: str = "doluluk") -> dict:
    if not gunler:
        return {}
    degerler = [g[metrik] for g in gunler]
    arr = np.array(degerler)
    # Trend: son 3 günün ortalaması vs ilk 3 günün ortalaması
    trend = "→"
    if len(arr) >= 4:
        ilk = np.mean(arr[:3])
        son = np.mean(arr[-3:])
        fark = son - ilk
        trend = "↑" if fark > 2 else "↓" if fark < -2 else "→"
    return {
        "min": round(float(np.min(arr)), 1),
        "max": round(float(np.max(arr)), 1),
        "ort": round(float(np.mean(arr)), 1),
        "std": round(float(np.std(arr)), 1),
        "medyan": round(float(np.median(arr)), 1),
        "toplam": round(float(np.sum(arr)), 1),
        "gun_sayisi": len(arr),
        "trend": trend,
    }


def _anomali_basit(gunler: list) -> list:
    """Z-score tabanlı basit anomali tespiti (dashboard için)."""
    if len(gunler) < 4:
        return []
    anomaliler = []
    for metrik in ["doluluk", "yolcu", "arac"]:
        degerler = np.array([g[metrik] for g in gunler])
        ort = np.mean(degerler)
        std = np.std(degerler)
        if std < 0.01:
            continue
        for i, val in enumerate(degerler):
            z = (val - ort) / std
            if abs(z) > 1.8:
                anomaliler.append({
                    "tarih": gunler[i]["tarih"],
                    "metrik": metrik,
                    "deger": round(float(val), 1),
                    "z_skor": round(float(z), 2),
                    "yon": "yüksek" if z > 0 else "düşük",
                })
    return anomaliler


# ── HTML ──────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ulaşım Dashboard v2</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root {
  --bg:       #0a0e1a;
  --bg2:      #0f1424;
  --surface:  #151b2e;
  --surface2: #1a2138;
  --border:   #1e2d4a;
  --border2:  #293d5a;
  --accent:   #3b82f6;
  --accent2:  #6366f1;
  --green:    #10b981;
  --red:      #ef4444;
  --yellow:   #f59e0b;
  --purple:   #8b5cf6;
  --cyan:     #06b6d4;
  --text:     #e2e8f0;
  --text2:    #94a3b8;
  --muted:    #64748b;
  --font:     'Inter', -apple-system, sans-serif;
  --mono:     'JetBrains Mono', monospace;
  --radius:   12px;
  --shadow:   0 4px 24px rgba(0,0,0,.3);
}

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font);
  min-height: 100vh;
  overflow-x: hidden;
}
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% -10%, rgba(59,130,246,.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 110%, rgba(99,102,241,.06) 0%, transparent 60%);
  pointer-events: none;
}

/* ── Layout ── */
.app { position: relative; z-index: 1; display: flex; min-height: 100vh; }
.sidebar {
  width: 280px; min-width: 280px;
  background: var(--bg2);
  border-right: 1px solid var(--border);
  padding: 20px 16px;
  display: flex; flex-direction: column; gap: 16px;
  overflow-y: auto;
  position: sticky; top: 0; height: 100vh;
}
.main { flex: 1; padding: 20px 24px; overflow-y: auto; }

@media (max-width: 900px) {
  .app { flex-direction: column; }
  .sidebar {
    width: 100%; min-width: auto;
    position: static; height: auto;
    flex-direction: row; flex-wrap: wrap;
    padding: 12px;
  }
  .sidebar .sb-section { min-width: 200px; flex: 1; }
  .main { padding: 12px; }
}

/* ── Sidebar ── */
.sb-logo { display: flex; align-items: center; gap: 10px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.sb-logo-icon {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  display: flex; align-items: center; justify-content: center; font-size: 18px;
}
.sb-logo h1 { font-size: 15px; font-weight: 800; letter-spacing: -0.3px; }
.sb-logo span { font-size: 11px; color: var(--muted); display: block; }

.sb-section { display: flex; flex-direction: column; gap: 8px; }
.sb-label {
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.5px; color: var(--muted); padding: 4px 0;
}
.sb-input {
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  color: var(--text); padding: 8px 10px; font-size: 13px; font-family: var(--mono);
  width: 100%; outline: none; transition: border-color .2s;
}
.sb-input:focus { border-color: var(--accent); }

.sb-btn {
  background: var(--accent); color: white; border: none; border-radius: 8px;
  padding: 9px 14px; font-size: 12px; font-weight: 600; cursor: pointer;
  transition: all .15s; font-family: var(--font); display: flex; align-items: center; gap: 6px;
}
.sb-btn:hover { background: #2563eb; transform: translateY(-1px); }
.sb-btn.secondary { background: var(--surface); border: 1px solid var(--border); color: var(--text2); }
.sb-btn.secondary:hover { border-color: var(--accent); color: var(--text); }
.sb-btn.small { padding: 6px 10px; font-size: 11px; }

.sb-chips { display: flex; flex-wrap: wrap; gap: 4px; }
.chip {
  padding: 5px 10px; border-radius: 20px; font-size: 11px; font-weight: 600;
  cursor: pointer; border: 1px solid var(--border); background: var(--surface);
  color: var(--text2); transition: all .15s;
}
.chip.active { background: var(--accent); border-color: var(--accent); color: white; }
.chip:hover:not(.active) { border-color: var(--accent); color: var(--text); }

.sb-toggle {
  display: flex; align-items: center; gap: 8px; cursor: pointer;
  font-size: 12px; color: var(--text2);
}
.sb-toggle input { display: none; }
.toggle-track {
  width: 36px; height: 20px; border-radius: 10px; background: var(--border);
  position: relative; transition: background .2s;
}
.toggle-track::after {
  content: ''; position: absolute; top: 2px; left: 2px;
  width: 16px; height: 16px; border-radius: 50%; background: white;
  transition: transform .2s;
}
.sb-toggle input:checked + .toggle-track { background: var(--green); }
.sb-toggle input:checked + .toggle-track::after { transform: translateX(16px); }

/* ── Header ── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 16px; border-bottom: 1px solid var(--border); margin-bottom: 20px;
}
.topbar-left { display: flex; align-items: center; gap: 12px; }
.status-dot {
  width: 8px; height: 8px; border-radius: 50%; background: var(--green);
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.4)} }
#status-txt { font-size: 12px; color: var(--text2); font-family: var(--mono); }
#clock { font-family: var(--mono); font-size: 12px; color: var(--muted); background: var(--surface); padding: 5px 12px; border-radius: 6px; border: 1px solid var(--border); }

/* ── KPI Grid ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px; margin-bottom: 20px;
}
.kpi {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px 18px; position: relative; overflow: hidden;
  transition: all .2s;
}
.kpi:hover { transform: translateY(-2px); border-color: var(--accent); box-shadow: var(--shadow); }
.kpi::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--kpi-c, var(--accent));
}
.kpi-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 800; line-height: 1; letter-spacing: -1px; }
.kpi-sub { font-size: 11px; color: var(--text2); margin-top: 4px; font-family: var(--mono); }
.kpi-delta { display: inline-flex; align-items: center; gap: 3px; font-size: 11px; font-weight: 600; margin-top: 4px; }
.up { color: var(--green); } .down { color: var(--red); } .flat { color: var(--muted); }

/* ── Stats Bar ── */
.stats-bar {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 8px; margin-bottom: 20px;
}
.stat-pill {
  background: var(--surface2); border: 1px solid var(--border); border-radius: 8px;
  padding: 10px 14px; text-align: center;
}
.stat-pill .label { font-size: 9px; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
.stat-pill .value { font-size: 18px; font-weight: 700; margin-top: 2px; font-family: var(--mono); }

/* ── Cards & Charts ── */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
.grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; margin-bottom: 16px; }
@media (max-width: 1100px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }

.card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; box-shadow: var(--shadow);
}
.card-title {
  font-size: 12px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1px; color: var(--muted); margin-bottom: 16px;
  display: flex; align-items: center; gap: 8px;
}
.card-title span { font-size: 15px; }
.chart-wrap { position: relative; height: 260px; }
.chart-wrap.tall { height: 320px; }

/* ── Table ── */
table { width: 100%; border-collapse: collapse; }
thead th {
  font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
  color: var(--muted); padding: 8px 12px; text-align: left;
  border-bottom: 1px solid var(--border); position: sticky; top: 0; background: var(--surface);
}
tbody tr { border-bottom: 1px solid rgba(30,45,69,.4); transition: background .15s; }
tbody tr:hover { background: rgba(59,130,246,.04); }
tbody tr.anomaly { background: rgba(239,68,68,.06); }
tbody td { padding: 10px 12px; font-size: 12px; font-family: var(--mono); }
tbody td:first-child { font-family: var(--font); font-weight: 600; }

.badge {
  display: inline-block; padding: 2px 8px; border-radius: 20px;
  font-size: 10px; font-weight: 700;
}
.badge-green  { background: rgba(16,185,129,.12); color: var(--green); }
.badge-yellow { background: rgba(245,158,11,.12);  color: var(--yellow); }
.badge-red    { background: rgba(239,68,68,.12);   color: var(--red); }
.badge-purple { background: rgba(139,92,246,.12); color: var(--purple); }

.bar-w { width: 70px; height: 5px; background: rgba(255,255,255,.06); border-radius: 3px; display: inline-block; vertical-align: middle; overflow: hidden; }
.bar-f { height: 100%; border-radius: 3px; }

/* ── Anomaly Panel ── */
.anomaly-list { max-height: 200px; overflow-y: auto; }
.anomaly-item {
  display: flex; align-items: center; gap: 10px;
  padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,.04); font-size: 12px;
}
.anomaly-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

/* ── Compare overlay ── */
.compare-info {
  background: var(--surface2); border: 1px solid var(--purple);
  border-radius: 8px; padding: 12px 16px; margin-bottom: 16px;
  display: none; font-size: 12px;
}
.compare-info.visible { display: flex; align-items: center; gap: 12px; }

/* ── Empty state ── */
.empty { text-align: center; padding: 60px 20px; color: var(--muted); font-size: 14px; }
.empty .icon { font-size: 48px; margin-bottom: 12px; opacity: .3; }

/* ── Footer ── */
footer {
  text-align: center; padding: 16px 0; font-size: 11px; color: var(--muted);
  border-top: 1px solid var(--border); margin-top: 20px;
}

/* ── Animations ── */
@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }
.kpi, .card, .stat-pill { animation: fadeUp .35s ease both; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border2); }
</style>
</head>
<body>
<div class="app">

  <!-- ═══ SIDEBAR ═══ -->
  <aside class="sidebar">
    <div class="sb-logo">
      <div class="sb-logo-icon">🚌</div>
      <div><h1>Dashboard</h1><span>Ulaşım Operasyon v2</span></div>
    </div>

    <div class="sb-section">
      <div class="sb-label">Tarih Aralığı</div>
      <input type="date" id="f-baslangic" class="sb-input">
      <input type="date" id="f-bitis" class="sb-input">
      <div style="display:flex;gap:4px">
        <button class="sb-btn small secondary" onclick="hizliFiltre(7)">7G</button>
        <button class="sb-btn small secondary" onclick="hizliFiltre(14)">14G</button>
        <button class="sb-btn small secondary" onclick="hizliFiltre(30)">30G</button>
        <button class="sb-btn small secondary" onclick="hizliFiltre(0)">Tümü</button>
      </div>
      <button class="sb-btn" onclick="veriGuncelle()">Filtrele</button>
    </div>

    <div class="sb-section">
      <div class="sb-label">Aktif Metrik</div>
      <div class="sb-chips" id="metrik-chips">
        <div class="chip active" data-m="doluluk" onclick="metrikSec(this)">Doluluk</div>
        <div class="chip" data-m="yolcu" onclick="metrikSec(this)">Yolcu</div>
        <div class="chip" data-m="arac" onclick="metrikSec(this)">Araç</div>
        <div class="chip" data-m="kapasite" onclick="metrikSec(this)">Kapasite</div>
      </div>
    </div>

    <div class="sb-section">
      <div class="sb-label">Karşılaştırma</div>
      <label class="sb-toggle">
        <input type="checkbox" id="compare-toggle" onchange="karsilastirToggle()">
        <div class="toggle-track"></div>
        Dönem Karşılaştır
      </label>
      <input type="date" id="c-baslangic" class="sb-input" style="display:none" placeholder="2. dönem başlangıç">
      <input type="date" id="c-bitis" class="sb-input" style="display:none" placeholder="2. dönem bitiş">
    </div>

    <div class="sb-section">
      <div class="sb-label">Ayarlar</div>
      <label class="sb-toggle">
        <input type="checkbox" id="auto-refresh" checked onchange="refreshToggle()">
        <div class="toggle-track"></div>
        Otomatik Yenile (60s)
      </label>
      <label class="sb-toggle">
        <input type="checkbox" id="anomaly-toggle" checked onchange="veriGuncelle()">
        <div class="toggle-track"></div>
        Anomali İşaretleri
      </label>
    </div>

    <div class="sb-section" style="margin-top:auto">
      <button class="sb-btn secondary" onclick="csvIndir()">📥 CSV İndir</button>
    </div>
  </aside>

  <!-- ═══ MAIN ═══ -->
  <div class="main">
    <!-- Top bar -->
    <div class="topbar">
      <div class="topbar-left">
        <div class="status-dot" id="status-dot"></div>
        <span id="status-txt">Yükleniyor…</span>
      </div>
      <div id="clock">--:--:--</div>
    </div>

    <!-- KPI -->
    <div class="kpi-grid" id="kpi-grid">
      <div class="kpi" style="--kpi-c:#3b82f6"><div class="kpi-label">Aktif Araç</div><div class="kpi-value" id="k-arac">—</div><div class="kpi-sub" id="k-arac-sub">—</div></div>
      <div class="kpi" style="--kpi-c:#10b981"><div class="kpi-label">Doluluk</div><div class="kpi-value" id="k-dol">—</div><div class="kpi-delta" id="k-dol-delta">—</div></div>
      <div class="kpi" style="--kpi-c:#f59e0b"><div class="kpi-label">Toplam Yolcu</div><div class="kpi-value" id="k-yolcu">—</div><div class="kpi-sub" id="k-yolcu-sub">—</div></div>
      <div class="kpi" style="--kpi-c:#8b5cf6"><div class="kpi-label">Gün Sayısı</div><div class="kpi-value" id="k-gun">—</div><div class="kpi-sub">filtrelenen</div></div>
      <div class="kpi" style="--kpi-c:#06b6d4"><div class="kpi-label">Ort. Doluluk</div><div class="kpi-value" id="k-ort">—</div><div class="kpi-sub" id="k-ort-sub">—</div></div>
      <div class="kpi" style="--kpi-c:#ec4899"><div class="kpi-label">Anomali</div><div class="kpi-value" id="k-anomali">—</div><div class="kpi-sub">tespit edilen</div></div>
    </div>

    <!-- Stats bar -->
    <div class="stats-bar" id="stats-bar"></div>

    <!-- Compare info -->
    <div class="compare-info" id="compare-info">
      <span style="color:var(--purple);font-weight:700">⚡ Karşılaştırma Modu</span>
      <span id="compare-desc">—</span>
    </div>

    <!-- Charts row 1 -->
    <div class="grid-2">
      <div class="card">
        <div class="card-title"><span>📈</span> Ana Trend</div>
        <div class="chart-wrap tall"><canvas id="trendChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title"><span>📊</span> Dağılım</div>
        <div class="chart-wrap tall"><canvas id="barChart"></canvas></div>
      </div>
    </div>

    <!-- Charts row 2 -->
    <div class="grid-3">
      <div class="card">
        <div class="card-title"><span>🎯</span> Doluluk vs Araç</div>
        <div class="chart-wrap"><canvas id="scatterChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title"><span>📉</span> Günlük Değişim</div>
        <div class="chart-wrap"><canvas id="deltaChart"></canvas></div>
      </div>
      <div class="card">
        <div class="card-title"><span>🔍</span> Anomaliler</div>
        <div class="anomaly-list" id="anomaly-list">
          <div class="empty"><div class="icon">🔍</div>Veri bekleniyor…</div>
        </div>
      </div>
    </div>

    <!-- Table -->
    <div class="card">
      <div class="card-title"><span>📋</span> Detay Tablosu</div>
      <div id="tablo-wrap" style="max-height:400px;overflow-y:auto"></div>
    </div>

    <footer>Dashboard v2  •  <span id="footer-tarih">—</span></footer>
  </div>
</div>

<script>
// ═══════════════════════════════════════════
//  STATE
// ═══════════════════════════════════════════
let STATE = {
  gunler: [],
  anomaliler: [],
  istatistik: {},
  metrik: 'doluluk',
  compare: false,
  compareGunler: [],
  refreshTimer: null,
};

const METRIK_CONF = {
  doluluk:  { label: 'Doluluk (%)', color: '#10b981', birim: '%', max: 110 },
  yolcu:    { label: 'Yolcu',       color: '#f59e0b', birim: '',  max: null },
  arac:     { label: 'Araç',        color: '#3b82f6', birim: '',  max: null },
  kapasite: { label: 'Kapasite',    color: '#8b5cf6', birim: '',  max: null },
};

// ═══════════════════════════════════════════
//  CLOCK
// ═══════════════════════════════════════════
function saatGuncelle() {
  document.getElementById('clock').textContent =
    new Date().toLocaleTimeString('tr-TR', {hour:'2-digit',minute:'2-digit',second:'2-digit'});
}
setInterval(saatGuncelle, 1000); saatGuncelle();

// ═══════════════════════════════════════════
//  SIDEBAR ACTIONS
// ═══════════════════════════════════════════
function hizliFiltre(gun) {
  if (gun === 0) {
    document.getElementById('f-baslangic').value = '';
    document.getElementById('f-bitis').value = '';
  } else {
    const end = new Date();
    const start = new Date(); start.setDate(start.getDate() - gun);
    document.getElementById('f-baslangic').value = start.toISOString().slice(0,10);
    document.getElementById('f-bitis').value = end.toISOString().slice(0,10);
  }
  veriGuncelle();
}

function metrikSec(el) {
  document.querySelectorAll('#metrik-chips .chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  STATE.metrik = el.dataset.m;
  grafikleriGuncelle();
  istatistikGuncelle();
}

function karsilastirToggle() {
  STATE.compare = document.getElementById('compare-toggle').checked;
  const s = STATE.compare ? 'block' : 'none';
  document.getElementById('c-baslangic').style.display = s;
  document.getElementById('c-bitis').style.display = s;
  if (STATE.compare) veriGuncelle();
  else {
    STATE.compareGunler = [];
    document.getElementById('compare-info').classList.remove('visible');
    grafikleriGuncelle();
  }
}

function refreshToggle() {
  if (document.getElementById('auto-refresh').checked) {
    STATE.refreshTimer = setInterval(veriGuncelle, 60000);
  } else {
    clearInterval(STATE.refreshTimer);
    STATE.refreshTimer = null;
  }
}

// ═══════════════════════════════════════════
//  CHARTS INIT
// ═══════════════════════════════════════════
const CHART_DEFAULTS = {
  responsive: true, maintainAspectRatio: false,
  plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } } } },
  scales: {
    x: { ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }, grid: { color: '#1e2d4a' } },
    y: { ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 9 } }, grid: { color: '#1e2d4a' } },
  }
};

// Trend
const trendChart = new Chart(document.getElementById('trendChart'), {
  type: 'line',
  data: { labels: [], datasets: [] },
  options: { ...CHART_DEFAULTS,
    interaction: { mode: 'index', intersect: false },
    plugins: { ...CHART_DEFAULTS.plugins,
      tooltip: { backgroundColor: '#151b2e', borderColor: '#293d5a', borderWidth: 1,
        titleFont: { family: 'JetBrains Mono' }, bodyFont: { family: 'JetBrains Mono' } }
    }
  }
});

// Bar
const barChart = new Chart(document.getElementById('barChart'), {
  type: 'bar',
  data: { labels: [], datasets: [] },
  options: { ...CHART_DEFAULTS }
});

// Scatter
const scatterChart = new Chart(document.getElementById('scatterChart'), {
  type: 'scatter',
  data: { datasets: [] },
  options: { ...CHART_DEFAULTS,
    scales: {
      x: { ...CHART_DEFAULTS.scales.x, title: { display: true, text: 'Araç Sayısı', color: '#64748b', font: {size:10} } },
      y: { ...CHART_DEFAULTS.scales.y, title: { display: true, text: 'Doluluk (%)', color: '#64748b', font: {size:10} } },
    },
    plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } }
  }
});

// Delta
const deltaChart = new Chart(document.getElementById('deltaChart'), {
  type: 'bar',
  data: { labels: [], datasets: [] },
  options: { ...CHART_DEFAULTS,
    plugins: { ...CHART_DEFAULTS.plugins, legend: { display: false } }
  }
});

// ═══════════════════════════════════════════
//  DATA FETCH
// ═══════════════════════════════════════════
async function veriGuncelle() {
  try {
    const b = document.getElementById('f-baslangic').value;
    const e = document.getElementById('f-bitis').value;
    let url = '/api/veri?';
    if (b) url += `baslangic=${b}&`;
    if (e) url += `bitis=${e}&`;

    const res = await fetch(url);
    const data = await res.json();
    if (!data.ok || !data.gunler.length) {
      document.getElementById('status-txt').textContent = '⚠ Veri bulunamadı.';
      return;
    }

    STATE.gunler = data.gunler;
    STATE.anomaliler = data.anomaliler || [];
    STATE.istatistik = data.istatistik || {};

    // Compare
    if (STATE.compare) {
      const cb = document.getElementById('c-baslangic').value;
      const ce = document.getElementById('c-bitis').value;
      if (cb && ce) {
        const res2 = await fetch(`/api/veri?baslangic=${cb}&bitis=${ce}`);
        const d2 = await res2.json();
        STATE.compareGunler = d2.ok ? d2.gunler : [];
        document.getElementById('compare-info').classList.add('visible');
        document.getElementById('compare-desc').textContent =
          `Dönem 1: ${b||'başlangıç'}→${e||'son'} | Dönem 2: ${cb}→${ce}`;
      }
    }

    kpiGuncelle();
    istatistikGuncelle();
    grafikleriGuncelle();
    anomaliPanelGuncelle();
    tabloGuncelle();

    document.getElementById('status-txt').textContent =
      `✓ ${data.gunler.length} gün yüklendi`;
    document.getElementById('footer-tarih').textContent =
      `Son güncelleme: ${new Date().toLocaleTimeString('tr-TR')}`;

  } catch(err) {
    document.getElementById('status-txt').textContent = '❌ Hata: ' + err.message;
  }
}

// ═══════════════════════════════════════════
//  KPI UPDATE
// ═══════════════════════════════════════════
function kpiGuncelle() {
  const g = STATE.gunler;
  if (!g.length) return;
  const son = g[g.length-1];
  const onceki = g.length > 1 ? g[g.length-2] : null;

  document.getElementById('k-arac').textContent = son.arac;
  document.getElementById('k-arac-sub').textContent = son.tarih;
  document.getElementById('k-dol').textContent = `%${son.doluluk.toFixed(1)}`;
  document.getElementById('k-yolcu').textContent = son.yolcu.toLocaleString('tr-TR');
  document.getElementById('k-yolcu-sub').textContent = `${son.kapasite.toLocaleString('tr-TR')} koltuk`;
  document.getElementById('k-gun').textContent = g.length;

  // Ort doluluk
  const ortDol = g.reduce((s,x) => s+x.doluluk, 0) / g.length;
  document.getElementById('k-ort').textContent = `%${ortDol.toFixed(1)}`;
  const minDol = Math.min(...g.map(x=>x.doluluk));
  const maxDol = Math.max(...g.map(x=>x.doluluk));
  document.getElementById('k-ort-sub').textContent = `min %${minDol.toFixed(0)} / max %${maxDol.toFixed(0)}`;

  // Anomali sayısı
  document.getElementById('k-anomali').textContent = STATE.anomaliler.length;

  // Delta
  const de = document.getElementById('k-dol-delta');
  if (onceki) {
    const fark = (son.doluluk - onceki.doluluk).toFixed(1);
    const cls = fark > 0 ? 'up' : fark < 0 ? 'down' : 'flat';
    const ok = fark > 0 ? '▲' : fark < 0 ? '▼' : '—';
    de.className = `kpi-delta ${cls}`;
    de.textContent = `${ok} ${Math.abs(fark)}% önceki güne göre`;
  }
}

// ═══════════════════════════════════════════
//  STATS BAR
// ═══════════════════════════════════════════
function istatistikGuncelle() {
  const m = STATE.metrik;
  const g = STATE.gunler;
  if (!g.length) return;

  const vals = g.map(x => x[m]);
  const stats = {
    Min: Math.min(...vals).toFixed(1),
    Max: Math.max(...vals).toFixed(1),
    Ortalama: (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(1),
    Medyan: vals.sort((a,b)=>a-b)[Math.floor(vals.length/2)].toFixed(1),
    'Std Sapma': stdDev(vals).toFixed(1),
    Trend: trendHesapla(vals),
  };

  const conf = METRIK_CONF[m];
  let html = '';
  for (const [k, v] of Object.entries(stats)) {
    const renk = k === 'Trend' ? (v === '↑' ? 'var(--green)' : v === '↓' ? 'var(--red)' : 'var(--muted)') : 'var(--text)';
    html += `<div class="stat-pill"><div class="label">${conf.label} ${k}</div><div class="value" style="color:${renk}">${v}${k!=='Trend'?conf.birim:''}</div></div>`;
  }
  document.getElementById('stats-bar').innerHTML = html;
}

function stdDev(arr) {
  const n = arr.length;
  const m = arr.reduce((a,b)=>a+b,0)/n;
  return Math.sqrt(arr.reduce((s,v) => s + (v-m)**2, 0) / n);
}
function trendHesapla(arr) {
  if (arr.length < 4) return '→';
  const ilk = arr.slice(0,3).reduce((a,b)=>a+b,0)/3;
  const son = arr.slice(-3).reduce((a,b)=>a+b,0)/3;
  return (son-ilk) > 2 ? '↑' : (son-ilk) < -2 ? '↓' : '→';
}

// ═══════════════════════════════════════════
//  CHARTS UPDATE
// ═══════════════════════════════════════════
function grafikleriGuncelle() {
  const g = STATE.gunler;
  if (!g.length) return;
  const m = STATE.metrik;
  const conf = METRIK_CONF[m];
  const labels = g.map(x => x.tarih.slice(5).replace('-','/'));
  const vals = g.map(x => x[m]);

  const showAnomaly = document.getElementById('anomaly-toggle').checked;
  const anomaliTarihler = new Set(STATE.anomaliler.filter(a => a.metrik === m).map(a => a.tarih));

  // Punkt renkleri — anomali noktaları kırmızı
  const ptColors = g.map(x => anomaliTarihler.has(x.tarih) && showAnomaly ? '#ef4444' : conf.color);
  const ptRadius = g.map(x => anomaliTarihler.has(x.tarih) && showAnomaly ? 7 : 4);

  // ── Trend ──
  const datasets = [{
    label: conf.label, data: vals, borderColor: conf.color,
    backgroundColor: conf.color + '15', tension: 0.4, fill: true,
    pointBackgroundColor: ptColors, pointRadius: ptRadius,
    pointHoverRadius: 9, borderWidth: 2,
  }];

  // Compare dataset
  if (STATE.compare && STATE.compareGunler.length) {
    const cLabels = STATE.compareGunler.map(x => x.tarih.slice(5).replace('-','/'));
    datasets.push({
      label: `${conf.label} (Karşılaştırma)`,
      data: STATE.compareGunler.map(x => x[m]),
      borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,.08)',
      tension: 0.4, fill: false, borderDash: [6, 3],
      pointRadius: 4, pointBackgroundColor: '#8b5cf6', borderWidth: 2,
    });
    // Etiketler: en uzun olanı kullan
    if (cLabels.length > labels.length) {
      for (let i = labels.length; i < cLabels.length; i++) labels.push(cLabels[i]);
    }
  }

  trendChart.data.labels = labels;
  trendChart.data.datasets = datasets;
  if (conf.max) trendChart.options.scales.y.max = conf.max;
  else delete trendChart.options.scales.y.max;
  trendChart.update('active');

  // ── Bar ──
  const barColors = vals.map(v => {
    if (m === 'doluluk') return v >= 80 ? '#10b98190' : v >= 60 ? '#f59e0b90' : '#ef444490';
    return conf.color + '90';
  });
  barChart.data.labels = labels;
  barChart.data.datasets = [{
    label: conf.label, data: vals, backgroundColor: barColors,
    borderRadius: 4, borderSkipped: false,
  }];
  barChart.update('active');

  // ── Scatter (doluluk vs araç — her zaman) ──
  const scatterData = g.map(x => ({x: x.arac, y: x.doluluk}));
  const scColors = g.map(x => anomaliTarihler.has(x.tarih) && showAnomaly ? '#ef4444' : '#3b82f6');
  scatterChart.data.datasets = [{
    data: scatterData, backgroundColor: scColors,
    pointRadius: 7, pointHoverRadius: 10,
  }];
  scatterChart.update('active');

  // ── Delta (günlük değişim) ──
  const deltas = [0];
  for (let i = 1; i < vals.length; i++) deltas.push(vals[i] - vals[i-1]);
  const deltaColors = deltas.map(d => d > 0 ? '#10b98180' : d < 0 ? '#ef444480' : '#64748b80');
  deltaChart.data.labels = labels;
  deltaChart.data.datasets = [{
    data: deltas, backgroundColor: deltaColors,
    borderRadius: 3, borderSkipped: false,
  }];
  deltaChart.update('active');
}

// ═══════════════════════════════════════════
//  ANOMALY PANEL
// ═══════════════════════════════════════════
function anomaliPanelGuncelle() {
  const list = document.getElementById('anomaly-list');
  if (!STATE.anomaliler.length) {
    list.innerHTML = '<div style="text-align:center;padding:24px;color:var(--muted);font-size:12px">✅ Anomali tespit edilmedi</div>';
    return;
  }
  let html = '';
  STATE.anomaliler.forEach(a => {
    const renk = a.yon === 'yüksek' ? 'var(--red)' : 'var(--yellow)';
    html += `<div class="anomaly-item">
      <div class="anomaly-dot" style="background:${renk}"></div>
      <div>
        <div style="font-weight:600">${a.tarih} — ${a.metrik}</div>
        <div style="color:var(--muted);font-size:11px">${a.yon} (z=${a.z_skor}) → ${a.deger}</div>
      </div>
    </div>`;
  });
  list.innerHTML = html;
}

// ═══════════════════════════════════════════
//  TABLE
// ═══════════════════════════════════════════
function tabloGuncelle() {
  const g = STATE.gunler;
  if (!g.length) return;
  const anomaliSet = new Set(STATE.anomaliler.map(a => a.tarih));

  let html = `<table><thead><tr>
    <th>Tarih</th><th>Araç</th><th>Yolcu</th><th>Kapasite</th><th>Doluluk</th><th>Değişim</th><th>Durum</th>
  </tr></thead><tbody>`;

  ;[...g].reverse().forEach((r, idx) => {
    const dol = r.doluluk;
    const onceki = idx < g.length-1 ? [...g].reverse()[idx+1] : null;
    let degisim = '—';
    if (onceki) {
      const f = (dol - onceki.doluluk).toFixed(1);
      degisim = `${f > 0 ? '▲':'▼'} ${Math.abs(f)}%`;
    }
    const bc = dol >= 80 ? 'badge-green' : dol >= 60 ? 'badge-yellow' : 'badge-red';
    const bt = dol >= 80 ? '✔ İyi' : dol >= 60 ? '⚠ Düşük' : '✖ Kritik';
    const barC = dol >= 80 ? '#10b981' : dol >= 60 ? '#f59e0b' : '#ef4444';
    const isAnomali = anomaliSet.has(r.tarih);

    html += `<tr class="${isAnomali?'anomaly':''}">
      <td>${r.tarih} ${isAnomali?'<span class="badge badge-purple">ANM</span>':''}</td>
      <td>${r.arac}</td>
      <td>${r.yolcu.toLocaleString('tr-TR')}</td>
      <td>${r.kapasite.toLocaleString('tr-TR')}</td>
      <td><span class="bar-w"><span class="bar-f" style="width:${Math.min(dol,100)}%;background:${barC}"></span></span> %${dol.toFixed(1)}</td>
      <td style="color:${degisim.includes('▲')?'var(--green)':'var(--red)'}">${degisim}</td>
      <td><span class="badge ${bc}">${bt}</span></td>
    </tr>`;
  });
  html += '</tbody></table>';
  document.getElementById('tablo-wrap').innerHTML = html;
}

// ═══════════════════════════════════════════
//  CSV EXPORT
// ═══════════════════════════════════════════
function csvIndir() {
  const b = document.getElementById('f-baslangic').value;
  const e = document.getElementById('f-bitis').value;
  let url = '/api/export?';
  if (b) url += `baslangic=${b}&`;
  if (e) url += `bitis=${e}&`;
  window.location.href = url;
}

// ═══════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════
veriGuncelle();
STATE.refreshTimer = setInterval(veriGuncelle, 60000);
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/veri")
def api_veri():
    """Ana veri endpoint'i — tarih filtresi + istatistik + anomali."""
    try:
        raw = _veri_oku()
        if not raw:
            return jsonify({"ok": False, "hata": "Veri yok.", "gunler": []})

        baslangic = request.args.get("baslangic")
        bitis = request.args.get("bitis")
        metrik = request.args.get("metrik", "doluluk")

        gunler = _gunler_listesi(raw, baslangic, bitis)
        if not gunler:
            return jsonify({"ok": False, "hata": "Filtrede veri yok.", "gunler": []})

        ist = _istatistik(gunler, metrik)
        anomaliler = _anomali_basit(gunler)

        return jsonify({
            "ok": True,
            "gunler": gunler,
            "toplam": len(gunler),
            "istatistik": ist,
            "anomaliler": anomaliler,
        })
    except Exception as e:
        return jsonify({"ok": False, "hata": str(e), "gunler": []})


@app.route("/api/son")
def api_son():
    """En son günün verisini döndürür."""
    try:
        raw = _veri_oku()
        if not raw:
            return jsonify({"ok": False, "hata": "Veri yok."})
        son_tarih = sorted(raw.keys())[-1]
        return jsonify({"ok": True, "tarih": son_tarih, **raw[son_tarih]})
    except Exception as e:
        return jsonify({"ok": False, "hata": str(e)})


@app.route("/api/istatistik")
def api_istatistik():
    """Belirtilen metrik için istatistik."""
    try:
        raw = _veri_oku()
        baslangic = request.args.get("baslangic")
        bitis = request.args.get("bitis")
        metrik = request.args.get("metrik", "doluluk")

        gunler = _gunler_listesi(raw, baslangic, bitis)
        ist = _istatistik(gunler, metrik)
        return jsonify({"ok": True, **ist})
    except Exception as e:
        return jsonify({"ok": False, "hata": str(e)})


@app.route("/api/anomali")
def api_anomali():
    """Anomali tespiti endpoint'i."""
    try:
        raw = _veri_oku()
        baslangic = request.args.get("baslangic")
        bitis = request.args.get("bitis")

        gunler = _gunler_listesi(raw, baslangic, bitis)
        anomaliler = _anomali_basit(gunler)
        return jsonify({"ok": True, "anomaliler": anomaliler, "toplam": len(anomaliler)})
    except Exception as e:
        return jsonify({"ok": False, "hata": str(e)})


@app.route("/api/export")
def api_export():
    """CSV olarak veri indirme."""
    try:
        raw = _veri_oku()
        baslangic = request.args.get("baslangic")
        bitis = request.args.get("bitis")

        gunler = _gunler_listesi(raw, baslangic, bitis)
        if not gunler:
            return Response("Veri yok.", mimetype="text/plain")

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["tarih", "arac", "yolcu", "kapasite", "doluluk"])
        writer.writeheader()
        for g in gunler:
            writer.writerow(g)

        tarih_str = datetime.now().strftime("%Y%m%d_%H%M")
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename=ulasim_veri_{tarih_str}.csv"},
        )
    except Exception as e:
        return Response(f"Hata: {e}", mimetype="text/plain")


# ══════════════════════════════════════════════════════════════════════════════
#  ÇALIŞTIR
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 50)
    print("🌐 Ulaşım Dashboard v2 başlatılıyor...")
    print("   Aç: http://localhost:8080")
    print("   Durdurmak için: Ctrl+C")
    print("=" * 50)
    app.run(host="0.0.0.0", port=8080, debug=False)
