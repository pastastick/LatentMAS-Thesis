#!/usr/bin/env python3
"""Tabel & gambar Bab IV — versi MoI-unggulan (tanpa gumbel, tanpa lengan faktor).

Sumber kebenaran tunggal: skripsi/komputation/results/. Skrip ini TIDAK membaca
berkas antara di analisis/ supaya angka tak bisa melenceng dua langkah.

Keluaran:
  assets/tables/b4_*.tex   (10 tabel)
  assets/images/b4_*.png   (6 gambar, 300 dpi)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "komputation" / "results"
TAB = ROOT / "assets" / "tables"
IMG = ROOT / "assets" / "images"
TAB.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

WARNA = {"raw": "#e34948", "soft": "#2a78d6", "sample": "#eda100", "moi": "#4a3aa7"}
ABU, ABU_TUA = "#9a9a95", "#52514e"
BIRU_TUA = "#1f3b73"

MODE = ["raw", "soft", "sample", "moi"]                 # urutan formulasi KV
TUGAS = ["gsm8k", "arc_challenge", "humanevalplus"]
LABEL = {"gsm8k": "GSM8K", "arc_challenge": "ARC-C", "humanevalplus": "HumanEval+"}
NAMA = {"raw": r"\texttt{raw}", "soft": r"\texttt{soft}",
        "sample": r"\texttt{sample}", "moi": r"\texttt{moi}"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c9c9c4", "axes.labelcolor": "#0b0b0b",
    "xtick.color": ABU_TUA, "ytick.color": ABU_TUA,
    "axes.grid": True, "grid.color": "#e8e8e4", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150,
})


def koma(x, n=2):
    if x is None:
        return "---"
    return f"{x:.{n}f}".replace(".", ",")


# ───────────────────────── pemuatan data ────────────────────────────────────
def muat():
    """Kembalikan (sel, pasangan) untuk sel n=100 saja."""
    an = json.loads((RES / "bench" / "analisis.json").read_text())
    tok = json.loads((RES / "pendukung" / "token_bench.json").read_text())

    sel, pasangan = {}, {}
    for g in an["groups"]:
        if g["n_items"] != 100:
            continue
        t = g["task"]
        sel[t] = {}
        for c in g["cells"]:
            metode, medium, _ = c["cell"].split("/")
            if metode == "gumbel":
                continue
            kunci = "tunggal" if medium == "baseline" else ("teks" if medium == "text" else metode)
            sel[t][kunci] = {"akurasi": c["accuracy"], "format": c["format_rate"],
                             "waktu": c["time_s"]}
        pasangan[t] = [p for p in g["pairs"] if "gumbel" not in p["a"] + p["b"]]

    for r in tok:
        if r["n_soal"] != 100 or r["metode"] in ("gumbel", "mix"):
            continue
        t, medium, metode = r["task"], r["medium"], r["metode"]
        kunci = "tunggal" if medium == "baseline" else ("teks" if medium == "text" else metode)
        if kunci in sel[t]:
            sel[t][kunci].update({
                "tok_out": r["token_per_soal"], "tok_in": r["token_masukan"],
                "detik": r["detik_per_soal"], "cjk": r["n_jawaban_ber_cjk"]})
    return sel, pasangan


def geometri():
    d = json.loads((RES / "probe" / "b7_probe_Qwen_Qwen3-8B.json").read_text())
    return d["geometry"], d["geometry_mix"]


def kanal(nama):
    f = RES / "probe" / f"channel_capacity_Qwen_Qwen3-8B_{nama}.json"
    d = json.loads(f.read_text())
    out = {}
    for r in d["_summary"]:
        out[(r["arm"], r["payload"])] = (r["recall"], r["exact"])
    return out


# ───────────────────────── tabel ────────────────────────────────────────────
def tabel_tugas():
    baris = [
        (r"GSM8K \citep{gsm8k}", "Penalaran aritmetika", "1.319", "100",
         r"\textit{exact match}"),
        (r"ARC-Challenge \citep{arc}", "Pilihan ganda sains", "1.172", "100",
         r"\textit{exact match}"),
        (r"HumanEval+ \citep{evalplus}", "Pembangkitan kode", "164", "100",
         r"pass@1"),
    ]
    isi = "\n".join(" & ".join(b) + r" \\" for b in baris)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Tolok ukur yang digunakan (\code{{openai/gsm8k}}, \code{{allenai/ai2\_arc}},
dan \code{{evalplus/humanevalplus}}). Seratus soal pertama diambil dengan
\textit{{sample seed}} yang sama untuk seluruh perlakuan sehingga setiap
perbandingan bersifat berpasangan pada soal yang identik.}}
\label{{tab:b4-tugas}}
\small
\begin{{tabular}}{{llccl}}
\hline
Tolok ukur & Jenis keluaran & $n$ total & $n$ dipakai & Metrik \\
\hline
{isi}
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_konfig():
    baris = [
        (r"Model dasar", r"Qwen3-8B \citep{qwen3}"),
        (r"Jumlah lapisan / \textit{head} (Q/KV)", r"36 / 32 \, / \, 8 (GQA)"),
        (r"Dimensi \textit{hidden state} $d$", r"$4.096$"),
        (r"Ukuran kosakata $V$", r"$151.936$"),
        (r"\textit{Tied embedding}", r"tidak"),
        (r"Rantai agen", r"\code{planner} $\to$ \code{critic} $\to$ \code{refiner} $\to$ \code{judger}"),
        (r"Jumlah langkah laten $m$", r"$10$"),
        (r"Suhu langkah laten $T$", r"$0{,}7$"),
        (r"Suhu dekode teks $T_{\mathrm{gen}}$ / \textit{top-}$p$", r"$0{,}6$ / $0{,}95$"),
        (r"Parameter MoI $\beta$", r"$1$ (bawaan, tanpa penalaan)"),
        (r"Regularisasi \textit{ridge} $\lambda$", r"$10^{-5}$"),
        (r"Norma sasaran $\tau$", r"$1{,}3758$"),
        (r"\textit{Max new tokens}", r"$2.048$"),
        (r"\textit{Seed}", r"$0$"),
    ]
    isi = "\n".join(f"{a} & {b}" + r" \\" for a, b in baris)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Konfigurasi yang dijaga tetap di seluruh perlakuan. Tidak ada
pencarian parameter: setiap metode dijalankan pada setelan bawaan
publikasi asalnya.}}
\label{{tab:b4-konfig}}
\small
\begin{{tabular}}{{ll}}
\hline
Komponen & Nilai \\
\hline
{isi}
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_akurasi(sel):
    kol = ["tunggal", "teks", "raw", "soft", "sample", "moi"]
    baris = []
    for t in TUGAS:
        v = [sel[t][k]["akurasi"] for k in kol]
        best = max(v[2:])
        sel_str = []
        for i, (k, x) in enumerate(zip(kol, v)):
            s = koma(x)
            if i >= 2 and abs(x - best) < 1e-9:
                s = r"\textbf{" + s + "}"
            sel_str.append(s)
        baris.append(f"{LABEL[t]} & " + " & ".join(sel_str) + r" \\")
    for t in TUGAS:
        pass
    isi = "\n".join(baris)
    barisf = "\n".join(
        f"{LABEL[t]} & " + " & ".join(koma(sel[t][k]["format"]) for k in kol) + r" \\"
        for t in TUGAS)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Akurasi dan laju keluaran berformat benar pada tiga tolok ukur,
$n=100$ soal identik per tolok ukur. Kolom \textsc{{Tunggal}} adalah satu agen
tanpa kolaborasi, \textsc{{Teks}} adalah rantai empat agen yang bertukar teks,
dan empat kolom terakhir adalah rantai yang sama dengan pertukaran
\textit{{KV-cache}}.}}
\label{{tab:b4-akurasi}}
\small
\begin{{tabular}}{{lcccccc}}
\hline
 & \multicolumn{{2}}{{c}}{{Acuan}} & \multicolumn{{4}}{{c}}{{Medium \textit{{KV-cache}}}} \\
\cline{{2-3}}\cline{{4-7}}
Tolok ukur & Tunggal & Teks & {NAMA['raw']} & {NAMA['soft']} & {NAMA['sample']} & {NAMA['moi']} \\
\hline
\multicolumn{{7}}{{l}}{{\textit{{Akurasi}}}} \\
{isi}
\hline
\multicolumn{{7}}{{l}}{{\textit{{Laju format benar}}}} \\
{barisf}
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_biaya(sel):
    kol = ["tunggal", "teks", "raw", "soft", "sample", "moi"]

    def blok(medan, n):
        out = []
        for t in TUGAS:
            v = []
            for k in kol:
                x = sel[t][k].get(medan)
                v.append("---" if x is None else koma(x, n))
            out.append(f"{LABEL[t]} & " + " & ".join(v) + r" \\")
        return "\n".join(out)

    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Biaya komputasi per soal. \textit{{Token}} keluaran dihitung atas
seluruh agen dalam rantai; waktu adalah waktu dinding per soal pada satu GPU.
Sel \textsc{{Tunggal}} GSM8K tidak menyimpan rincian \textit{{token}} per agen.}}
\label{{tab:b4-biaya}}
\small
\begin{{tabular}}{{lcccccc}}
\hline
 & \multicolumn{{2}}{{c}}{{Acuan}} & \multicolumn{{4}}{{c}}{{Medium \textit{{KV-cache}}}} \\
\cline{{2-3}}\cline{{4-7}}
Tolok ukur & Tunggal & Teks & {NAMA['raw']} & {NAMA['soft']} & {NAMA['sample']} & {NAMA['moi']} \\
\hline
\multicolumn{{7}}{{l}}{{\textit{{Token}} keluaran per soal}} \\
{blok('tok_out', 1)}
\hline
\multicolumn{{7}}{{l}}{{Waktu per soal (detik)}} \\
{blok('detik', 1)}
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_efisiensi(sel):
    baris = []
    for t in TUGAS:
        teks = sel[t]["teks"]
        for m in MODE:
            c = sel[t][m]
            hemat = 100 * (1 - c["tok_out"] / teks["tok_out"])
            cepat = teks["detik"] / c["detik"]
            baris.append(f"{LABEL[t]} & {NAMA[m]} & {koma(c['akurasi'])} & "
                         f"{koma(hemat, 1)}\\% & $\\times{koma(cepat, 1)}$" + r" \\")
        baris.append(r"\hline")
    isi = "\n".join(baris[:-1])
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Efisiensi medium \textit{{KV-cache}} relatif terhadap medium teks,
per persamaan langkah laten. Penghematan dihitung atas \textit{{token}} keluaran
per soal, percepatan atas waktu per soal.}}
\label{{tab:b4-efisiensi}}
\small
\begin{{tabular}}{{llccc}}
\hline
Tolok ukur & Langkah laten & Akurasi & Penghematan \textit{{token}} & Percepatan \\
\hline
{isi}
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_selisih(pasangan):
    """Selisih akurasi moi/kv terhadap tiga acuan, berpasangan pada soal sama."""
    acuan = [("raw/kv/m10", r"\texttt{raw} (KV)"),
             ("raw/text/m10", "Teks"),
             ("raw/baseline/m10", "Tunggal")]
    baris = []
    for t in TUGAS:
        for kunci, nama in acuan:
            for p in pasangan[t]:
                if {p["a"], p["b"]} == {"moi/kv/m10", kunci}:
                    c = p["correct"]
                    tanda = 1 if p["a"] == "moi/kv/m10" else -1
                    d = tanda * c["delta"]
                    lo, hi = sorted([tanda * c["ci95"][0], tanda * c["ci95"][1]])
                    pv = c["p_mcnemar"]
                    ps = "$<0{,}0001$" if pv < 1e-4 else f"${koma(pv, 4)}$"
                    baris.append(
                        f"{LABEL[t]} & {nama} & ${'+' if d >= 0 else '-'}{koma(abs(d))}$ & "
                        f"$[{'+' if lo >= 0 else '-'}{koma(abs(lo))}; "
                        f"{'+' if hi >= 0 else '-'}{koma(abs(hi))}]$ & {ps}" + r" \\")
                    break
        baris.append(r"\hline")
    isi = "\n".join(baris[:-1])
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Selisih akurasi \texttt{{moi}} pada medium \textit{{KV-cache}} terhadap
tiga acuan, dihitung berpasangan pada soal yang sama. Selang kepercayaan
\textit{{bootstrap}} 95\% dan nilai $p$ uji McNemar eksak.}}
\label{{tab:b4-selisih}}
\small
\begin{{tabular}}{{llccc}}
\hline
Tolok ukur & Acuan & $\Delta$ akurasi & SK 95\% & $p$ \\
\hline
{isi}
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_geometri(geo):
    bentuk = {"raw": r"$\tau\, hM/\lVert hM\rVert$ (bukan $w^\top W_{\mathrm{in}}$)",
              "soft": r"$w=p$",
              "sample": r"$w=e_y$",
              "moi": r"$w=\bigl(\tilde H p+(\beta{+}1{-}\tilde H)e_y\bigr)/(\beta{+}1)$"}
    hull = {"raw": "tidak", "soft": "ya", "sample": "ya", "moi": "ya"}
    baris = []
    for m in MODE:
        g = geo[m]
        baris.append(
            f"{NAMA[m]} & {bentuk[m]} & {hull[m]} & {koma(g['max_cos_embed_mean'], 3)} & "
            f"$[{koma(g['max_cos_embed_min'], 3)};\\ {koma(g['max_cos_embed_max'], 3)}]$" + r" \\")
    isi = "\n".join(baris)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Geometri vektor langkah laten pada Qwen3-8B, diukur atas $m=10$
langkah. Kolom terakhir adalah kosinus terhadap baris matriks
\textit{{embedding}} masukan yang paling dekat; nilai $1$ berarti vektor tepat
berimpit dengan \textit{{embedding}} sebuah token nyata.}}
\label{{tab:b4-geometri}}
\footnotesize
\begin{{tabular}}{{llccc}}
\hline
Langkah laten & Vektor yang disisipkan & Konveks & $\max_i\cos$ & Rentang \\
\hline
{isi}
\hline
\multicolumn{{5}}{{l}}{{\footnotesize $\lVert M-I\rVert_F/\lVert I\rVert_F=1{{,}}04$;\quad
$\overline{{\cos(h,hM)}}=0{{,}}0112$;\quad
$\overline{{\cos(W_{{\mathrm{{in}}}}[i],W_{{\mathrm{{out}}}}[i])}}=0{{,}}0041$.}} \\
\end{{tabular}}
\end{{table}}
"""


def tabel_ambang(geo, geomix, sel):
    """Sumbu interpolasi raw->soft: geometri vs akurasi HumanEval+."""
    bench = json.loads((RES / "bench" / "analisis.json").read_text())
    mixacc = {}
    for a, f in [(0.25, "a025"), (0.5, "a05"), (0.75, "a075")]:
        d = json.loads((RES / "bench" / f"bench_humanevalplus_mix_kv_s0_{f}.json").read_text())
        mixacc[a] = d["summary"]["accuracy"]
    titik = [(0.0, r"$=$ \texttt{raw}", geomix["0.0"]["max_cos_embed_mean"],
              sel["humanevalplus"]["raw"]["akurasi"]),
             (0.25, "", geomix["0.25"]["max_cos_embed_mean"], mixacc[0.25]),
             (0.50, "", geomix["0.5"]["max_cos_embed_mean"], mixacc[0.5]),
             (0.75, "", geomix["0.75"]["max_cos_embed_mean"], mixacc[0.75]),
             (1.0, r"$=$ \texttt{soft}", geomix["1.0"]["max_cos_embed_mean"],
              sel["humanevalplus"]["soft"]["akurasi"])]
    isi = "\n".join(
        f"${koma(a)}$ {k} & {koma(g, 3)} & {koma(acc)}" + r" \\" for a, k, g, acc in titik)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Sumbu interpolasi antara \texttt{{raw}} dan \texttt{{soft}}. Vektor
langkah laten dibentuk sebagai $(1-\alpha)$ bagian \texttt{{raw}} dan $\alpha$
bagian \texttt{{soft}}, lalu dinormalkan. Kolom kedua adalah kedekatan
geometris terhadap \textit{{embedding}} token nyata, kolom ketiga akurasi
HumanEval+ pada $n=100$ soal yang sama.}}
\label{{tab:b4-ambang}}
\small
\begin{{tabular}}{{lcc}}
\hline
$\alpha$ & $\max_i\cos$ & Akurasi HumanEval+ \\
\hline
{isi}
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_kanal():
    lengan = [("kv_full", r"\code{kv\_full}", r"seluruh \textit{KV-cache}"),
              ("kv_prompt_only", r"\code{kv\_prompt\_only}", r"posisi \textit{prompt} saja"),
              ("kv_latent_only", r"\code{kv\_latent\_only}", "posisi langkah laten saja"),
              ("none", r"\code{none}", "tanpa warisan")]
    d10 = {m: kanal(f"{m}_m10") for m in MODE}
    d40 = {m: kanal(f"{m}_m40") for m in ["raw", "sample", "moi"]}
    baris = []
    for arm, nama, ket in lengan:
        v10 = [koma(d10[m][(arm, "dsl")][0]) for m in MODE]
        baris.append(f"{nama} & {ket} & " + " & ".join(v10) + r" \\")
    isi10 = "\n".join(baris)
    v40 = []
    for m in MODE:
        v40.append(koma(d40[m][("kv_latent_only", "dsl")][0]) if m in d40 else "---")
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Kapasitas kanal simbolik: proporsi muatan (nama fungsi) yang dapat
dipulihkan agen hilir ketika hanya sebagian \textit{{KV-cache}} diwariskan,
20 percobaan per sel. Baris \code{{kv\_latent\_only}} mengisolasi posisi langkah
laten dari posisi \textit{{prompt}}.}}
\label{{tab:b4-kanal}}
\small
\begin{{tabular}}{{llcccc}}
\hline
Lengan & Yang diwariskan & {NAMA['raw']} & {NAMA['soft']} & {NAMA['sample']} & {NAMA['moi']} \\
\hline
\multicolumn{{6}}{{l}}{{$m=10$ (setelan produksi)}} \\
{isi10}
\hline
\multicolumn{{6}}{{l}}{{$m=40$}} \\
\code{{kv\_latent\_only}} & posisi langkah laten saja & {" & ".join(v40)} \\
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_beta():
    betas = ["0.25", "0.5", "1", "2", "4", "8"]
    nama = {"1": "moi_m10"}
    dsl, tok = [], []
    for b in betas:
        f = "moi_m10" if b == "1" else f"moi_b{b}_m10"
        d = kanal(f)
        dsl.append(koma(d[("kv_latent_only", "dsl")][0]))
        tok.append(koma(d[("kv_latent_only", "token")][0]))
    hdr = " & ".join(f"${koma(float(b))}$" for b in betas)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Ketakpekaan MoI terhadap $\beta$. Nilai $\beta$ disapu pada rentang
32 kali lipat sementara seluruh komponen lain dijaga tetap; \textit{{recall}}
diukur pada lengan \code{{kv\_latent\_only}}, $m=10$, 20 percobaan per nilai.}}
\label{{tab:b4-beta}}
\small
\begin{{tabular}}{{lcccccc}}
\hline
$\beta$ & {hdr} \\
\hline
\textit{{Recall}} nama fungsi & {" & ".join(dsl)} \\
\textit{{Recall}} token acak & {" & ".join(tok)} \\
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_korupsi(sel):
    kol = ["tunggal", "teks", "raw", "soft", "sample", "moi"]
    baris = []
    jml = {k: 0 for k in kol}
    for t in TUGAS:
        v = []
        for k in kol:
            c = sel[t][k].get("cjk")
            v.append("---" if c is None else str(c))
            if c:
                jml[k] += c
        baris.append(f"{LABEL[t]} & " + " & ".join(v) + r" \\")
    isi = "\n".join(baris)
    tot = " & ".join(str(jml[k]) for k in kol)
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Jumlah jawaban yang memuat aksara non-Latin pada tugas berbahasa
Inggris, dari 100 jawaban per sel.}}
\label{{tab:b4-korupsi}}
\small
\begin{{tabular}}{{lcccccc}}
\hline
Tolok ukur & Tunggal & Teks & {NAMA['raw']} & {NAMA['soft']} & {NAMA['sample']} & {NAMA['moi']} \\
\hline
{isi}
\hline
Jumlah & {tot} \\
\hline
\end{{tabular}}
\end{{table}}
"""


def tabel_ringkas(sel):
    """Satu baris per formulasi: rerata lintas tiga tolok ukur."""
    kol = ["teks", "raw", "soft", "sample", "moi"]
    nm = {"teks": "Teks (acuan)", "raw": NAMA["raw"], "soft": NAMA["soft"],
          "sample": NAMA["sample"], "moi": NAMA["moi"]}
    baris = []
    for k in kol:
        akur = np.mean([sel[t][k]["akurasi"] for t in TUGAS])
        tokk = np.mean([sel[t][k]["tok_out"] for t in TUGAS])
        det = np.mean([sel[t][k]["detik"] for t in TUGAS])
        fmt = np.mean([sel[t][k]["format"] for t in TUGAS])
        cjk = sum(sel[t][k]["cjk"] for t in TUGAS)
        s = (f"{nm[k]} & {koma(akur, 3)} & {koma(fmt, 3)} & {koma(tokk, 1)} & "
             f"{koma(det, 1)} & {cjk}")
        if k == "moi":
            s = s.replace(koma(akur, 3), r"\textbf{" + koma(akur, 3) + "}")
        baris.append(s + r" \\")
    isi = "\n".join(baris[:1] + [r"\hline"] + baris[1:])
    return rf"""\begin{{table}}[htbp]
\centering
\caption{{Rekapitulasi lintas tiga tolok ukur (rerata tak berbobot atas GSM8K,
ARC-C, dan HumanEval+; kolom terakhir adalah jumlah).}}
\label{{tab:b4-ringkas}}
\small
\begin{{tabular}}{{lccccc}}
\hline
Perlakuan & Akurasi & Laju format & \textit{{Token}}/soal & Detik/soal & Jawaban rusak \\
\hline
{isi}
\hline
\end{{tabular}}
\end{{table}}
"""


# ───────────────────────── gambar ───────────────────────────────────────────
def g1_hasil_utama(sel):
    kol = ["teks", "raw", "soft", "sample", "moi"]
    # WARNA = {"raw": "#e34948", "soft": "#2a78d6", "sample": "#eda100", "moi": "#4a3aa7"}
    # PENYESUAIAN 1: Palet warna soft tapi kontras
    # Menggunakan warna pastel/muted yang modern
    warna = {
        "teks": "#9a9a95",    # Soft Grey
        "raw": "#D64700",     # Soft Orange
        "soft": "#2a78d6",    # Soft Teal
        "sample": "#eda100",  # Soft Yellow
        "moi": "#7200D6"      # Soft Purple
    }
    nm = {"teks": "teks", "raw": "raw", "soft": "soft", "sample": "sample", "moi": "moi"}
    
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.7))
    x = np.arange(len(TUGAS))
    lebar = 0.165
    panel = [("akurasi", "Akurasi", 2, (0, 1.12)),
             ("tok_out", "Token keluaran per soal", 0, None),
             ("detik", "Waktu per soal (detik)", 1, None)]
             
    for ax, (medan, judul, nd, ylim) in zip(axes, panel):
        for j, k in enumerate(kol):
            v = [sel[t][k][medan] for t in TUGAS]
            off = (j - (len(kol) - 1) / 2) * lebar
            ax.bar(x + off, v, lebar, color=warna[k], edgecolor="white",
                   linewidth=0.7, zorder=3, label=nm[k])
            
            # PENYESUAIAN 2: Blok ax.annotate dihapus agar tidak muncul angka di atas bar
            
        ax.set_xticks(x, [LABEL[t] for t in TUGAS], fontsize=7.5)
        ax.set_title(judul, fontsize=8.6, loc="left")
        ax.grid(axis="x", visible=False)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, max(sel[t][k][medan] for t in TUGAS for k in kol) * 1.28)
            
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", fontsize=7.4, frameon=False, ncol=5,
               bbox_to_anchor=(0.5, -0.06), columnspacing=1.6, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(IMG / "b4_g1_hasil_utama.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



def g2_pareto(sel):
    penanda = {"gsm8k": "o", "arc_challenge": "s", "humanevalplus": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    for ax, medan, xl in [(axes[0], "tok_out", "Token keluaran per soal"),
                          (axes[1], "detik", "Waktu per soal (detik)")]:
        for t in TUGAS:
            for k in ["teks"] + MODE:
                c = sel[t][k]
                w = ABU if k == "teks" else WARNA[k]
                ax.scatter(c[medan], c["akurasi"], s=54, marker=penanda[t],
                           color=w, edgecolor="white", linewidth=0.8, zorder=3)
            # garis penghubung teks -> moi
            ax.plot([sel[t]["teks"][medan], sel[t]["moi"][medan]],
                    [sel[t]["teks"]["akurasi"], sel[t]["moi"]["akurasi"]],
                    color=ABU, lw=0.7, ls=":", zorder=2)
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel("Akurasi", fontsize=8)
        ax.set_ylim(0.35, 1.0)
    from matplotlib.lines import Line2D
    l1 = [Line2D([], [], marker="o", ls="", color=WARNA[k], label=k, markersize=6,
                 markeredgecolor="white") for k in MODE]
    l1.insert(0, Line2D([], [], marker="o", ls="", color=ABU, label="teks",
                        markersize=6, markeredgecolor="white"))
    l2 = [Line2D([], [], marker=penanda[t], ls="", color=ABU_TUA, label=LABEL[t],
                 markersize=6, markeredgecolor="white") for t in TUGAS]
    axes[0].legend(handles=l1, loc="lower right", fontsize=6.5, frameon=False, ncol=1)
    axes[1].legend(handles=l2, loc="lower right", fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "b4_g2_pareto.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def g3_geometri(geo):
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    y = np.arange(len(MODE))[::-1]
    for i, m in enumerate(MODE):
        g = geo[m]
        lo, hi, mu = g["max_cos_embed_min"], g["max_cos_embed_max"], g["max_cos_embed_mean"]
        ax.plot([lo, hi], [y[i], y[i]], color=WARNA[m], lw=5, solid_capstyle="round",
                alpha=0.35, zorder=3)
        ax.scatter([mu], [y[i]], s=52, color=WARNA[m], edgecolor="white",
                   linewidth=0.9, zorder=4)
        ax.annotate(koma(mu, 3), (mu, y[i]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7.2, color=ABU_TUA)
    ax.axvline(1.0, color=ABU_TUA, lw=0.8, ls="--", zorder=2)
    ax.annotate("berimpit dengan embedding\ntoken nyata", (1.0, 3.55),
                xytext=(-7, 0), textcoords="offset points", ha="right", va="top",
                fontsize=6.6, color=ABU_TUA)
    ax.set_yticks(y, MODE)
    ax.set_ylim(-0.7, 3.75)
    ax.set_xlim(0, 1.16)
    ax.set_xlabel(r"$\max_i\ \cos\left(\Phi(h),\ W_{\mathrm{in}}[i]\right)$", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "b4_g3_geometri.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def g4_kanal():
    lengan = ["kv_full", "kv_prompt_only", "kv_latent_only", "none"]
    d10 = {m: kanal(f"{m}_m10") for m in MODE}
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(lengan))
    lebar = 0.2
    for j, m in enumerate(MODE):
        v = [d10[m][(a, "dsl")][0] for a in lengan]
        off = (j - 1.5) * lebar
        ax.bar(x + off, v, lebar, color=WARNA[m], edgecolor="white", linewidth=0.7,
               zorder=3, label=m)
        for i, y in enumerate(v):
            ax.annotate(koma(y), (x[i] + off, y), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=6.2,
                        color=ABU_TUA)
    ax.set_xticks(x, ["kv_full\n(seluruh cache)", "kv_prompt_only\n(prompt saja)",
                      "kv_latent_only\n(langkah laten saja)", "none\n(tanpa warisan)"],
                  fontsize=7)
    ax.set_ylabel("Recall muatan simbolik", fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=4,
              columnspacing=0.9, handlelength=1.1)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(IMG / "b4_g4_kanal.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def g5_beta():
    betas = [0.25, 0.5, 1, 2, 4, 8]
    dsl, tok = [], []
    for b in betas:
        f = "moi_m10" if b == 1 else f"moi_b{b:g}_m10"
        d = kanal(f)
        dsl.append(d[("kv_latent_only", "dsl")][0])
        tok.append(d[("kv_latent_only", "token")][0])
    fig, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.plot(betas, dsl, "-o", color=WARNA["moi"], lw=1.6, ms=5,
            markeredgecolor="white", label="nama fungsi", zorder=3)
    ax.plot(betas, tok, "-s", color=WARNA["moi"], lw=1.6, ms=5, alpha=0.5,
            markeredgecolor="white", label="token acak", zorder=3)
    for b, v in zip(betas, dsl):
        dx = 12 if b == 1 else 0
        ax.annotate(koma(v), (b, v), xytext=(dx, 7), textcoords="offset points",
                    ha="center", fontsize=6.5, color=ABU_TUA)
    ax.axvline(1, color=ABU_TUA, lw=0.8, ls="--", zorder=2)
    ax.annotate("nilai bawaan", (1, 0.55), xytext=(5, 0), textcoords="offset points",
                fontsize=6.6, color=ABU_TUA)
    ax.set_xscale("log", base=2)
    ax.set_xticks(betas, [koma(b) for b in betas])
    ax.set_xlabel(r"$\beta$", fontsize=9)
    ax.set_ylabel("Recall muatan simbolik", fontsize=8)
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "b4_g5_beta.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def g6_ambang(geo, geomix, sel):
    mixacc = {}
    for a, f in [(0.25, "a025"), (0.5, "a05"), (0.75, "a075")]:
        d = json.loads((RES / "bench" / f"bench_humanevalplus_mix_kv_s0_{f}.json").read_text())
        mixacc[a] = d["summary"]["accuracy"]
    xs = [geomix[k]["max_cos_embed_mean"] for k in ["0.0", "0.25", "0.5", "0.75", "1.0"]]
    ys = [sel["humanevalplus"]["raw"]["akurasi"], mixacc[0.25], mixacc[0.5],
          mixacc[0.75], sel["humanevalplus"]["soft"]["akurasi"]]
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.plot(xs, ys, "-", color=ABU, lw=1.2, zorder=2)
    geser = {0: (0, -14), 0.25: (0, -14), 0.5: (0, -14), 0.75: (-32, -13), 1.0: (20, -13)}
    for x, y, a in zip(xs, ys, [0, 0.25, 0.5, 0.75, 1.0]):
        ax.scatter([x], [y], s=40, color=ABU_TUA, edgecolor="white", linewidth=0.8,
                   zorder=3)
        ax.annotate(rf"$\alpha={koma(a)}$", (x, y), xytext=geser[a],
                    textcoords="offset points", ha="center", fontsize=6.3, color=ABU)
    label = {"raw": (0, 12), "soft": (0, 13), "sample": (28, 4), "moi": (28, -2)}
    for m in MODE:
        ax.scatter([geo[m]["max_cos_embed_mean"]], [sel["humanevalplus"][m]["akurasi"]],
                   s=95, marker="D", color=WARNA[m], edgecolor="white", linewidth=1.0,
                   zorder=4)
        ax.annotate(m, (geo[m]["max_cos_embed_mean"], sel["humanevalplus"][m]["akurasi"]),
                    xytext=label[m], textcoords="offset points", ha="center",
                    fontsize=7.4, color=WARNA[m], weight="bold")
    ax.axvspan(0.45, 0.72, color="#f3c9c9", alpha=0.35, zorder=1)
    ax.annotate("zona peralihan", (0.585, 0.34), ha="center", fontsize=6.6, color=ABU_TUA)
    ax.set_xlabel(r"$\max_i\ \cos\left(\Phi(h),\ W_{\mathrm{in}}[i]\right)$", fontsize=8.5)
    ax.set_ylabel("Akurasi HumanEval+", fontsize=8.5)
    ax.set_xlim(0.20, 1.14)
    ax.set_ylim(0.20, 0.85)
    fig.tight_layout()
    fig.savefig(IMG / "b4_g6_ambang.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# def main():
#     sel, pasangan = muat()
#     geo, geomix = geometri()

#     berkas = {
#         "b4_tugas.tex": tabel_tugas(),
#         "b4_konfig.tex": tabel_konfig(),
#         "b4_akurasi.tex": tabel_akurasi(sel),
#         "b4_biaya.tex": tabel_biaya(sel),
#         "b4_efisiensi.tex": tabel_efisiensi(sel),
#         "b4_selisih.tex": tabel_selisih(pasangan),
#         "b4_geometri.tex": tabel_geometri(geo),
#         "b4_ambang.tex": tabel_ambang(geo, geomix, sel),
#         "b4_kanal.tex": tabel_kanal(),
#         "b4_beta.tex": tabel_beta(),
#         "b4_korupsi.tex": tabel_korupsi(sel),
#         "b4_ringkas.tex": tabel_ringkas(sel),
#     }
#     for nama, isi in berkas.items():
#         (TAB / nama).write_text(isi)
#         print("tabel:", nama)

#     g1_hasil_utama(sel)
#     g2_pareto(sel)
#     g3_geometri(geo)
#     g4_kanal()
#     g5_beta()
#     g6_ambang(geo, geomix, sel)
#     for g in ["b4_g1_hasil_utama", "b4_g2_pareto", "b4_g3_geometri", "b4_g4_kanal",
#               "b4_g5_beta", "b4_g6_ambang"]:
#         print("gambar:", g + ".png")


# if __name__ == "__main__":
#     main()

def main():
    # Memuat data (sel dan pasangan)
    sel, pasangan = muat()
    
    # Hanya memanggil fungsi grafik yang diinginkan
    g1_hasil_utama(sel)
    
    print("gambar: b4_g1_hasil_utama.png")


if __name__ == "__main__":
    main()