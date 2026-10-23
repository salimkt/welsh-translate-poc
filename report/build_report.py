#!/usr/bin/env python3
"""Build the "Welsh ⇄ English Machine Translation Benchmark" PDF report.

usage:
    python build_report.py --data data --out welsh_mt_benchmark.pdf [--models models.json]

Inputs (in --data):
    scores.json          {system: {"en-cy"|"cy-en": {bleu, chrf, comet, secs_per_sent, comet_sent[]}}}
    hyps_<system>.json   {"idx": [...], "en-cy": {src, ref, hyp, secs}, "cy-en": {...}}
    ../models.json       per-system metadata (name, family, hf, params_b, runtime, quant, vram_gb,
                         license, deployed); override with --models.

Everything quantitative in the report is computed from these files; only the methodology and
deployment facts are fixed text. Requires reportlab + matplotlib and the DejaVu Sans TTF fonts.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import tempfile
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.lib.units import cm, mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas as rl_canvas  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    CondPageBreak,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)

# --------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------
TITLE = "Welsh ⇄ English Machine Translation Benchmark"
DEFAULT_DATE = "22 October 2026"
DIRECTIONS = ("en-cy", "cy-en")
DIR_LABEL = {"en-cy": "en→cy", "cy-en": "cy→en"}
DIR_LONG = {"en-cy": "English → Welsh", "cy-en": "Welsh → English"}
FLORES_LINES = 1012

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONTS = {
    "DejaVu": "DejaVuSans.ttf",
    "DejaVu-Bold": "DejaVuSans-Bold.ttf",
    "DejaVu-Oblique": "DejaVuSans-Oblique.ttf",
    "DejaVu-BoldOblique": "DejaVuSans-BoldOblique.ttf",
    "DejaVuMono": "DejaVuSansMono.ttf",
}

# Palette: dark grey text, Welsh red used sparingly, one colour per direction.
ACCENT = "#c8102e"
TEXT = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"
ZEBRA = "#f5f5f5"
HEAD_BG = "#ebebeb"
DIR_COLOR = {"en-cy": "#2f5f8a", "cy-en": "#8fb8d8"}

PAGE_W, PAGE_H = A4
MARGIN = 2 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

# Inference-stack versions used for the runs (the scoring stack is detected at build time).
INFERENCE_STACK = [
    ("Ollama", "0.21.2", "TranslateGemma, Gemma 4 (Q4) and Qwen runs (llama.cpp backend)"),
    ("transformers", "5.20", "Gemma 4 int8 and NLLB runs"),
    ("torch", "2.14", "CUDA inference"),
]

DEPLOYMENT = {
    "model": "Qwen 3.6 27B (Q4_K_M) served by Ollama — replaced TranslateGemma 27B (Q8_0) on "
             "23 October 2026 on the strength of this benchmark",
    "api": "FastAPI translation service on port 10083 (backend/server_ollama.py)",
    "vram": "about 22 GB of GPU memory (TranslateGemma 27B used about 32 GB)",
    "public": "https://dev.studio.bluemesh.ai/welsh",
    "ui": "https://salimkt.github.io/welsh-translate-poc/",
    "speech": "vLLM serving distil-whisper/distil-medium.en (English-only speech recognition)",
}


# --------------------------------------------------------------------------------------------
# Fonts
# --------------------------------------------------------------------------------------------
def register_fonts() -> None:
    for name, fname in FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / fname)))
    pdfmetrics.registerFontFamily(
        "DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Oblique",
        boldItalic="DejaVu-BoldOblique",
    )
    for fname in FONTS.values():
        font_manager.fontManager.addfont(str(FONT_DIR / fname))
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.edgecolor": "#9a9a9a",
        "axes.labelcolor": TEXT,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "savefig.dpi": 220,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
@dataclass
class System:
    key: str
    meta: dict
    scores: dict  # direction -> metrics
    hyps: dict = field(default_factory=dict)  # direction -> {src, ref, hyp, secs}
    idx: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.meta.get("name") or self.key

    @property
    def deployed(self) -> bool:
        return bool(self.meta.get("deployed"))

    @property
    def family(self) -> str:
        return self.meta.get("family") or "Unknown"

    @property
    def vram(self):
        return self.meta.get("vram_gb")

    def m(self, d: str, metric: str) -> float:
        return float(self.scores[d][metric])

    @property
    def mean_comet(self) -> float:
        return statistics.fmean(self.m(d, "comet") for d in DIRECTIONS)

    @property
    def mean_secs(self) -> float:
        return statistics.fmean(self.m(d, "secs_per_sent") for d in DIRECTIONS)

    def runaways(self, d: str):
        """Count of runaway outputs, or None when hypotheses are unavailable."""
        h = self.hyps.get(d)
        if not h:
            return None
        return sum(is_runaway(a, r) for a, r in zip(h["hyp"], h["ref"]))

    def sent(self, d: str) -> list[float]:
        return [float(x) for x in self.scores[d].get("comet_sent", [])]

    @property
    def commercial(self) -> bool:
        return "NC" not in (self.meta.get("license") or "")

    @property
    def usable(self) -> bool:
        """COMET of at least 80 in both directions (see the bands in the executive summary)."""
        return all(self.m(d, "comet") >= 80 for d in DIRECTIONS)


def paired_bootstrap(a: list[float], b: list[float], n: int = 2000, seed: int = 0):
    """Mean of a-b per sentence and its 95% bootstrap interval (same sentences resampled for both)."""
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    k = len(diffs)
    means = sorted(statistics.fmean(diffs[rng.randrange(k)] for _ in range(k)) for _ in range(n))
    return statistics.fmean(diffs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def load_systems(data_dir: Path, models_path: Path) -> list[System]:
    scores = json.loads((data_dir / "scores.json").read_text(encoding="utf-8"))
    models = json.loads(models_path.read_text(encoding="utf-8")) if models_path.exists() else {}
    systems = []
    for key, sc in scores.items():
        if not all(d in sc for d in DIRECTIONS):
            print(f"warning: {key} lacks a direction in scores.json; skipped")
            continue
        meta = models.get(key, {"name": key})
        s = System(key, meta, sc)
        hp = data_dir / f"hyps_{key}.json"
        if hp.exists():
            h = json.loads(hp.read_text(encoding="utf-8"))
            s.idx = list(h.get("idx", []))
            s.hyps = {d: h[d] for d in DIRECTIONS if d in h}
        else:
            print(f"warning: {hp.name} missing; {key} will be absent from the examples")
        systems.append(s)
    systems.sort(key=lambda s: s.mean_comet, reverse=True)
    if not systems:
        raise SystemExit("no systems found in scores.json")
    return systems


# --------------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------------
def f1(x: float) -> str:
    return f"{x:.1f}"


def fsecs(x: float) -> str:
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def fparams(p) -> str:
    if p is None:
        return "n/a"
    return f"{p:g}B"


def fvram(v) -> str:
    return "n/a" if v is None else f"{v:g} GB"


def band(comet: float) -> str:
    if comet >= 88:
        return "near-professional"
    if comet >= 83:
        return "good, but needs review for official text"
    if comet >= 80:
        return "usable, with frequent minor errors"
    return "noticeable errors"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def join_names(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def is_runaway(hyp: str, ref: str) -> bool:
    """Output far longer than the reference (typically a generation loop)."""
    return len(hyp) > 2.5 * len(ref) + 50


def clip_output(hyp: str, ref: str) -> str:
    limit = max(220, int(1.6 * len(ref)))
    if len(hyp) <= limit:
        return escape(hyp)
    cut = hyp[:limit].rsplit(" ", 1)[0]
    return (escape(cut) + f" <font color='{ACCENT}'>[… output truncated for display; "
            f"{len(hyp):,} characters in total]</font>")


def pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not detected"


# --------------------------------------------------------------------------------------------
# Styles
# --------------------------------------------------------------------------------------------
def make_styles() -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="DejaVu", fontSize=9.5, leading=13.5, textColor=TEXT,
                          spaceAfter=5)
    st = {
        "body": base,
        "small": ParagraphStyle("small", parent=base, fontSize=8, leading=10.5, spaceAfter=3),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8, leading=10.5,
                                  textColor=MUTED, spaceBefore=2, spaceAfter=10),
        "h1": ParagraphStyle("h1", parent=base, fontName="DejaVu-Bold", fontSize=16, leading=20,
                             spaceBefore=4, spaceAfter=10, keepWithNext=1),
        "h2": ParagraphStyle("h2", parent=base, fontName="DejaVu-Bold", fontSize=12, leading=15,
                             spaceBefore=12, spaceAfter=6, keepWithNext=1),
        "h3": ParagraphStyle("h3", parent=base, fontName="DejaVu-Bold", fontSize=10, leading=13,
                             spaceBefore=8, spaceAfter=4, keepWithNext=1),
        # Example labels: no keepWithNext (a CondPageBreak guards them) so tables can split.
        "h3x": ParagraphStyle("h3x", parent=base, fontName="DejaVu-Bold", fontSize=10, leading=13,
                              spaceBefore=8, spaceAfter=4, keepWithNext=0),
        "cell": ParagraphStyle("cell", parent=base, fontSize=8, leading=10, spaceAfter=0),
        "cell_r": ParagraphStyle("cell_r", parent=base, fontSize=8, leading=10, spaceAfter=0,
                                 alignment=TA_RIGHT),
        "cell_c": ParagraphStyle("cell_c", parent=base, fontSize=8, leading=10, spaceAfter=0,
                                 alignment=TA_CENTER),
        "head": ParagraphStyle("head", parent=base, fontName="DejaVu-Bold", fontSize=8,
                               leading=10, spaceAfter=0),
        "head_c": ParagraphStyle("head_c", parent=base, fontName="DejaVu-Bold", fontSize=8,
                                 leading=10, spaceAfter=0, alignment=TA_CENTER),
        "ex": ParagraphStyle("ex", parent=base, fontSize=7.5, leading=9.5, spaceAfter=0),
        "ex_lab": ParagraphStyle("ex_lab", parent=base, fontName="DejaVu-Bold", fontSize=7.5,
                                 leading=9.5, spaceAfter=0),
        "code": ParagraphStyle("code", parent=base, fontName="DejaVuMono", fontSize=7.5,
                               leading=10, backColor=ZEBRA, borderPadding=6, spaceBefore=10,
                               spaceAfter=16),
        "title": ParagraphStyle("title", parent=base, fontName="DejaVu-Bold", fontSize=26,
                                leading=32, spaceAfter=10, alignment=TA_LEFT),
        "subtitle": ParagraphStyle("subtitle", parent=base, fontSize=11.5, leading=15,
                                   textColor=MUTED, spaceAfter=4),
        "box_h": ParagraphStyle("box_h", parent=base, fontName="DejaVu-Bold", fontSize=10.5,
                                leading=14, spaceAfter=4, textColor=ACCENT),
        "bullet": ParagraphStyle("bullet", parent=base, spaceAfter=3),
    }
    return st


def bullets(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(t, style), leftIndent=12, value="•") for t in items],
        bulletType="bullet", bulletFontName="DejaVu", bulletFontSize=8, leftIndent=12,
        bulletColor=colors.HexColor(ACCENT), spaceAfter=6,
    )


def table(rows, col_widths, header_rows=1, extra=None, zebra=True) -> Table:
    t = Table(rows, colWidths=col_widths, repeatRows=header_rows)
    cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor(HEAD_BG)),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, colors.HexColor(ACCENT)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
    ]
    if zebra and len(rows) > header_rows:
        cmds.append(("ROWBACKGROUNDS", (0, header_rows), (-1, -1),
                     [colors.white, colors.HexColor(ZEBRA)]))
    if extra:
        cmds.extend(extra)
    t.setStyle(TableStyle(cmds))
    return t


# --------------------------------------------------------------------------------------------
# Page decoration
# --------------------------------------------------------------------------------------------
class NumberedCanvas(rl_canvas.Canvas):
    """Canvas that knows the total page count, for "Page n of N" footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_pages = []

    def showPage(self):
        self._saved_pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved_pages)
        for state in self._saved_pages:
            self.__dict__.update(state)
            self._decorate(total)
            super().showPage()
        super().save()

    def _decorate(self, total: int) -> None:
        page = self._pageNumber
        self.saveState()
        if page == 1:
            self.setFillColor(colors.HexColor(ACCENT))
            self.rect(0, PAGE_H - 8 * mm, PAGE_W, 8 * mm, stroke=0, fill=1)
        else:
            self.setFont("DejaVu", 7.5)
            self.setFillColor(colors.HexColor(MUTED))
            self.drawString(MARGIN, PAGE_H - 1.3 * cm, TITLE)
            self.drawRightString(PAGE_W - MARGIN, PAGE_H - 1.3 * cm, "BlueMesh / Netstratum")
            self.setStrokeColor(colors.HexColor(ACCENT))
            self.setLineWidth(0.6)
            self.line(MARGIN, PAGE_H - 1.45 * cm, PAGE_W - MARGIN, PAGE_H - 1.45 * cm)
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.4)
            self.line(MARGIN, 1.45 * cm, PAGE_W - MARGIN, 1.45 * cm)
            self.drawString(MARGIN, 1.05 * cm, "FLORES-200 devtest evaluation")
            self.drawRightString(PAGE_W - MARGIN, 1.05 * cm, f"Page {page} of {total}")
        self.restoreState()


# --------------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------------
def _style_deployed_ticks(ax, systems: list[System], axis: str = "y") -> None:
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lab, s in zip(labels, systems):
        if s.deployed:
            lab.set_color(ACCENT)
            lab.set_fontweight("bold")


def _tick_names(systems: list[System]) -> list[str]:
    return [s.name + ("  (deployed)" if s.deployed else "") for s in systems]


def chart_metric_bars(systems: list[System], metric: str, label: str, path: Path) -> float:
    """Grouped horizontal bars of one metric for both directions. Returns the x-axis start."""
    n = len(systems)
    fig, ax = plt.subplots(figsize=(6.7, (0.55 if n <= 4 else 0.4) * n + 1.2))
    h = 0.38
    vals = [s.m(d, metric) for s in systems for d in DIRECTIONS]
    lo = max(0.0, (min(vals) - 8) // 5 * 5)
    hi = min(100.0, max(vals) + 4)
    for j, d in enumerate(DIRECTIONS):
        ys = [i + (j - 0.5) * h for i in range(n)]
        xs = [s.m(d, metric) for s in systems]
        ax.barh(ys, [x - lo for x in xs], left=lo, height=h * 0.92, color=DIR_COLOR[d],
                label=DIR_LONG[d], zorder=2)
        for y, x in zip(ys, xs):
            ax.text(x + 0.3, y, f1(x), va="center", ha="left", fontsize=7.5, color=TEXT)
    ax.set_yticks(range(n), _tick_names(systems))
    ax.invert_yaxis()
    _style_deployed_ticks(ax, systems)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(label)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return lo


def chart_quality_speed(systems: list[System], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.7, 3.9))
    xs = [s.mean_secs for s in systems]
    ys = [s.mean_comet for s in systems]
    known = [s.vram for s in systems if s.vram]
    vmax = max(known) if known else 1
    sizes = [60 + 340 * (s.vram / vmax) if s.vram else 55 for s in systems]
    for s, x, y, size in zip(systems, xs, ys, sizes):
        edge = ACCENT if s.deployed else DIR_COLOR["en-cy"]
        if s.vram:
            ax.scatter(x, y, s=size, color=DIR_COLOR["cy-en"], edgecolor=edge,
                       linewidth=1.6 if s.deployed else 1.0, alpha=0.9, zorder=3)
        else:
            ax.scatter(x, y, s=size, facecolor="white", edgecolor=edge,
                       linewidth=1.6 if s.deployed else 1.2, zorder=3)
    ax.set_xscale("log")
    xmin, xmax = min(xs), max(xs)
    ax.set_xlim(xmin / 1.8, xmax * 2.6)
    ypad = max(1.5, (max(ys) - min(ys)) * 0.2)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)
    fig.canvas.draw()
    # Label placement: try right/left/above/below each marker and take the first spot that
    # overlaps neither another label nor a marker; otherwise shift down on the right.
    ppt = fig.dpi / 72.0  # pixels per point
    pts = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    radii = [((sz ** 0.5) / 2) * ppt for sz in sizes]
    boxes = []  # placed label rectangles in pixels (x0, y0, x1, y1)

    def clash(r):
        if any(not (r[2] < b[0] or r[0] > b[2] or r[3] < b[1] or r[1] > b[3]) for b in boxes):
            return True
        for (cx, cy), rad in zip(pts, radii):
            nx, ny = min(max(cx, r[0]), r[2]), min(max(cy, r[1]), r[3])
            if (nx - cx) ** 2 + (ny - cy) ** 2 < (rad + 1) ** 2:
                return True
        ax_box = ax.get_window_extent()
        return r[0] < ax_box.x0 or r[2] > ax_box.x1 or r[1] < ax_box.y0 or r[3] > ax_box.y1

    for i in sorted(range(len(systems)), key=lambda i: -ys[i]):
        s = systems[i]
        label = s.name + (" (deployed)" if s.deployed else "")
        w, h = len(label) * (5.0 if s.deployed else 4.5) * ppt, 9 * ppt
        (cx, cy), rad = pts[i], radii[i]
        gap = rad + 4 * ppt
        cands = [(cx + gap, cy - h / 2, "left"), (cx - gap - w, cy - h / 2, "left"),
                 (cx - w / 2, cy + gap, "left"), (cx - w / 2, cy - gap - h, "left")]
        cands += [(cx + gap, cy - h / 2 - k * h, "left") for k in range(1, 6)]
        cands += [(cx + gap, cy - h / 2 + k * h, "left") for k in range(1, 6)]
        x0, y0, _ = next((c for c in cands if not clash((c[0], c[1], c[0] + w, c[1] + h))),
                         cands[0])
        boxes.append((x0, y0, x0 + w, y0 + h))
        tx, ty = ax.transData.inverted().transform((x0, y0 + h / 2))
        far = abs(y0 + h / 2 - cy) > rad + h
        ax.annotate(
            label, xy=(xs[i], ys[i]), xytext=(tx, ty), textcoords="data", fontsize=7.5,
            va="center", ha="left",
            color=ACCENT if s.deployed else TEXT, fontweight="bold" if s.deployed else "normal",
            arrowprops=(dict(arrowstyle="-", color="#b0b0b0", lw=0.5, shrinkA=1,
                             shrinkB=rad / ppt) if far else None),
        )
    ax.set_xlabel("Seconds per sentence (mean of both directions, log scale) — lower is faster")
    ax.set_ylabel("Mean COMET (both directions)")
    ax.grid(True, color=GRID, zorder=0, which="major")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    note = "Marker area ∝ measured VRAM; hollow = VRAM not measured. Red outline = deployed."
    ax.text(0.0, -0.2, note, transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def chart_distributions(systems: list[System], path: Path) -> None:
    n = len(systems)
    fig, axes = plt.subplots(1, 2, figsize=(6.7, (0.42 if n <= 4 else 0.32) * n + 1.4),
                             sharey=True)
    for ax, d in zip(axes, DIRECTIONS):
        data = [s.sent(d) for s in systems]
        bp = ax.boxplot(
            data, orientation="horizontal", positions=range(n), widths=0.55, patch_artist=True,
            showfliers=True, medianprops=dict(color=TEXT, linewidth=1.2),
            whiskerprops=dict(color="#8a8a8a"), capprops=dict(color="#8a8a8a"),
            flierprops=dict(marker="o", markersize=2.5, markerfacecolor="#9a9a9a",
                            markeredgecolor="none", alpha=0.7),
        )
        for patch, s in zip(bp["boxes"], systems):
            patch.set_facecolor(DIR_COLOR[d])
            patch.set_edgecolor(ACCENT if s.deployed else "#5a5a5a")
            patch.set_linewidth(1.4 if s.deployed else 0.8)
            patch.set_alpha(0.9)
        means = [statistics.fmean(x) if x else float("nan") for x in data]
        ax.scatter(means, range(n), marker="D", s=14, color="white", edgecolor=TEXT,
                   linewidth=0.8, zorder=4)
        ax.set_title(DIR_LONG[d])
        ax.set_xlabel("Sentence-level COMET")
        ax.xaxis.grid(True, color=GRID, zorder=0)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(n), _tick_names(systems))
    axes[0].invert_yaxis()
    _style_deployed_ticks(axes[0], systems)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def img(path: Path, width: float = CONTENT_W) -> Image:
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw)


# --------------------------------------------------------------------------------------------
# Analysis helpers
# --------------------------------------------------------------------------------------------
def ranked(systems: list[System], d: str, metric: str = "comet", reverse=True) -> list[System]:
    return sorted(systems, key=lambda s: s.m(d, metric), reverse=reverse)


def best(systems, d, metric="comet"):
    lower_better = metric == "secs_per_sent"
    return ranked(systems, d, metric, reverse=not lower_better)[0]


def family_kind(s: System) -> str:
    f = s.family.lower()
    if "dedicated" in f or f.startswith("mt"):
        return "dedicated"
    if "translation" in f:
        return "llm_mt"
    if "llm" in f:
        return "llm_general"
    return "other"


def tg_family(systems):
    return sorted([s for s in systems if "translategemma" in (s.meta.get("hf") or "").lower()],
                  key=lambda s: s.meta.get("params_b") or 0)


def quant_groups(systems):
    """Same underlying checkpoint run at different quantisations."""
    groups: dict[str, list[System]] = {}
    for s in systems:
        hf = s.meta.get("hf")
        if hf:
            groups.setdefault(hf, []).append(s)
    return {k: v for k, v in groups.items() if len({s.meta.get('quant') for s in v}) > 1}


def direction_observations(systems: list[System], d: str) -> list[str]:
    obs = []
    r = ranked(systems, d)
    top, bottom = r[0], r[-1]
    n = len(systems)
    if n == 1:
        obs.append(f"Only one system was scored: <b>{escape(top.name)}</b> reached COMET "
                   f"{f1(top.m(d, 'comet'))} ({band(top.m(d, 'comet'))}, by the rule of thumb).")
        return obs
    spread = top.m(d, "comet") - bottom.m(d, "comet")
    obs.append(
        f"<b>{escape(top.name)}</b> has the highest COMET ({f1(top.m(d, 'comet'))}); "
        f"<b>{escape(bottom.name)}</b> the lowest ({f1(bottom.m(d, 'comet'))}). "
        f"The spread between best and worst is {f1(spread)} COMET points."
    )
    second = r[1]
    gap = top.m(d, "comet") - second.m(d, "comet")
    if gap < 1.0:
        obs.append(
            f"The top two systems ({escape(top.name)} and {escape(second.name)}) are only "
            f"{f1(gap)} COMET apart, which is within the sampling uncertainty of a 100-sentence "
            "test set; treat them as statistically tied on this evidence."
        )
    elif gap < 2.0:
        obs.append(f"The lead over the runner-up ({escape(second.name)}) is {f1(gap)} COMET — "
                   "a meaningful but modest margin given the sample size.")
    else:
        obs.append(f"The lead over the runner-up ({escape(second.name)}) is {f1(gap)} COMET, "
                   "larger than the expected sampling noise (about 1–2 points).")

    # Surface-metric agreement
    bc, bf, bb = best(systems, d, "comet"), best(systems, d, "chrf"), best(systems, d, "bleu")
    if bf.key != bc.key or bb.key != bc.key:
        others = []
        if bf.key != bc.key:
            others.append(f"chrF++ ({escape(bf.name)}, {f1(bf.m(d, 'chrf'))})")
        if bb.key != bc.key:
            others.append(f"BLEU ({escape(bb.name)}, {f1(bb.m(d, 'bleu'))})")
        obs.append(
            "The metrics do not agree on the winner: the best "
            + " and ".join(others)
            + f" differ from the COMET leader. String-overlap metrics reward wording that matches "
            "the single reference, whereas COMET estimates meaning preservation, so this usually "
            "indicates that outputs differ in phrasing rather than in accuracy."
        )
    else:
        obs.append(f"COMET, chrF++ and BLEU all rank {escape(bc.name)} first in this direction.")

    # Family comparisons
    kinds: dict[str, list[System]] = {}
    for s in systems:
        kinds.setdefault(family_kind(s), []).append(s)
    if "llm_mt" in kinds and "llm_general" in kinds:
        a, b = best(kinds["llm_mt"], d), best(kinds["llm_general"], d)
        diff = a.m(d, "comet") - b.m(d, "comet")
        lead = "translation-tuned" if diff > 0 else "general-purpose"
        obs.append(
            f"Translation-tuned vs general LLMs: the best translation-tuned model "
            f"({escape(a.name)}, {f1(a.m(d, 'comet'))}) vs the best general LLM "
            f"({escape(b.name)}, {f1(b.m(d, 'comet'))}) — the {lead} model leads by "
            f"{f1(abs(diff))} COMET."
        )
    llms = kinds.get("llm_mt", []) + kinds.get("llm_general", [])
    if "dedicated" in kinds and llms:
        a, b = best(kinds["dedicated"], d), best(llms, d)
        diff = b.m(d, "comet") - a.m(d, "comet")
        who = "the LLM" if diff > 0 else "the dedicated MT model"
        obs.append(
            f"Dedicated MT vs LLMs: best dedicated system {escape(a.name)} "
            f"({f1(a.m(d, 'comet'))}) vs best LLM {escape(b.name)} ({f1(b.m(d, 'comet'))}); "
            f"{who} is ahead by {f1(abs(diff))} COMET."
        )

    # TranslateGemma model size
    tg = tg_family(systems)
    if len({s.meta.get('params_b') for s in tg}) > 1:
        parts = [f"{escape(s.name)} {f1(s.m(d, 'comet'))}" for s in tg]
        big, small = tg[-1], tg[0]
        diff = big.m(d, "comet") - small.m(d, "comet")
        trend = "gains" if diff > 0 else "does not gain"
        obs.append(
            f"Model size within TranslateGemma: {'; '.join(parts)}. Going from "
            f"{fparams(small.meta.get('params_b'))} to {fparams(big.meta.get('params_b'))} "
            f"{trend} {f1(abs(diff))} COMET in this direction"
            + (" (note the smaller models ran at a different quantisation)."
               if len({s.meta.get('quant') for s in tg}) > 1 else ".")
        )

    # Quantisation of the same checkpoint
    for hf, group in quant_groups(systems).items():
        g = sorted(group, key=lambda s: s.m(d, "comet"), reverse=True)
        parts = [f"{escape(s.meta.get('quant') or '?')} ({escape(s.meta.get('runtime') or '?')}): "
                 f"{f1(s.m(d, 'comet'))}" for s in g]
        diff = g[0].m(d, "comet") - g[-1].m(d, "comet")
        obs.append(
            f"Quantisation of {escape(hf)}: {'; '.join(parts)}. The difference is {f1(diff)} "
            "COMET; because runtime and quantisation changed together, this cannot be attributed "
            "to quantisation alone."
        )

    loops = []
    for s in systems:
        k = s.runaways(d)
        if not k:
            continue
        h, sc = s.hyps[d], s.sent(d)
        bad = [f1(sc[i]) for i, (a, r) in enumerate(zip(h["hyp"], h["ref"]))
               if is_runaway(a, r) and i < len(sc)]
        loops.append(f"{escape(s.name)} {k}" + (f" (COMET {', '.join(bad)})" if bad else ""))
    if loops:
        obs.append(
            "Runaway outputs — text more than 2.5× the reference length, typically a word or "
            f"phrase repeated until the generation limit: {'; '.join(loops)}. These failures are "
            "rare but severe, and a production deployment would need a length guard."
        )

    fast = best(systems, d, "secs_per_sent")
    slow = ranked(systems, d, "secs_per_sent", reverse=True)[0]
    obs.append(
        f"Speed: {escape(fast.name)} was fastest at {fsecs(fast.m(d, 'secs_per_sent'))} s per "
        f"sentence and {escape(slow.name)} slowest at {fsecs(slow.m(d, 'secs_per_sent'))} s "
        "(single sequential requests on a shared GPU server; indicative only)."
    )
    return obs


def pick_examples(systems: list[System], d: str, k: int):
    """Return [(position, reason)] — sentences with most disagreement plus one consensus case."""
    with_h = [s for s in systems if d in s.hyps and s.sent(d)]
    if not with_h:
        return [], with_h
    n = min(len(s.sent(d)) for s in with_h)
    n = min(n, *(len(s.hyps[d]["hyp"]) for s in with_h))
    if len(with_h) >= 2:
        disagree = sorted(range(n), key=lambda i: statistics.pstdev([s.sent(d)[i] for s in with_h]),
                          reverse=True)
    else:
        disagree = sorted(range(n), key=lambda i: with_h[0].sent(d)[i])

    def short_enough(i):
        return len(with_h[0].hyps[d]["src"][i]) <= 260

    picks = []
    for i in disagree:
        if short_enough(i):
            picks.append((i, "large disagreement between systems" if len(with_h) > 1
                          else "low-scoring sentence"))
        if len(picks) >= k - 1:
            break
    consensus = sorted(range(n), key=lambda i: min(s.sent(d)[i] for s in with_h), reverse=True)
    for i in consensus:
        if short_enough(i) and i not in [p[0] for p in picks]:
            picks.append((i, "all systems score well"))
            break
    return picks, with_h


# --------------------------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------------------------
class Report:
    def __init__(self, systems: list[System], st: dict, tmp: Path, date: str, n_examples: int):
        self.S = systems
        self.st = st
        self.tmp = tmp
        self.date = date
        self.n_examples = n_examples
        self.deployed = next((s for s in systems if s.deployed), None)
        self.n_sent = self._sample_size()

    def _sample_size(self) -> int:
        for s in self.S:
            if s.idx:
                return len(s.idx)
        return len(self.S[0].sent(DIRECTIONS[0])) or 100

    def P(self, text, style="body"):
        return Paragraph(text, self.st[style])

    # -- title page ---------------------------------------------------------------------------
    def title_page(self):
        st, S = self.st, self.S
        out = [Spacer(1, 2.2 * cm), self.P(TITLE, "title"),
               self.P(f"FLORES-200 devtest · {self.n_sent} sentences per direction · {self.date}",
                      "subtitle"),
               self.P("Prepared for BlueMesh / Netstratum", "subtitle"),
               Spacer(1, 0.4 * cm)]
        hr = Table([[""]], colWidths=[3 * cm], rowHeights=[2])
        hr.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(ACCENT))]))
        hr.hAlign = "LEFT"
        out += [hr, Spacer(1, 0.9 * cm)]

        rec = S[0]
        items = []
        for d in DIRECTIONS:
            b = best(S, d)
            items.append(f"Best {DIR_LONG[d]}: <b>{escape(b.name)}</b> — COMET "
                         f"{f1(b.m(d, 'comet'))} ({band(b.m(d, 'comet'))}).")
        if self.deployed:
            dep = self.deployed
            rank = S.index(dep) + 1
            items.append(
                f"Deployed system <b>{escape(dep.name)}</b> ranks {rank} of {len(S)} on mean "
                f"COMET ({f1(dep.mean_comet)}: en→cy {f1(dep.m('en-cy', 'comet'))}, "
                f"cy→en {f1(dep.m('cy-en', 'comet'))})."
            )
        usable = [s for s in S if s.usable] or S
        fast = min(usable, key=lambda s: s.mean_secs)
        sp = (f"Fastest usable system (COMET ≥ 80 both ways): <b>{escape(fast.name)}</b> at "
              f"{fsecs(fast.mean_secs)} s per sentence (mean of both directions)")
        if self.deployed and fast.key != self.deployed.key:
            sp += f"; the deployed system averages {fsecs(self.deployed.mean_secs)} s"
        items.append(sp + ".")
        known = [s for s in S if s.vram is not None]
        if known:
            vt = "; ".join(f"{escape(s.name)} {fvram(s.vram)}" for s in known)
            missing = len(S) - len(known)
            items.append(f"Measured GPU memory: {vt}"
                         + (f" ({plural(missing, 'system')} not measured)." if missing else "."))
        rec_txt = (f"<b>Recommendation:</b> {escape(rec.name)} has the highest mean COMET "
                   f"({f1(rec.mean_comet)}) across both directions")
        if self.deployed and rec.key == self.deployed.key:
            rec_txt += " and is already the deployed system — keep it in production."
        elif self.deployed:
            gap = rec.mean_comet - self.deployed.mean_comet
            rec_txt += f", {f1(gap)} points above the deployed {escape(self.deployed.name)}"
            if not rec.commercial:
                rec_txt += (f", but its licence ({escape(rec.meta.get('license', ''))}) rules out "
                            "commercial use")
            rec_txt += "."
            best_c = next((s for s in S if s.commercial), None)
            if best_c and best_c.key != rec.key:
                if best_c.key == self.deployed.key:
                    rec_txt += (f" Among commercially usable systems the deployed "
                                f"{escape(best_c.name)} scores highest.")
                else:
                    rec_txt += (f" The best commercially usable system is "
                                f"<b>{escape(best_c.name)}</b> ({f1(best_c.mean_comet)}, VRAM "
                                f"{fvram(best_c.vram)}) — the strongest candidate to replace "
                                f"{escape(self.deployed.name)}; see the significance table before "
                                "switching.")
        else:
            rec_txt += "."
        box_content = [self.P("Key findings", "box_h"), bullets(items, st["bullet"]),
                       Spacer(1, 2), self.P(rec_txt)]
        box = Table([[box_content]], colWidths=[CONTENT_W])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f7f7")),
            ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(ACCENT)),
            ("LEFTPADDING", (0, 0), (-1, -1), 14),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        out += [box, Spacer(1, 1.0 * cm),
                self.P(f"Systems evaluated: {len(S)}. Metrics: COMET (wmt22-comet-da), chrF++ and "
                       "BLEU. All scores in this report are computed from the benchmark output "
                       "files; COMET is shown on a 0–100 scale.", "small"),
                PageBreak()]
        return out

    # -- executive summary --------------------------------------------------------------------
    def executive_summary(self):
        S = self.S
        out = [self.P("1. Executive summary", "h1")]
        rec = S[0]
        b_en, b_cy = best(S, "en-cy"), best(S, "cy-en")
        avg = {d: statistics.fmean(s.m(d, "comet") for s in S) for d in DIRECTIONS}
        harder = "en-cy" if avg["en-cy"] < avg["cy-en"] else "cy-en"
        easier = "cy-en" if harder == "en-cy" else "en-cy"
        out.append(self.P(
            f"We compared {plural(len(S), 'machine-translation system')} on {self.n_sent} "
            f"sentences of the public FLORES-200 benchmark, translating in both directions "
            f"between English and Welsh. Each output was compared to a professional human "
            f"translation using three automatic metrics; the main one, COMET, estimates how well "
            f"the meaning is preserved and correlates best with human judgement."
        ))
        out.append(self.P(
            f"<b>{DIR_LONG[harder]} is the harder direction.</b> Averaged over all systems, COMET is "
            f"{f1(avg[harder])} for {DIR_LONG[harder]} versus {f1(avg[easier])} for "
            f"{DIR_LONG[easier]}. Producing correct Welsh — mutations, gender, word order and "
            f"idiomatic vocabulary — is where systems differ most, so it should weigh most when "
            f"choosing a model for Welsh-language output."
        ))
        sent = (f"The best English → Welsh system is <b>{escape(b_en.name)}</b> "
                f"({f1(b_en.m('en-cy', 'comet'))}); the best Welsh → English system is "
                f"<b>{escape(b_cy.name)}</b> ({f1(b_cy.m('cy-en', 'comet'))}). Across both "
                f"directions <b>{escape(rec.name)}</b> has the highest mean COMET "
                f"({f1(rec.mean_comet)})")
        if self.deployed:
            sent += (", and it is the system currently deployed." if rec.key == self.deployed.key
                     else f", while the deployed system is {escape(self.deployed.name)} "
                          f"({f1(self.deployed.mean_comet)}).")
        else:
            sent += "."
        out.append(self.P(sent))

        out.append(self.P("What the COMET numbers mean in practice", "h3"))
        out.append(self.P(
            "COMET has no absolute pass mark. The bands below are a <i>rule of thumb</i> for "
            "this kind of sentence-level news/encyclopaedic text, based on typical correlations "
            "with human ratings — they are not guarantees, and individual sentences can still "
            "contain serious errors at any score."
        ))
        rows = [[self.P("COMET", "head"), self.P("Rule-of-thumb reading", "head"),
                 self.P("Systems in this band (direction)", "head")]]
        bands = [("≥ 88", 88, 999, "Near-professional: fluent and accurate; occasional "
                  "slips that a reviewer would catch quickly."),
                 ("83 – 88", 83, 88, "Good: meaning usually right, but wording, mutations or "
                  "terminology need review before official publication."),
                 ("80 – 83", 80, 83, "Usable for gist; frequent minor errors."),
                 ("< 80", -1, 80, "Noticeable errors: mistranslations, untranslated or "
                  "invented words; not suitable without full post-editing.")]
        for lab, lo, hi, txt in bands:
            members = [f"{escape(s.name)} ({DIR_LABEL[d]} {f1(s.m(d, 'comet'))})"
                       for s in S for d in DIRECTIONS if lo <= s.m(d, "comet") < hi]
            rows.append([self.P(f"<b>{lab}</b>", "cell"), self.P(txt, "cell"),
                         self.P("; ".join(members) or "—", "cell")])
        out.append(table(rows, [2.0 * cm, 7.4 * cm, CONTENT_W - 9.4 * cm]))
        out.append(Spacer(1, 8))
        out.append(self.P(
            "In everyday terms: a system around COMET 89 produces Welsh or English that a fluent "
            "reader would mostly accept as written; around 84 the text is understandable and "
            "largely correct, but a Welsh speaker would want to edit a noticeable share of "
            "sentences; in the high 70s readers will meet clear errors in many sentences."
        ))
        out.append(self.P(
            f"Scores are from a {self.n_sent}-sentence sample, so differences of about 1–2 COMET "
            "points between systems should not be over-interpreted. Speed figures were measured on "
            "a shared production GPU server and are indicative only."
        ))
        out.append(PageBreak())
        return out

    # -- methodology --------------------------------------------------------------------------
    def methodology(self):
        n = self.n_sent
        step = FLORES_LINES // n if n else 10
        out = [self.P("2. Methodology", "h1"), self.P("Test set", "h2")]
        out.append(self.P(
            f"We used the <b>FLORES-200 devtest</b> split from Meta's public release, aligned "
            f"English (<font name='DejaVuMono'>eng_Latn</font>) and Welsh "
            f"(<font name='DejaVuMono'>cym_Latn</font>) files of {FLORES_LINES:,} sentences each. "
            f"An evenly spaced sample of {n} sentences was taken (every {step}th line, starting at "
            f"line 1), and the same sentences were translated in both directions: English → Welsh "
            f"(en→cy) and Welsh → English (cy→en). The FLORES texts are professionally translated "
            f"sentences drawn from Wikipedia-style articles (news, science, travel, health)."
        ))
        out.append(self.P("Decoding and prompting", "h2"))
        out.append(bullets([
            "LLMs used greedy decoding (temperature 0), one sentence per request, with "
            "“thinking”/reasoning modes disabled.",
            "TranslateGemma models used Google's official translation prompt template "
            "(reproduced in the appendix).",
            "Other LLMs used a generic system prompt asking for a professional, fluent "
            "translation that preserves meaning, tone, names and numbers, outputting only the "
            "translation.",
            "NLLB-based models used beam search with 4 beams and the FLORES language codes as "
            "source/target tokens.",
        ], self.st["bullet"]))
        out.append(self.P("Metrics", "h2"))
        out.append(bullets([
            "<b>COMET</b> (Unbabel/wmt22-comet-da), reference-based: a neural metric built on "
            "XLM-RoBERTa (whose training languages include Welsh) that scores each translation "
            "against the source and the reference. Reported ×100; the system score is the mean of "
            "sentence scores. Primary metric for this report.",
            "<b>chrF++</b> (sacrebleu, corpus-level, word_order=2): character n-gram F-score plus "
            "word bigrams; more tolerant than BLEU of Welsh morphology.",
            "<b>BLEU</b> (sacrebleu, corpus-level, default tokenisation): word n-gram overlap with "
            "the reference.",
            "<b>Latency</b>: wall-clock seconds per sentence, measured as total time for the "
            "sample divided by the number of sentences, sending sequential single requests.",
        ], self.st["bullet"]))
        out.append(self.P("Hardware", "h2"))
        out.append(self.P(
            "All systems ran on one server with 2× NVIDIA RTX PRO 6000 Blackwell GPUs (96 GB "
            "each). The GPUs were shared with other production services during the runs, so "
            "timings are indicative rather than controlled benchmarks."
        ))
        out.append(self.P("Limitations", "h2"))
        out.append(bullets([
            f"Sample size: {n} sentences per direction gives an uncertainty of roughly ±1–2 "
            "COMET points; small differences are not significant.",
            "Single reference: valid alternative translations are penalised, especially by "
            "BLEU and chrF++.",
            "Domain: FLORES is Wikipedia-style prose, which differs from public-service, legal "
            "or conversational text.",
            "BLEU penalises Welsh initial mutations and flexible word order even when the "
            "translation is correct, so it understates Welsh-output quality.",
            "Latency was affected by other workloads on the GPUs.",
            "Systems ran on different runtimes and quantisations (see Models evaluated), so "
            "quality and speed reflect the configuration tested, not the model in the abstract.",
        ], self.st["bullet"]))
        return out

    # -- models -------------------------------------------------------------------------------
    def models(self):
        out = [CondPageBreak(6 * cm), self.P("3. Models evaluated", "h1")]
        out.append(self.P(
            "Systems are listed in order of mean COMET across both directions. VRAM is the GPU "
            "memory observed while serving, where it was measured."
        ))
        rows = [[self.P(h, "head") for h in
                 ("System", "Type", "Size", "Runtime / quantisation", "VRAM", "Licence")]]
        extra = []
        for i, s in enumerate(self.S, start=1):
            m = s.meta
            name = escape(s.name) + (" <font color='%s'>(deployed)</font>" % ACCENT
                                     if s.deployed else "")
            hf = escape(m.get("hf") or "")
            rows.append([
                self.P(f"<b>{name}</b><br/><font size=6.5 color='{MUTED}'>{hf}</font>", "cell"),
                self.P(escape(s.family), "cell"),
                self.P(fparams(m.get("params_b")), "cell_r"),
                self.P(f"{escape(m.get('runtime') or 'n/a')}<br/>{escape(m.get('quant') or 'n/a')}",
                       "cell"),
                self.P(fvram(s.vram), "cell_r"),
                self.P(escape(m.get("license") or "n/a"), "cell"),
            ])
        out.append(table(rows, [5.0 * cm, 3.0 * cm, 1.4 * cm, 3.3 * cm, 1.5 * cm,
                                CONTENT_W - 14.2 * cm], extra=extra))
        out.append(Spacer(1, 6))
        nc = [s for s in self.S if "nc" in (s.meta.get("license") or "").lower().split("-")]
        notes = ["<b>Gemma Terms of Use</b> permit commercial use subject to Google's "
                 "prohibited-use policy; <b>Apache-2.0</b> is a permissive open-source licence."]
        if nc:
            notes.append(
                f"<b>CC-BY-NC</b> licences ({escape(join_names([s.name for s in nc]))}) are "
                "<b>non-commercial</b>: these models can be used for research and evaluation but "
                "not in a commercial product or paid service without a separate licence."
            )
        else:
            notes.append("<b>CC-BY-NC</b> licences (used by NLLB-based models) are non-commercial; "
                         "none of the evaluated systems in this run carries one.")
        out.append(self.P("Licence notes", "h3"))
        out.append(bullets(notes, self.st["bullet"]))
        return out

    # -- results ------------------------------------------------------------------------------
    def results(self):
        S, tmp = self.S, self.tmp
        out = [CondPageBreak(9 * cm), self.P("4. Results", "h1")]
        out.append(self.P(
            "Table 1 gives corpus-level scores. Higher is better for COMET, chrF++ and BLEU; "
            "lower is better for seconds per sentence (mean of the two directions). The best "
            "value in each column is shown in bold."
        ))
        metrics = [("en-cy", "comet"), ("en-cy", "chrf"), ("en-cy", "bleu"),
                   ("cy-en", "comet"), ("cy-en", "chrf"), ("cy-en", "bleu")]
        best_val = {dm: max(s.m(*dm) for s in S) for dm in metrics}
        best_speed = min(s.mean_secs for s in S)
        hc = self.st["head_c"]
        rows = [
            [self.P("System", "head"), Paragraph("English → Welsh", hc), "", "",
             Paragraph("Welsh → English", hc), "", "", Paragraph("Speed", hc)],
            ["", *[Paragraph(x, hc) for x in ("COMET", "chrF++", "BLEU") * 2],
             Paragraph("s / sent.", hc)],
        ]
        for s in S:
            cells = [self.P(escape(s.name) + (f" <font color='{ACCENT}'>●</font>"
                                              if s.deployed else ""), "cell")]
            for dm in metrics:
                v = s.m(*dm)
                txt = f1(v)
                cells.append(self.P(f"<b>{txt}</b>" if abs(v - best_val[dm]) < 1e-9 else txt,
                                    "cell_r"))
            v = s.mean_secs
            cells.append(self.P(f"<b>{fsecs(v)}</b>" if abs(v - best_speed) < 1e-9 else fsecs(v),
                                "cell_r"))
            rows.append(cells)
        num_w = (CONTENT_W - 4.6 * cm) / 7
        extra = [("SPAN", (0, 0), (0, 1)), ("SPAN", (1, 0), (3, 0)), ("SPAN", (4, 0), (6, 0)),
                 ("VALIGN", (0, 0), (-1, 1), "MIDDLE"),
                 ("LINEBEFORE", (1, 0), (1, -1), 0.4, colors.HexColor("#c8c8c8")),
                 ("LINEBEFORE", (4, 0), (4, -1), 0.4, colors.HexColor("#c8c8c8")),
                 ("LINEBEFORE", (7, 0), (7, -1), 0.4, colors.HexColor("#c8c8c8"))]
        t = table(rows, [4.6 * cm] + [num_w] * 7, header_rows=2, extra=extra)
        cap = ("<b>Table 1.</b> Corpus-level scores, ordered by mean COMET. "
               f"<font color='{ACCENT}'>●</font> marks the deployed system.")
        out += [KeepTogether([t, self.P(cap, "caption")])]

        p = tmp / "comet_bars.png"
        lo = chart_metric_bars(S, "comet", "COMET (higher is better)", p)
        out.append(KeepTogether([
            self.P("COMET by system and direction", "h3"), img(p),
            self.P(f"<b>Figure 1.</b> Corpus COMET for each system. The axis starts at {lo:g} to "
                   "make differences visible; deployed system labelled in red.", "caption")]))

        p = tmp / "quality_speed.png"
        chart_quality_speed(S, p)
        out.append(KeepTogether([
            self.P("Quality vs speed", "h3"), img(p),
            self.P("<b>Figure 2.</b> Mean COMET against mean seconds per sentence. Up and to the "
                   "left is better. Timings come from a shared GPU server and are indicative.",
                   "caption")]))

        p = tmp / "distributions.png"
        chart_distributions(S, p)
        out.append(KeepTogether([
            self.P("Sentence-level score distribution", "h3"), img(p),
            self.P("<b>Figure 3.</b> Distribution of per-sentence COMET. Boxes show the "
                   "inter-quartile range, the line the median, the white diamond the mean, and "
                   "dots individual outlying sentences. A long left tail means occasional badly "
                   "wrong translations even when the average is good.", "caption")]))
        rows = [[self.P(h, "head") for h in ("System", "Direction", "Median", "10th pct.",
                                             "Sentences < 70", "Sentences ≥ 88",
                                             "Runaway outputs")]]
        for s in S:
            for d in DIRECTIONS:
                x = sorted(s.sent(d))
                if not x:
                    continue
                p10 = x[max(0, int(round(0.1 * (len(x) - 1))))]
                rows.append([self.P(escape(s.name), "cell"), self.P(DIR_LABEL[d], "cell"),
                             self.P(f1(statistics.median(x)), "cell_r"), self.P(f1(p10), "cell_r"),
                             self.P(str(sum(v < 70 for v in x)), "cell_r"),
                             self.P(str(sum(v >= 88 for v in x)), "cell_r"),
                             self.P("n/a" if s.runaways(d) is None else str(s.runaways(d)),
                                    "cell_r")])
        out.append(KeepTogether([
            self.P("Tail behaviour", "h3"),
            table(rows, [4.6 * cm, 1.8 * cm] + [(CONTENT_W - 6.4 * cm) / 5] * 5),
            self.P(f"<b>Table 2.</b> Per-sentence COMET statistics (out of {self.n_sent} "
                   "sentences per direction). Runaway outputs are hypotheses more than 2.5× the "
                   "reference length, usually repetition loops.", "caption")]))

        p = tmp / "chrf_bars.png"
        lo = chart_metric_bars(S, "chrf", "chrF++ (higher is better)", p)
        out.append(KeepTogether([
            self.P("chrF++ by system and direction", "h3"), img(p),
            self.P(f"<b>Figure 4.</b> Corpus chrF++ (axis starts at {lo:g}). chrF++ measures "
                   "character overlap with the reference and is less sensitive to meaning than "
                   "COMET.", "caption")]))
        out += self.significance()
        return out

    def significance(self):
        """Paired bootstrap of per-sentence COMET against the deployed system."""
        dep = self.deployed
        if not dep or not dep.sent(DIRECTIONS[0]):
            return []
        out = [CondPageBreak(8 * cm), self.P("Is the difference real? (significance)", "h3"),
               self.P(
                   f"Each system is compared with the deployed {escape(dep.name)} on the same "
                   "sentences: the per-sentence COMET differences are resampled 2,000 times "
                   "(paired bootstrap) to give a 95% interval. If the interval excludes zero the "
                   "difference is unlikely to be sampling noise; otherwise it is reported as a tie.")]
        rows = [[self.P(h, "head") for h in
                 ("System", "en→cy Δ COMET [95% CI]", "", "cy→en Δ COMET [95% CI]", "")]]
        for s in self.S:
            if s.key == dep.key or not s.sent(DIRECTIONS[0]):
                continue
            row = [self.P(escape(s.name), "cell")]
            for d in DIRECTIONS:
                mean, lo, hi = paired_bootstrap(s.sent(d), dep.sent(d))
                verdict = "better" if lo > 0 else "worse" if hi < 0 else "tie"
                row += [self.P(f"{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]", "cell_r"),
                        self.P(f"<b>{verdict}</b>" if verdict != "tie" else verdict, "cell")]
            rows.append(row)
        out.append(table(rows, [5.4 * cm, 3.9 * cm, 1.5 * cm, 3.9 * cm, CONTENT_W - 14.7 * cm]))
        out.append(self.P(f"<b>Table 3.</b> COMET difference versus {escape(dep.name)} "
                          "(positive = scored higher than the deployed system).", "caption"))
        return out

    # -- per-direction analysis ---------------------------------------------------------------
    def direction_analysis(self):
        out = [CondPageBreak(10 * cm), self.P("5. Analysis by direction", "h1")]
        for i, d in enumerate(DIRECTIONS, start=1):
            out.append(self.P(f"5.{i} {DIR_LONG[d]}", "h2"))
            r = ranked(self.S, d)
            rank_txt = "; ".join(f"{k}. {escape(s.name)} ({f1(s.m(d, 'comet'))})"
                                 for k, s in enumerate(r, start=1))
            out.append(self.P(f"<b>Ranking by COMET:</b> {rank_txt}."))
            out.append(bullets(direction_observations(self.S, d), self.st["bullet"]))
        out.append(self.P(
            "These observations are generated directly from the scores. With a 100-sentence "
            "sample, differences under about 1–2 COMET points should be read as ties.", "small"))
        keys = {s.key for s in self.S}
        notes = []
        if keys & {"nllb33", "techiaith"}:
            notes.append(
                "<b>Possible home advantage for NLLB-based systems.</b> FLORES-200 was built by "
                "Meta's NLLB team and its dev split was used when developing NLLB, so NLLB-200 and "
                "the Techiaith fine-tunes built on it may match the test set's style more closely "
                "than the LLMs do. Their lead should be confirmed on in-domain text.")
        if "techiaith" in keys:
            notes.append(
                "<b>Techiaith English → Welsh model is legislation-domain.</b> The general-domain "
                "English → Welsh fine-tune on Hugging Face is missing part of its weights, so the "
                "legislation fine-tune (techiaith/nllb-200-1.3B-ft-eng-to-cym-legislation) was used "
                "instead.")
        notes.append(
            "<b>High scores still hide real errors.</b> In spot checks, NLLB-200 rendered "
            "“dinosaur feathers” as “rhewiau'r ddinasoedd” (“the cities' frosts”) while scoring "
            "well overall, and Qwen 3.6 left English technical terms untranslated (“barbs a "
            "barbules”).")
        if "tg4" in keys:
            notes.append(
                "<b>TranslateGemma 4B is not usable for English → Welsh.</b> Its output contains "
                "invented words and occasionally another language (e.g. Dutch “Omdat”).")
        if "gemma4" in keys:
            notes.append(
                "<b>Gemma 4 31B (int8) produced runaway outputs</b> (long repetition loops) on two "
                "English → Welsh sentences, which also explains its high average latency.")
        out.append(self.P("5.3 Caveats from manual spot checks", "h2"))
        out.append(bullets(notes, self.st["bullet"]))
        return out

    # -- qualitative examples -----------------------------------------------------------------
    def examples(self):
        out = [PageBreak(), self.P("6. Qualitative examples", "h1")]
        out.append(self.P(
            "For each direction we show sentences where the systems disagree most (largest "
            "spread of sentence-level COMET) and one sentence that every system handles well. "
            "Each output is followed by its sentence COMET. Line numbers refer to the FLORES-200 "
            "devtest file (1-based)."
        ))
        for j, d in enumerate(DIRECTIONS, start=1):
            picks, with_h = pick_examples(self.S, d, self.n_examples or
                                          (4 if len(self.S) <= 4 else 3))
            heading = Paragraph(f"6.{j} {DIR_LONG[d]}",
                                ParagraphStyle("h2x", parent=self.st["h2"], keepWithNext=0))
            if not picks:
                out += [heading, self.P("No hypothesis files available for this direction.")]
                continue
            for k, (pos, why) in enumerate(picks, start=1):
                ref_sys = with_h[0]
                line = ref_sys.idx[pos] + 1 if pos < len(ref_sys.idx) else pos + 1
                label = self.P(f"Example {j}.{k} — devtest line {line} ({why})", "h3x")
                rows = [
                    [self.P("Source", "ex_lab"), self.P(escape(ref_sys.hyps[d]["src"][pos]), "ex")],
                    [self.P("Reference", "ex_lab"),
                     self.P(escape(ref_sys.hyps[d]["ref"][pos]), "ex")],
                ]
                order = sorted(with_h, key=lambda s: s.sent(d)[pos], reverse=True)
                for s in order:
                    c = s.sent(d)[pos]
                    col = ACCENT if c < 70 else TEXT
                    rows.append([
                        self.P(f"{escape(s.name)}<br/><font color='{col}'>COMET {f1(c)}</font>",
                               "ex"),
                        self.P(clip_output(s.hyps[d]["hyp"][pos], s.hyps[d]["ref"][pos]),
                               "ex")])
                t = Table(rows, colWidths=[3.8 * cm, CONTENT_W - 3.8 * cm], repeatRows=2)
                t.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("BACKGROUND", (0, 0), (-1, 1), colors.HexColor(HEAD_BG)),
                    ("LINEBELOW", (0, 1), (-1, 1), 0.8, colors.HexColor(ACCENT)),
                    ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor(ZEBRA)]),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#bdbdbd")),
                ]))
                # Long tables may split (source/reference rows repeat); the CondPageBreak keeps
                # headings from being stranded at the foot of a page.
                out.append(CondPageBreak(6 * cm))
                if k == 1:
                    out.append(heading)
                out += [label, t, Spacer(1, 8)]
        out.append(KeepTogether([
            self.P("Common error patterns", "h2"),
            self.P("<i>Observations from manual spot checks of the outputs, not a systematic "
                   "error annotation.</i>", "small"),
            bullets([
                "Untranslated English words left in Welsh output, especially technical terms, "
                "names of organisations and rarer vocabulary.",
                "Wrong day or month names and incorrect or missing initial mutations "
                "(soft, nasal, aspirate) after prepositions, possessives and numerals.",
                "Literal word-for-word choices that are grammatical but unidiomatic, or that pick "
                "the wrong sense of an ambiguous word.",
                "Dropped words or clauses in long sentences — fluent output that silently omits "
                "part of the meaning, which surface metrics can miss.",
                "Occasional invented or garbled words (in the weaker systems) that look "
                "plausibly Welsh but are not real words.",
            ], self.st["bullet"]),
        ]))
        return out

    # -- deployment ---------------------------------------------------------------------------
    def deployment(self):
        S, dep = self.S, self.deployed
        out = [CondPageBreak(10 * cm), self.P("7. Deployment and recommendations", "h1"),
               self.P("Current deployment", "h2")]
        rows = [[self.P("Component", "head"), self.P("Details", "head")]]
        for k, v in (("Translation model", DEPLOYMENT["model"]),
                     ("API", DEPLOYMENT["api"]),
                     ("GPU memory", DEPLOYMENT["vram"]),
                     ("Public endpoint", DEPLOYMENT["public"]),
                     ("Web UI", DEPLOYMENT["ui"]),
                     ("Speech input", DEPLOYMENT["speech"])):
            rows.append([self.P(k, "cell"), self.P(escape(v), "cell")])
        out.append(table(rows, [3.6 * cm, CONTENT_W - 3.6 * cm]))
        out.append(Spacer(1, 4))
        if dep:
            out.append(self.P(
                f"In this benchmark the deployed {escape(dep.name)} scored COMET "
                f"{f1(dep.m('en-cy', 'comet'))} for English → Welsh and "
                f"{f1(dep.m('cy-en', 'comet'))} for Welsh → English, at "
                f"{fsecs(dep.m('en-cy', 'secs_per_sent'))} and "
                f"{fsecs(dep.m('cy-en', 'secs_per_sent'))} seconds per sentence respectively."))
        out.append(self.P(
            "The speech front-end uses an English-only Whisper model, so Welsh speech cannot "
            "currently be transcribed; spoken input works only for English → Welsh."))

        out.append(self.P("Recommendations", "h2"))
        recs = []
        rec = S[0]
        if dep and rec.key == dep.key:
            recs.append(f"<b>Keep {escape(dep.name)} in production for quality.</b> It has the "
                        f"highest mean COMET ({f1(dep.mean_comet)}) of the systems tested.")
        elif dep:
            gap_en = rec.m("en-cy", "comet") - dep.m("en-cy", "comet")
            recs.append(
                f"<b>{escape(rec.name)}</b> has the highest mean COMET ({f1(rec.mean_comet)}), "
                f"{f1(rec.mean_comet - dep.mean_comet)} above the deployed {escape(dep.name)} "
                f"({f1(gap_en):s} on English → Welsh). If the gap exceeds the 1–2 point sampling "
                "noise and the licence and resource cost are acceptable, consider switching; "
                f"otherwise keep {escape(dep.name)}."
                + ("" if rec.commercial else
                   f" Note: its licence ({escape(rec.meta.get('license', ''))}) does not allow "
                   "commercial use."))
            best_c = next((s for s in S if s.commercial), None)
            if best_c and best_c.key not in (rec.key, dep.key):
                recs.append(
                    f"<b>Best commercially usable alternative: {escape(best_c.name)}</b> "
                    f"(mean COMET {f1(best_c.mean_comet)} vs {f1(dep.mean_comet)}, VRAM "
                    f"{fvram(best_c.vram)} vs {fvram(dep.vram)}). It is the strongest candidate "
                    f"to replace {escape(dep.name)}; check Table 3 for whether the gain is "
                    "significant, and spot-check for untranslated English terms first.")
        else:
            recs.append(f"<b>{escape(rec.name)}</b> has the highest mean COMET "
                        f"({f1(rec.mean_comet)}) and is the recommended choice for quality.")

        # VRAM-reduction options: smaller members of the deployed model's family.
        if dep:
            fam = tg_family(S) if dep in tg_family(S) else [s for s in S if s.family == dep.family]
            smaller = [s for s in fam if s.key != dep.key
                       and (s.meta.get("params_b") or 0) < (dep.meta.get("params_b") or 0)]
            if smaller:
                parts = []
                for s in sorted(smaller, key=lambda s: -(s.meta.get("params_b") or 0)):
                    g_en = dep.m("en-cy", "comet") - s.m("en-cy", "comet")
                    g_cy = dep.m("cy-en", "comet") - s.m("cy-en", "comet")
                    close = max(g_en, g_cy) <= 1.5
                    parts.append(
                        f"{escape(s.name)} ({fparams(s.meta.get('params_b'))}, "
                        f"{escape(s.meta.get('quant') or '')}, VRAM {fvram(s.vram)}) is "
                        f"{f1(g_en)} COMET lower on en→cy and {f1(g_cy)} on cy→en"
                        + (" — close enough to be a realistic lower-memory option"
                           if close else " — a noticeable quality cost"))
                recs.append("<b>Options to reduce GPU memory:</b> " + "; ".join(parts) + ". "
                            "Gaps are relative to the deployed model; negative values mean the "
                            "smaller model scored higher. A quantised build of the current model "
                            "is another route, but should be re-scored before switching.")
            else:
                recs.append("<b>Options to reduce GPU memory:</b> smaller TranslateGemma variants "
                            "(12B, 4B) were not part of this run; score them before considering "
                            "a switch.")
        recs.append("<b>Keep a human in the loop for official Welsh.</b> Even the best system "
                    "makes mutation, terminology and omission errors; Welsh-language publications "
                    "that fall under the Welsh Language Standards should be reviewed by a fluent "
                    "Welsh speaker before release.")
        out.append(bullets(recs, self.st["bullet"]))

        out.append(self.P("Suggested next steps", "h2"))
        out.append(bullets([
            "Run a larger evaluation (500+ sentences) on in-domain text, e.g. Senedd proceedings "
            "and Welsh public-service corpora, to measure performance on the content the service "
            "will actually translate.",
            "Commission a human evaluation by Welsh speakers (adequacy and fluency ratings, error "
            "categories) on a subset, to calibrate what COMET differences mean for readers.",
            "Replace the English-only speech model with a multilingual Whisper model (or a "
            "Welsh fine-tune) so that Welsh speech can be transcribed and translated.",
            "Add glossary / terminology support (preferred translations for organisation names "
            "and official terms) and evaluate its effect on consistency.",
            "Re-run the benchmark on an otherwise idle GPU for reliable latency and throughput "
            "figures, including batched requests.",
        ], self.st["bullet"]))
        return out

    # -- appendix -----------------------------------------------------------------------------
    def appendix(self):
        S = self.S
        out = [PageBreak(), self.P("Appendix A. Full scores", "h1")]
        rows = [[self.P(h, "head") for h in ("System", "Dir.", "COMET", "chrF++", "BLEU",
                                             "Median sent. COMET", "s / sent.", "Total s")]]
        for s in S:
            for d in DIRECTIONS:
                sent = s.sent(d)
                tot = s.hyps.get(d, {}).get("secs")
                rows.append([
                    self.P(escape(s.name), "cell"), self.P(DIR_LABEL[d], "cell"),
                    self.P(f"{s.m(d, 'comet'):.2f}", "cell_r"),
                    self.P(f"{s.m(d, 'chrf'):.2f}", "cell_r"),
                    self.P(f"{s.m(d, 'bleu'):.2f}", "cell_r"),
                    self.P(f"{statistics.median(sent):.2f}" if sent else "n/a", "cell_r"),
                    self.P(f"{s.m(d, 'secs_per_sent'):.3f}", "cell_r"),
                    self.P(f"{tot:.1f}" if isinstance(tot, (int, float)) else "n/a", "cell_r"),
                ])
        w = (CONTENT_W - 6.4 * cm) / 6
        out.append(table(rows, [5.0 * cm, 1.4 * cm] + [w] * 6))
        out.append(self.P("<b>Table A1.</b> All scores to two decimals. Total s is the wall-clock "
                          "time for the whole sample in that direction.", "caption"))

        out.append(self.P("Appendix B. Environment", "h1"))
        rows = [[self.P(h, "head") for h in ("Component", "Version", "Used for")]]
        for name, ver, use in INFERENCE_STACK:
            rows.append([self.P(name, "cell"), self.P(ver, "cell"), self.P(use, "cell")])
        for name, use in (("sacrebleu", "BLEU and chrF++"),
                          ("unbabel-comet", "COMET (Unbabel/wmt22-comet-da)")):
            rows.append([self.P(name, "cell"), self.P(pkg_version(name), "cell"),
                         self.P(use, "cell")])
        for name, use in (("reportlab", "this report"), ("matplotlib", "charts")):
            rows.append([self.P(name, "cell"), self.P(pkg_version(name), "cell"),
                         self.P(use, "cell")])
        rows.append([self.P("GPUs", "cell"), self.P("2× RTX PRO 6000 Blackwell", "cell"),
                     self.P("96 GB each; shared with production services", "cell")])
        out.append(table(rows, [3.6 * cm, 4.4 * cm, CONTENT_W - 8.0 * cm]))
        out.append(self.P("<b>Table B1.</b> Software and hardware. Scoring and report package "
                          "versions are detected when the report is built.", "caption"))

        out.append(self.P("Appendix C. Prompts", "h1"))
        out.append(self.P("TranslateGemma prompt template (single user turn; {s}/{t} are the "
                          "language names, {sc}/{tc} the codes, e.g. English (en) → Welsh (cy). "
                          "The blank lines before the text are part of the template.", "small"))
        tg = ("You are a professional {s} ({sc}) to {t} ({tc}) translator. Your goal is to\n"
              "accurately convey the meaning and nuances of the original {s} text while\n"
              "adhering to {t} grammar, vocabulary, and cultural sensitivities.\n"
              "Produce only the {t} translation, without any additional explanations or\n"
              "commentary. Please translate the following {s} text into {t}:\n\n\n{text}")
        out.append(XPreformatted(escape(tg), self.st["code"]))
        out.append(self.P("Generic system prompt used for the other LLMs (the sentence is sent as "
                          "the user message):", "small"))
        gp = ("You are a professional {s} to {t} translator. Translate the user's text into\n"
              "natural, fluent, grammatically correct {t}, preserving meaning, tone, names\n"
              "and numbers. Output only the translation, with no notes or explanations.")
        out.append(XPreformatted(escape(gp), self.st["code"]))
        out.append(self.P("Line breaks inside the templates above are added for display; the "
                          "prompts were sent as continuous sentences.", "small"))
        return out

    def story(self):
        return (self.title_page() + self.executive_summary() + self.methodology() + self.models()
                + self.results() + self.direction_analysis() + self.examples()
                + self.deployment() + self.appendix())


# --------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data", help="directory with scores.json and hyps_*.json")
    ap.add_argument("--out", default="welsh_mt_benchmark.pdf")
    ap.add_argument("--models", default=None, help="models.json (default: <data>/../models.json)")
    ap.add_argument("--date", default=DEFAULT_DATE)
    ap.add_argument("--examples", type=int, default=0,
                    help="examples per direction (default: 4, or 3 with more than 4 systems)")
    args = ap.parse_args()

    data_dir = Path(args.data)
    models_path = Path(args.models) if args.models else data_dir.resolve().parent / "models.json"
    register_fonts()
    systems = load_systems(data_dir, models_path)
    st = make_styles()
    with tempfile.TemporaryDirectory() as td:
        rep = Report(systems, st, Path(td), args.date, args.examples)
        doc = SimpleDocTemplate(
            args.out, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=MARGIN, title=TITLE,
            author="BlueMesh / Netstratum", subject="FLORES-200 Welsh–English MT evaluation",
        )
        doc.build(rep.story(), canvasmaker=NumberedCanvas)
    print(f"wrote {args.out} ({len(systems)} systems: {', '.join(s.key for s in systems)})")


if __name__ == "__main__":
    main()
