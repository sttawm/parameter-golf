#!/usr/bin/env python3
import tempfile, os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from lxml import etree

FONT       = "Helvetica Neue"
NEAR_BLACK = RGBColor(0x1A, 0x1A, 0x1A)
BODY_GRAY  = RGBColor(0x44, 0x44, 0x44)
PASTEL_SEP = RGBColor(0xB8, 0xD4, 0xBE)   # soft sage

CHART_IMG = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/best_three_embed_corr.png"
OUT_PATH  = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/correlation_slide.pptx"

# Flatten RGBA → RGB so PowerPoint doesn't mangle transparency
_img = Image.open(CHART_IMG).convert("RGBA")
_bg  = Image.new("RGB", _img.size, (255, 255, 255))
_bg.paste(_img, mask=_img.split()[3])
_tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
_bg.save(_tmp.name, "PNG")
CHART_IMG_FLAT = _tmp.name
IMG_W, IMG_H = _bg.size

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank, white

def add_rect(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    # Remove any theme-inherited shadow
    spPr = shape._element.find(qn("p:spPr"))
    if spPr is not None:
        el = spPr.find(qn("a:effectLst"))
        if el is None:
            el = etree.SubElement(spPr, qn("a:effectLst"))
        for child in list(el):
            el.remove(child)
    return shape

def add_text(slide, text, left, top, width, height,
             size=18, bold=False, italic=False,
             color=NEAR_BLACK, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf  = box.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = align
    r   = p.add_run()
    r.text = text
    r.font.size   = Pt(size)
    r.font.bold   = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name   = FONT
    return box, tf

# ── Title ────────────────────────────────────────────────────────────────────
add_text(slide, "Correlation of probability & similarity",
         0.55, 0.22, 12.0, 0.65,
         size=30, bold=False, color=NEAR_BLACK)

# ── Pastel separator under title ─────────────────────────────────────────────
add_rect(slide, 0.55, 0.90, 8.5, 0.03, PASTEL_SEP)

# ── Bullets (no dot — em-dash as a quiet leader) ─────────────────────────────
bullet_box = slide.shapes.add_textbox(Inches(0.55), Inches(1.05), Inches(12.0), Inches(1.1))
tf = bullet_box.text_frame
tf.word_wrap = True

def add_bullet(tf, plain_before, bold_phrase, plain_after, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before = Pt(5)
    p.space_after  = Pt(5)

    def r(text, bold=False, color=BODY_GRAY):
        run = p.add_run()
        run.text = text
        run.font.size  = Pt(17)
        run.font.bold  = bold
        run.font.color.rgb = color
        run.font.name  = FONT
        return run

    r("– ", color=RGBColor(0xCC, 0xCC, 0xCC))
    if plain_before:
        r(plain_before)
    r(bold_phrase, bold=True, color=NEAR_BLACK)
    if plain_after:
        r(plain_after)

add_bullet(tf, "High-probability tokens tend to ", "cluster in embedding space", ".", first=True)
add_bullet(tf, "The pattern ", "persists across vocabulary sizes", " (1 K → 50 K).")

# ── Chart — explicit size to preserve aspect ratio ───────────────────────────
chart_w = Inches(12.43)
chart_h = int(chart_w * IMG_H / IMG_W)
slide.shapes.add_picture(CHART_IMG_FLAT, Inches(0.45), Inches(2.1), chart_w, chart_h)
os.unlink(CHART_IMG_FLAT)

# ── Caption ──────────────────────────────────────────────────────────────────
cap_box = slide.shapes.add_textbox(Inches(0.45), Inches(7.05), Inches(12.43), Inches(0.3))
tf = cap_box.text_frame
p  = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
r  = p.add_run()
r.text = "500 FineWeb contexts per model  ·  bin means ± 1 std  ·  Pearson r / Spearman ρ per panel"
r.font.size   = Pt(9.5)
r.font.italic = True
r.font.color.rgb = RGBColor(0x99, 0x99, 0x99)
r.font.name   = FONT

prs.save(OUT_PATH)
print(f"Saved → {OUT_PATH}")
