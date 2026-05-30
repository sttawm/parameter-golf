#!/usr/bin/env python3
"""Rebuild slide 5 (Experiment #1A) in the clean Helvetica Neue style."""

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
PASTEL_SEP = RGBColor(0xB8, 0xD4, 0xBE)

SRC_PPTX = "/Users/sttawm/Downloads/NLP Presentation (1).pptx"
OUT_PPTX = "/Users/sttawm/Downloads/NLP Presentation (1) - updated.pptx"

EU_IMG   = "/Users/sttawm/dev/parameter-golf/eu_sweep_plot_regen.png"
GPU_IMG  = "/Users/sttawm/dev/parameter-golf/8gpu_wallclock_plot.png"


def flatten_png(path):
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGB", img.size, (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    bg.save(tmp.name, "PNG")
    return tmp.name, img.size


def add_rect(slide, left, top, width, height, color):
    shape = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    spPr = shape._element.find(qn("p:spPr"))
    if spPr is not None:
        el = spPr.find(qn("a:effectLst"))
        if el is None:
            el = etree.SubElement(spPr, qn("a:effectLst"))
        for child in list(el):
            el.remove(child)
    return shape


def add_picture_fit(slide, path, left, top, width):
    tmp, (iw, ih) = flatten_png(path)
    height = int(Inches(width) * ih / iw)
    slide.shapes.add_picture(tmp, Inches(left), Inches(top), Inches(width), height)
    os.unlink(tmp)


# ── Open pptx and replace slide 5 ────────────────────────────────────────────
prs   = Presentation(SRC_PPTX)
W     = prs.slide_width.inches    # 10.0
H     = prs.slide_height.inches   # 5.62

slide = prs.slides[4]

# Remove all existing shapes
sp_tree = slide.shapes._spTree
for sp in list(sp_tree)[2:]:   # keep first two (background/layout anchors)
    sp_tree.remove(sp)

# White background
bg = slide.background
bg.fill.solid()
bg.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

# ── Title ─────────────────────────────────────────────────────────────────────
tb = slide.shapes.add_textbox(Inches(0.38), Inches(0.14), Inches(9.24), Inches(0.52))
tf = tb.text_frame; tf.word_wrap = False
p  = tf.paragraphs[0]
r  = p.add_run()
r.text = "Next-token prediction from scratch"
r.font.size = Pt(24); r.font.bold = False
r.font.color.rgb = NEAR_BLACK; r.font.name = FONT

# ── Separator ─────────────────────────────────────────────────────────────────
add_rect(slide, 0.38, 0.68, 7.0, 0.022, PASTEL_SEP)

# ── Bullets ───────────────────────────────────────────────────────────────────
bt = slide.shapes.add_textbox(Inches(0.38), Inches(0.80), Inches(9.24), Inches(0.85))
tf = bt.text_frame; tf.word_wrap = True

def bullet(tf, pre, bold_part, post, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.space_before = Pt(4); p.space_after = Pt(4)
    def r(text, bold=False, color=BODY_GRAY):
        run = p.add_run()
        run.text = text
        run.font.size = Pt(15); run.font.bold = bold
        run.font.color.rgb = color; run.font.name = FONT
    r("– ", color=RGBColor(0xCC, 0xCC, 0xCC))
    if pre: r(pre)
    r(bold_part, bold=True, color=NEAR_BLACK)
    if post: r(post)

bullet(tf, "", "No improvement in 10 minutes", " at full budget.", first=True)
bullet(tf, "The loss term carries signal — training on it alone acts as a ",
       "weak surrogate for cross-entropy", ".")

# ── Two charts side by side ───────────────────────────────────────────────────
add_picture_fit(slide, EU_IMG,  0.18, 1.72, 4.82)
add_picture_fit(slide, GPU_IMG, 5.10, 1.72, 4.72)

prs.save(OUT_PPTX)
print(f"Saved → {OUT_PPTX}")
