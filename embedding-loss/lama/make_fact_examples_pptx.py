#!/usr/bin/env python3
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

GRAY   = RGBColor(0x33, 0x33, 0x33)
RED    = RGBColor(0xC0, 0x39, 0x2B)
BGFILL = RGBColor(0xF9, 0xF9, 0xF9)
BGLINE = RGBColor(0xE0, 0xE0, 0xE0)
NUMBG  = RGBColor(0xE4, 0xE4, 0xE4)

def add_textbox(slide, l, t, w, h, text, size=12, bold=False,
                color=RGBColor(0x22,0x22,0x22), align=PP_ALIGN.LEFT, wrap=False):
    tx = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tx.text_frame; tf.word_wrap = wrap
    p  = tf.paragraphs[0]; p.alignment = align
    run = p.add_run(); run.text = text
    run.font.size = Pt(size); run.font.bold = bold
    run.font.color.rgb = color
    return tx

def add_rect(slide, l, t, w, h, fill_rgb, line_rgb=None):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = fill_rgb
    if line_rgb:
        shape.line.color.rgb = line_rgb; shape.line.width = Pt(0.75)
    else:
        shape.line.fill.background()
    return shape

# ── Title ──────────────────────────────────────────────────────────────────────
add_textbox(slide, 0.5, 0.15, 12.3, 0.55,
            "Same fact, different sentences",
            size=24, bold=True, align=PP_ALIGN.CENTER)

# ── Answer badge ─────────────────────────────────────────────────────────────
# Single pill: "Answer: Copenhagen"
add_rect(slide, 5.17, 0.85, 2.99, 0.72,
         RGBColor(0xD4, 0xED, 0xDA), RGBColor(0xBB,0xBB,0xBB))
add_textbox(slide, 5.22, 0.88, 2.89, 0.28, "Answer",
            size=9, color=RGBColor(0x77,0x77,0x77), align=PP_ALIGN.CENTER)
add_textbox(slide, 5.22, 1.14, 2.89, 0.40, "Copenhagen",
            size=14, bold=True, align=PP_ALIGN.CENTER)

# ── 2×2 grid of sentence cards ────────────────────────────────────────────────
SENTENCES = [
    ("It is located on the east coast of the Jutland peninsula, in the geographical centre of Denmark, 187 km northwest of ",
     "[MASK]", "."),
    ("",
     "[MASK]", " is home to the University of Copenhagen, the Technical University of Denmark and Copenhagen Business School."),
    ("Copenhagen is home to the University of ",
     "[MASK]", ", the Technical University of Denmark and Copenhagen Business School."),
    ("The city is located in central Denmark, 187 kilometres northwest of ",
     "[MASK]", ", and 289 kilometres north of Hamburg, Germany."),
]

# 2 columns, 2 rows
COL_L  = [0.40, 6.90]   # left edge of each column
COL_W  = 6.10            # card width
ROW_T  = [1.85, 4.25]   # top of each row
ROW_H  = 2.15            # card height — tall enough for wrapping

for i, (pre, mask, post) in enumerate(SENTENCES):
    col = i % 2
    row = i // 2
    cl  = COL_L[col]
    rt  = ROW_T[row]

    # Card background
    add_rect(slide, cl, rt, COL_W, ROW_H, BGFILL, BGLINE)

    # Number badge
    add_rect(slide, cl + 0.10, rt + 0.18, 0.40, 0.40, NUMBG)
    add_textbox(slide, cl + 0.10, rt + 0.16, 0.40, 0.44, str(i + 1),
                size=12, bold=True, color=RGBColor(0x55,0x55,0x55),
                align=PP_ALIGN.CENTER)

    # Sentence text with [MASK] in red — single textbox, word wrap on
    tx = slide.shapes.add_textbox(
        Inches(cl + 0.65), Inches(rt + 0.18),
        Inches(COL_W - 0.75), Inches(ROW_H - 0.30))
    tf = tx.text_frame
    tf.word_wrap = True
    p  = tf.paragraphs[0]

    def add_run(p, text, color, bold=False, size=13):
        if not text: return
        run = p.add_run(); run.text = text
        run.font.size = Pt(size); run.font.color.rgb = color
        run.font.bold = bold

    add_run(p, pre,  GRAY)
    add_run(p, mask, RED, bold=True, size=14)
    add_run(p, post, GRAY)

# ── Footnote ──────────────────────────────────────────────────────────────────
add_textbox(slide, 0.5, 6.9, 12.3, 0.45,
            "All four sentences encode the same (Denmark, capital of, Copenhagen) triple.  "
            "With a random split, variants of the same fact appear in both training and test sets.",
            size=9, color=RGBColor(0x99,0x99,0x99), align=PP_ALIGN.CENTER, wrap=True)

out = "/Users/sttawm/Documents/fact_examples.pptx"
prs.save(out)
print(f"Saved → {out}")
