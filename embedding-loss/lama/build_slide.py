#!/usr/bin/env python3
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches, Pt

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]  # completely blank
slide = prs.slides.add_slide(blank_layout)

def rgb(hex_str):
    h = hex_str.lstrip("#")
    return RGBColor(int(h[0:2],16), int(h[2:4],16), int(h[4:6],16))

def add_textbox(slide, left, top, width, height, text, font_size,
                bold=False, color="#333333", align=PP_ALIGN.LEFT,
                italic=False, font_name="Calibri"):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    run.font.name = font_name
    return txBox

def add_pill(slide, left, top, width, height, text, font_size,
             text_color, bg_color, border_color, font_name="Courier New"):
    """Rounded rectangle with centered text — makes numbers pop."""
    from pptx.util import Emu
    from pptx.enum.shapes import PP_PLACEHOLDER
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.ROUNDED_RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(bg_color)
    shape.line.color.rgb = rgb(border_color)
    shape.line.width = Pt(1.2)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = True
    run.font.color.rgb = rgb(text_color)
    run.font.name = font_name

def add_mixed_row(slide, left_label, sentence_x, word, word_color,
                  ce_val, sim_val, y):
    """One row: [label]  This class is [WORD]   [CE]  [SIM]"""
    # role label
    add_textbox(slide, 0.3, y, 1.5, 0.45, left_label,
                font_size=14, color="#aaaaaa", italic=True)

    # "This class is " — plain text
    txBox = slide.shapes.add_textbox(
        Inches(sentence_x), Inches(y), Inches(3.8), Inches(0.45))
    tf = txBox.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]

    r1 = p.add_run()
    r1.text = "This class is "
    r1.font.size = Pt(20)
    r1.font.color.rgb = rgb("#333333")
    r1.font.name = "Calibri"

    r2 = p.add_run()
    r2.text = word
    r2.font.size = Pt(20)
    r2.font.bold = True
    r2.font.color.rgb = rgb(word_color)
    r2.font.name = "Calibri"

    # CE pill — neutral
    add_pill(slide, 5.15, y + 0.02, 0.55, 0.38, ce_val,
             font_size=16, text_color="#444444",
             bg_color="#f0f0f0", border_color="#cccccc")

    # Similarity pill — neutral
    add_pill(slide, 5.85, y + 0.02, 1.0, 0.38, sim_val,
             font_size=16, text_color="#444444",
             bg_color="#f0f0f0", border_color="#cccccc")

# ── Title ────────────────────────────────────────────────────────────────────
add_textbox(slide, 0.3, 0.25, 6.5, 0.7, "Overview of Method",
            font_size=28, bold=False, color="#111111")

# ── Section label ────────────────────────────────────────────────────────────
add_textbox(slide, 0.3, 1.25, 3.5, 0.35, "Next-token prediction:",
            font_size=13, bold=True, color="#888888")

# ── Column headers ───────────────────────────────────────────────────────────
add_textbox(slide, 5.2, 1.25, 0.6, 0.35, "CE",
            font_size=13, bold=True, color="#888888", align=PP_ALIGN.CENTER)
add_textbox(slide, 5.9, 1.25, 1.0, 0.35, "Similarity",
            font_size=13, bold=True, color="#888888", align=PP_ALIGN.CENTER)

# ── Three sentence rows ───────────────────────────────────────────────────────
rows = [
    ("Ground Truth", "challenging", "#f28e2b", "0",  "+ 0.0"),
    ("Prediction A", "difficult",   "#e15759", "1",  "+ 0.38"),
    ("Prediction B", "purple",      "#b07ecf", "1",  "+ 1.92"),
]
y_positions = [1.65, 2.45, 3.25]

for (label, word, color, ce, sim), y in zip(rows, y_positions):
    add_mixed_row(slide, label, 1.85, word, color, ce, sim, y)

# ── Bottom text ───────────────────────────────────────────────────────────────
add_textbox(slide, 0.3, 4.2, 6.5, 0.55,
            "Our approach: reward similarity",
            font_size=20, bold=True, color="#333333")

# Formula split across two lines in a pill box
add_pill(slide, 0.3, 4.85, 6.5, 0.85,
         "Loss = CE\n      + λ · embedding_similarity(gt, pred)",
         font_size=20, text_color="#333333",
         bg_color="#f7f7f7", border_color="#dddddd", font_name="Calibri")

# "our term" annotation
add_textbox(slide, 3.5, 5.75, 3.0, 0.35,
            "↑ our term",
            font_size=13, color="#999999", italic=True)

# ── Embedding chart image ─────────────────────────────────────────────────────
img_path = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/mock_embedding_3.png"
slide.shapes.add_picture(img_path,
    Inches(6.7), Inches(0.5), Inches(6.3), Inches(6.7))

out = "/Users/sttawm/dev/parameter-golf/embedding-loss/lama/overview_slide.pptx"
prs.save(out)
print(f"Saved → {out}")
