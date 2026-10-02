"""Render a Figure 6 TikZ layout from PDF to the bitmap set."""
import sys
from pathlib import Path

import pymupdf
from PIL import Image

HERE = Path(__file__).resolve().parent
TARGET_W = 2261


def main(stem="figure6_pipeline"):
    pdf = HERE / f"{stem}.pdf"
    if not pdf.exists():
        sys.exit(f"compile {stem}.tex with pdflatex first (from figures/)")
    page = pymupdf.open(pdf)[0]
    zoom = TARGET_W / page.rect.width
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    out = HERE / stem
    pix.save(out.with_suffix(".png"))
    pix.save(out.with_suffix(".jpg"), jpg_quality=95)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    img.save(out.with_suffix(".tiff"), compression="tiff_lzw", dpi=(300, 300))
    print("    -> %s.{png,jpg,tiff}  %dx%d" % (stem, pix.width, pix.height))


if __name__ == "__main__":
    main(*sys.argv[1:2])
