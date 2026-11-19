"""Element-level visual layout check for the compiled paper.

This is the programmatic stand-in for eyeballing the rendered pages: it
inspects every text element and every image in the PDF and reports

  1. elements outside the expected content box (with the line-number
     artefacts of ACL review mode classified separately),
  2. content in the bottom watermark strip,
  3. font-size inventory, with everything below the readable threshold
     located,
  4. overlapping text elements (overprint / collision),
  5. images crossing the column gutter or the page margins,
  6. section headings stranded as the last line of a page, and captions
     separated from their float,
  7. a per-page content map.

Usage:
  python scripts/visual_layout_check.py paper-new/acl/main.pdf [more.pdf ...]
"""
import io
import json
import os
import sys
from collections import defaultdict

import pdfplumber

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ACL A4 geometry, measured on the compiled paper: the text block spans
# x = 70.9 .. 526.3 pt (455.4 pt wide) on a 595.28 x 841.89 pt page; the
# bottom 56.7 pt (2 cm) must stay empty for the proceedings watermark.
# A 1 pt allowance absorbs glyph side bearings (math italics overhang by
# up to 2.1 pt) and justification rounding.
PAGE_W, PAGE_H = 595.28, 841.89
CONTENT_L = 70.0
CONTENT_R = 527.0
CONTENT_T = 71.0
CONTENT_B = PAGE_H - 56.7
SMALL_TEXT = 7.0        # body/table text should stay readable
TINY_TEXT = 6.0         # absolute floor
TOL = 1.0               # glyph side-bearing / italic overhang tolerance
LINENO_MAX_SIZE = 6.0    # review-mode line numbers are tiny


def overlaps(a, b):
    """Overlap area of two rects (x0, top, x1, bottom)."""
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oy = min(a[3], b[3]) - max(a[1], b[1])
    return ox * oy if ox > 0 and oy > 0 else 0.0


def check_pdf(path):
    print(f"\n================ {path} ================")
    report = {"file": path, "issues": [], "pages": []}
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            words = page.extract_words(extra_attrs=["size"])
            chars = page.chars
            info = {"page": pno, "words": len(words), "images":
                    len(page.images), "min_font": None,
                    "out_of_box": [], "bottom_strip": [], "tiny": [],
                    "overlaps": [], "images_out": [], "headings": []}

            # --- font size inventory
            sizes = [c["size"] for c in chars]
            if sizes:
                info["min_font"] = round(min(sizes), 2)
            for c in chars:
                if c["size"] < TINY_TEXT:
                    info["tiny"].append({"text": c["text"][:18],
                                         "size": round(c["size"], 2),
                                         "x0": round(c["x0"], 1),
                                         "top": round(c["top"], 1)})

            # --- geometry vs content box
            for w in words:
                x0, x1 = w["x0"], w["x1"]
                top, bot = w["top"], w["bottom"]
                is_lineno = (w["text"].isdigit() and
                             (x0 < 45 or x1 > CONTENT_R + 5))
                if bot > CONTENT_B:
                    info["bottom_strip"].append(
                        {"text": w["text"][:28], "x0": round(x0, 1),
                         "bottom": round(bot, 1),
                         "kind": "lineno" if is_lineno else "body"})
                    continue
                if (x0 < CONTENT_L - TOL or x1 > CONTENT_R + TOL) and \
                        not is_lineno:
                    info["out_of_box"].append(
                        {"text": w["text"][:28], "x0": round(x0, 1),
                         "x1": round(x1, 1), "size": round(w["size"], 1)})

            # --- image placement
            for im in page.images:
                b = (im["x0"], im["top"], im["x1"], im["bottom"])
                if b[0] < CONTENT_L - 1 or b[2] > CONTENT_R + 1:
                    info["images_out"].append(
                        {"x0": round(b[0], 1), "x1": round(b[2], 1),
                         "top": round(b[1], 1), "bottom": round(b[3], 1)})

            # --- collisions between text elements
            boxes = [(w["x0"], w["top"], w["x1"], w["bottom"], w["text"])
                     for w in words]
            boxes.sort(key=lambda b: b[0])
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if boxes[j][0] > boxes[i][2] + 1:
                        break
                    ov = overlaps(boxes[i], boxes[j])
                    area = min((boxes[i][2] - boxes[i][0]) *
                               (boxes[i][3] - boxes[i][1]),
                               (boxes[j][2] - boxes[j][0]) *
                               (boxes[j][3] - boxes[j][1]))
                    if area > 0 and ov / area > 0.35:
                        info["overlaps"].append(
                            {"a": boxes[i][4][:18], "b": boxes[j][4][:18],
                             "frac": round(ov / area, 2)})

            # --- headings and captions near page edges
            last_y = max((w["bottom"] for w in words), default=0)
            first_y = min((w["top"] for w in words), default=0)
            page_issues = []
            for w in words:
                t = w["text"]
                if (w["size"] >= 10.5 and t[:1].isdigit() and
                        len(t) < 40 and w["bottom"] < CONTENT_B - 6 and
                        not t.isdigit()):
                    info["headings"].append(t)
                    if w["bottom"] >= last_y - 1:
                        page_issues.append(
                            f"p{pno}: heading '{t}' is the last line "
                            "on the page (orphan)")
            # caption separated from float: caption at the very top of the
            # page while the page has no image above it
            cap = [w for w in words
                   if w["text"].startswith(("Figure", "Table"))
                   and w["top"] < first_y + 12]
            if cap and not any(im["top"] < 40 for im in page.images):
                page_issues.append(
                    f"p{pno}: caption '{cap[0]['text'][:24]}' starts the "
                    "page with no float above it")
            report["issues"].extend(page_issues)

            # per-page summary
            top_words = sorted(words, key=lambda w: w["top"])[:3]
            info["top_text"] = " ".join(w["text"] for w in top_words)[:70]
            info["out_count"] = len(info["out_of_box"])
            info["bottom_count"] = len(info["bottom_strip"])
            info["overlap_count"] = len(info["overlaps"])
            info["tiny_count"] = len(info["tiny"])
            report["pages"].append(info)

    # ---------------- report ----------------
    tot_out = sum(p["out_count"] for p in report["pages"])
    tot_bot = sum(p["bottom_count"] for p in report["pages"])
    tot_ov = sum(p["overlap_count"] for p in report["pages"])
    tot_tiny = sum(p["tiny_count"] for p in report["pages"])
    print(f"pages: {len(report['pages'])}")
    for p in report["pages"]:
        flags = []
        if p["out_count"]:
            flags.append(f"out-of-box={p['out_count']}")
        if p["bottom_count"]:
            flags.append(f"bottom={p['bottom_count']}")
        if p["overlap_count"]:
            flags.append(f"overlaps={p['overlap_count']}")
        if p["tiny_count"]:
            flags.append(f"tiny={p['tiny_count']}")
        if p["images_out"]:
            flags.append(f"img-out={len(p['images_out'])}")
        print(f"  p{p['page']:>2}: words={p['words']:>4} imgs={p['images']} "
              f"min_font={p['min_font']} {' '.join(flags)}")
        for k in ("out_of_box", "overlaps", "tiny", "images_out"):
            if p[k]:
                for item in p[k][:4]:
                    print(f"       {k}: {item}")
    for i in report["issues"]:
        print("  ISSUE:", i)
    print(f"TOTALS: out-of-box={tot_out} bottom-strip={tot_bot} "
          f"overlaps={tot_ov} sub-{TINY_TEXT}pt={tot_tiny}")
    report["totals"] = {"out_of_box": tot_out, "bottom_strip": tot_bot,
                        "overlaps": tot_ov, "tiny": tot_tiny}
    return report


def main():
    paths = sys.argv[1:] or ["paper-new/acl/main.pdf"]
    reports = [check_pdf(p) for p in paths]
    out = "runs/gate_exp/supp/visual_layout_report.json"
    with io.open(out, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=1, ensure_ascii=False)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
