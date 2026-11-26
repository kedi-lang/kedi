"""Draw the receipt images and their ground truth, deterministically.

    python examples/local_vision/make_receipts.py

The receipts are synthetic: shop names, items and prices are invented. Text uses the font
bundled with Pillow (Aileron, SIL Open Font License). One receipt's printed total is wrong
on purpose, so the program's check has something to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent

RECEIPTS = {
    "cafe": ("CORNER CAFE", [("Latte", 4.50), ("Croissant", 3.20), ("Orange juice", 3.90)], None),
    "books": ("PAGE TURNER BOOKS", [("Notebook", 6.00), ("Novel", 14.99)], None),
    "pharmacy": (
        "GREEN CROSS PHARMACY",
        [("Plasters", 2.49), ("Vitamin C", 7.99), ("Hand cream", 5.25), ("Tissues", 1.10)],
        None,
    ),
    "garage": ("RIVERSIDE GARAGE", [("Oil change", 49.00), ("Wiper blades", 18.50)], None),
    "bakery": ("SUNRISE BAKERY", [("Sourdough", 5.80), ("Cinnamon roll", 2.90)], 9.70),
    "hardware": (
        "NUTS AND BOLTS",
        [("Hammer", 12.40), ("Screws", 3.15), ("Tape measure", 8.99)],
        None,
    ),
}


def draw(merchant: str, items: list[tuple[str, float]], total: float) -> Image.Image:
    font = ImageFont.load_default(size=32)
    image = Image.new("RGB", (560, 160 + 64 * (len(items) + 2)), "white")
    pen = ImageDraw.Draw(image)
    pen.text((40, 40), merchant, fill="black", font=font)
    y = 130
    for name, price in items:
        pen.text((40, y), name, fill="black", font=font)
        pen.text((400, y), f"{price:.2f}", fill="black", font=font)
        y += 64
    pen.text((40, y + 24), "TOTAL", fill="black", font=font)
    pen.text((400, y + 24), f"{total:.2f}", fill="black", font=font)
    return image


def main() -> None:
    truth = {}
    for key, (merchant, items, printed) in RECEIPTS.items():
        total = printed if printed is not None else round(sum(price for _, price in items), 2)
        draw(merchant, items, total).save(HERE / "receipts" / f"{key}.png")
        lines = [merchant, *(f"{name} {price:.2f}" for name, price in items), f"TOTAL {total:.2f}"]
        truth[key] = {
            "merchant": merchant,
            "items": [{"name": name, "price": price} for name, price in items],
            "total": total,
            "adds_up": printed is None,
            "text": "\n".join(lines),
        }
    (HERE / "receipts" / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")


if __name__ == "__main__":
    main()
