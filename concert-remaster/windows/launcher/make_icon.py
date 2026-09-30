"""Draw the app icon (same design as gui/static/icon.svg) as a multi-size Windows .ico."""
import sys

from PIL import Image, ImageDraw

BARS = [(12, 26, 12), (21, 18, 28), (30, 10, 44), (39, 20, 24), (48, 28, 8)]  # x, y, height on a 64 grid


def draw(size: int) -> Image.Image:
    scale = 4  # draw big, then shrink for smooth edges
    s = size * scale
    k = s / 64
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=14 * k, fill=(17, 19, 26, 255))
    grad = Image.new("RGBA", (s, s))
    a, b = (139, 92, 246), (34, 211, 238)
    for y in range(s):
        for_x = [(int(a[i] + (b[i] - a[i]) * (y / s)) ) for i in range(3)]
        ImageDraw.Draw(grad).line([(0, y), (s, y)], fill=(*for_x, 255))
    mask = Image.new("L", (s, s), 0)
    m = ImageDraw.Draw(mask)
    for x, y, h in BARS:
        m.rounded_rectangle([x * k, y * k, (x + 5) * k, (y + h) * k], radius=2.5 * k, fill=255)
    img.paste(grad, (0, 0), mask)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
    big = draw(256)
    big.save(sys.argv[1], format="ICO", sizes=[(n, n) for n in sizes], append_images=[draw(n) for n in sizes[:-1]])
