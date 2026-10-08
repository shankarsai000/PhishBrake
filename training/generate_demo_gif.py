from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

initial_path = DOCS / "phishbrake_initial.png"
usps_path = DOCS / "phishbrake_usps_demo.png"
family_path = DOCS / "phishbrake_family_demo.png"

img_initial = Image.open(initial_path).convert("RGB")
img_usps = Image.open(usps_path).convert("RGB")
img_family = Image.open(family_path).convert("RGB")

target_size = (1280, 720)


def prepare_slide(img: Image.Image, banner_title: str, banner_subtitle: str) -> Image.Image:
    canvas = Image.new("RGB", target_size, "#0d1b2a")
    
    # Scale image to fit within viewport
    scale = min((target_size[0] - 40) / img.width, (target_size[1] - 120) / img.height)
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Paste image in center below banner
    pos_x = (target_size[0] - new_w) // 2
    pos_y = 100 + (target_size[1] - 120 - new_h) // 2
    canvas.paste(resized, (pos_x, pos_y))
    
    # Draw Top Banner Bar
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (target_size[0], 85)], fill="#102638", outline="#25465c")
    
    # Draw Green Accent Strip
    draw.rectangle([(0, 0), (8, 85)], fill="#10b981")
    
    # Draw Text
    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_sub = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    draw.text((25, 15), banner_title, fill="#ffffff", font=font_title)
    draw.text((25, 52), banner_subtitle, fill="#10b981", font=font_sub)
    
    return canvas


slide1 = prepare_slide(
    img_initial,
    "PhishBrake — Private Local Scam Defense",
    "Step 1: Paste any suspicious text message, email, DM, or upload a QR code."
)

slide2 = prepare_slide(
    img_usps,
    "Demo 1: Package Phishing (USPS Scam)",
    "Step 2: Instant risk verdict (CRITICAL), Scam DNA, tactic badges & safest next step."
)

slide3 = prepare_slide(
    img_family,
    "Demo 2: Family Impersonation Scam",
    "Step 3: High-pressure impersonation detection + 1-Click Trusted Contact Note."
)

frames = []

# Slide 1 hold
for _ in range(25):
    frames.append(slide1)

# Slide 2 hold
for _ in range(35):
    frames.append(slide2)

# Slide 3 hold
for _ in range(35):
    frames.append(slide3)

output_gif = DOCS / "phishbrake_demo.gif"
frames[0].save(
    output_gif,
    save_all=True,
    append_images=frames[1:],
    duration=100,  # 100ms per frame
    loop=0
)

print(f"Successfully generated animated demo GIF: {output_gif} ({output_gif.stat().st_size} bytes)")
