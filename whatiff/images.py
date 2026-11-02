"""Bounded image decoding and measurable scene-preservation evidence."""

import base64
import hashlib
import warnings
from io import BytesIO

from PIL import Image, ImageChops, ImageDraw, ImageStat

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 16_000_000
MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode(data):
    if not isinstance(data, bytes) or not 0 < len(data) <= MAX_BYTES:
        raise ValueError("Image size must be between 1 byte and 20 MB.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as image:
            if image.format not in MEDIA_TYPES or image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Require one PNG/JPEG/WebP frame under 16 megapixels.")
            if image.getexif().get(274, 1) != 1:
                raise ValueError("Normalize EXIF orientation before uploading.")
            media_type = MEDIA_TYPES[image.format]
            image.load()
            # Composite alpha consistently for pixel comparisons.
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", image.size, "white")
            background.alpha_composite(rgba)
            return background.convert("RGB"), media_type


def data_url(data):
    _, media_type = decode(data)
    return f"data:{media_type};base64," + base64.b64encode(data).decode("ascii")


def preservation(original, edited, regions):
    """Regions are curator-supplied normalized xyxy boxes, never agent ground truth."""
    before, _ = decode(original)
    after, _ = decode(edited)
    if before.size != after.size:
        return {"status": "invalid", "reason": "Image dimensions changed.", "original_size": list(before.size), "edited_size": list(after.size)}
    difference = ImageChops.difference(before, after)
    channel_max = ImageChops.lighter(ImageChops.lighter(*difference.split()[:2]), difference.split()[2])
    changed = channel_max.point(lambda value: 255 if value > 24 else 0)
    global_fraction = ImageStat.Stat(changed).mean[0] / 255
    result = {"pixel_threshold": 24, "global_changed_fraction": global_fraction, "original_size": list(before.size), "edited_size": list(after.size)}
    if difference.getbbox() is None:
        return {**result, "status": "invalid", "reason": "No visible pixel change."}
    if not regions:
        return {**result, "status": "unverified", "reason": "No curator-defined edit regions; preservation cannot be established."}
    outside = Image.new("L", before.size, 255)
    draw = ImageDraw.Draw(outside)
    for box in regions:
        if (len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not 0 <= v <= 1 for v in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError("Edit regions must be normalized xyxy boxes.")
        width, height = before.size
        draw.rectangle((int(box[0] * width), int(box[1] * height), min(width - 1, int(box[2] * width)), min(height - 1, int(box[3] * height))), fill=0)
    outside_count = sum(outside.histogram()[1:])
    coverage = 1 - outside_count / (before.width * before.height)
    if coverage > 0.5:
        return {**result, "status": "unverified", "reason": "Edit regions cover over half the scene.", "region_fraction": coverage}
    masked = ImageChops.multiply(changed, outside)
    fraction = sum(masked.histogram()[1:]) / outside_count
    mean_difference = sum(ImageStat.Stat(difference, outside).mean) / 3
    return {**result, "status": "invalid" if fraction > 0.02 or mean_difference > 4 else "consistent",
            "reason": "Outside-region pixel comparison (not a semantic or physical guarantee).",
            "outside_changed_fraction": fraction, "outside_mean_difference": mean_difference,
            "max_outside_changed_fraction": 0.02, "max_outside_mean_difference": 4,
            "region_fraction": coverage, "regions": regions}
