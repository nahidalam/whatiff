"""One controlled recipe: a SAM-grounded observation-only occlusion.

This paints a sensor occlusion, not a new physical obstacle. Muse Image remains
an optional separate Generator implementation. No general editing framework.
"""

from io import BytesIO

from PIL import Image, ImageChops, ImageDraw

from .contracts import GeneratedObservation
from .images import decode, digest


def occlusion_rectangle(box, size):
    width, height = size
    if (box["w"], box["h"]) != size:
        raise ValueError("SAM coordinates must match the original resolution.")
    x1, y1, x2, y2 = (box[key] for key in ("x1", "y1", "x2", "y2"))
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x1, y1, x2, y2)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError("Invalid SAM box.")
    # Half-open xyxy rectangle covering about 13% of the grounded box, with
    # its rightmost 78% (including this fixture's handle) left untouched.
    rectangle = (x1, y1 + int((y2 - y1) * .2), x1 + int((x2 - x1) * .22), y1 + int((y2 - y1) * .8))
    if rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1]:
        raise ValueError("Target region too small for controlled occlusion.")
    return rectangle


class SamOcclusionGenerator:
    def __init__(self, segmenter, concept="white mug"):
        self.segmenter, self.concept = segmenter, concept
        self.settings = {"backend": "pillow_controlled_occlusion", "recipe_version": "occlusion-v1",
                         "concept": concept, "color": [66, 75, 84], "scope": "observation_only"}

    def generate(self, image, task, proposal):
        if proposal.intervention != "occlusion":
            raise ValueError("This bounded backend implements only partial occlusion.")
        before, _ = decode(image)
        grounding = self.segmenter.segment(image, self.concept)
        if len(grounding["boxes"]) != 1:
            raise ValueError("Controlled edit requires exactly one grounded target.")
        box = grounding["boxes"][0]
        rectangle = occlusion_rectangle(box, before.size)
        edited = before.copy()
        x1, y1, x2, y2 = rectangle
        ImageDraw.Draw(edited).rectangle((x1, y1, x2 - 1, y2 - 1), fill=tuple(self.settings["color"]))
        output = BytesIO()
        edited.save(output, format="PNG")
        return GeneratedObservation(output.getvalue(), "image/png",
                                    "Opaque sensor-layer occlusion over the left portion of the grounded target; world state unchanged by construction.",
                                    {"settings": self.settings, "grounding": grounding, "box": box,
                                     "rectangle_xyxy": list(rectangle), "original_sha256": digest(image)})


def verify_occlusion(original, edited, metadata):
    before, _ = decode(original)
    after, _ = decode(edited)
    if metadata["original_sha256"] != digest(original) or before.size != after.size:
        raise ValueError("Image identity or dimensions changed.")
    rectangle = occlusion_rectangle(metadata["box"], before.size)
    if list(rectangle) != metadata["rectangle_xyxy"]:
        raise ValueError("Edit region does not match the fixed recipe.")
    x1, y1, x2, y2 = rectangle
    difference = ImageChops.difference(before, after)
    outside = difference.copy()
    ImageDraw.Draw(outside).rectangle((x1, y1, x2 - 1, y2 - 1), fill=(0, 0, 0))
    changed = sum(pixel != (0, 0, 0) for pixel in difference.crop(rectangle).getdata())
    area = (x2 - x1) * (y2 - y1)
    box = metadata["box"]
    box_area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
    checks = {"outside_pixels_identical": outside.getbbox() is None,
              "visible_occlusion": changed / area >= .95,
              "small_partial_occlusion": .08 <= area / box_area <= .18,
              "right_target_region_unchanged": before.crop((x2, box["y1"], box["x2"], box["y2"])).tobytes() == after.crop((x2, box["y1"], box["x2"], box["y2"])).tobytes()}
    return {"checks": checks, "passed": all(checks.values()), "rectangle_xyxy": list(rectangle),
            "occluded_box_fraction": area / box_area, "changed_pixels": changed,
            "outside_changed_pixels": sum(pixel != (0, 0, 0) for pixel in outside.getdata()),
            "limitation": "Bounding-box fraction is not segmented mug area. Identifiability needs scene review; this does not establish physical grasp feasibility."}
