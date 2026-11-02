import unittest
from io import BytesIO

from PIL import Image

from whatiff.contracts import Action, Proposal
from whatiff.controlled import SamOcclusionGenerator, verify_occlusion


class FixtureSegmenter:
    def segment(self, image, concept):
        return {"boxes": [{"x1": 40, "y1": 30, "x2": 100, "y2": 90, "w": 160, "h": 120}], "mode": "mock"}


class ControlledTests(unittest.TestCase):
    def setUp(self):
        output = BytesIO()
        Image.new("RGB", (160, 120), "white").save(output, format="PNG")
        self.original = output.getvalue()
        self.proposal = Proposal("occlusion", "Partial sensor occlusion", "Test only", Action.PICK_TARGET)
        self.generator = SamOcclusionGenerator(FixtureSegmenter())

    def test_exact_preservation_and_partial_visible_edit(self):
        edited = self.generator.generate(self.original, "Task", self.proposal)
        result = verify_occlusion(self.original, edited.image, edited.raw)
        self.assertTrue(result["passed"])
        self.assertEqual(result["outside_changed_pixels"], 0)
        self.assertGreater(result["changed_pixels"], 0)
        self.assertAlmostEqual(result["occluded_box_fraction"], .13)

    def test_unrelated_pixel_change_fails_even_below_tolerance(self):
        edited = self.generator.generate(self.original, "Task", self.proposal)
        image = Image.open(BytesIO(edited.image))
        image.putpixel((0, 0), (254, 255, 255))
        output = BytesIO()
        image.save(output, format="PNG")
        self.assertFalse(verify_occlusion(self.original, output.getvalue(), edited.raw)["passed"])

    def test_unchanged_image_fails(self):
        edited = self.generator.generate(self.original, "Task", self.proposal)
        self.assertFalse(verify_occlusion(self.original, self.original, edited.raw)["passed"])

    def test_multiple_or_missing_targets_rejected(self):
        for boxes in ([], FixtureSegmenter().segment(None, None)["boxes"] * 2):
            self.generator.segmenter.segment = lambda image, concept: {"boxes": boxes}
            with self.subTest(count=len(boxes)), self.assertRaises(ValueError):
                self.generator.generate(self.original, "Task", self.proposal)

    def test_scope_and_metadata_cannot_expand_edit(self):
        with self.assertRaises(ValueError):
            self.generator.generate(self.original, "Task", Proposal("relocation", "", "", Action.PICK_TARGET))
        edited = self.generator.generate(self.original, "Task", self.proposal)
        edited.raw["rectangle_xyxy"][2] += 1
        with self.assertRaises(ValueError):
            verify_occlusion(self.original, edited.image, edited.raw)

    def test_grounding_resolution_mismatch_rejected(self):
        box = FixtureSegmenter().segment(None, None)["boxes"][0]
        self.generator.segmenter.segment = lambda image, concept: {"boxes": [{**box, "w": 999}]}
        with self.assertRaises(ValueError):
            self.generator.generate(self.original, "Task", self.proposal)
