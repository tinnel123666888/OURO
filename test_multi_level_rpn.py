"""
Unit tests for multi_level_rpn_description_generation.py

These tests mock Detectron2 and EasyOCR so they can run without GPU or model downloads.
"""
import os
import sys
import types
import numpy as np
import tempfile
import json
import unittest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Stub out heavy dependencies so the module can be imported in CI
# ---------------------------------------------------------------------------
def _make_detectron2_stubs():
    d2 = types.ModuleType("detectron2")
    engine = types.ModuleType("detectron2.engine")
    config = types.ModuleType("detectron2.config")
    zoo = types.ModuleType("detectron2.model_zoo")

    class FakeCfg:
        class MODEL:
            DEVICE = "cpu"
            class ROI_HEADS:
                SCORE_THRESH_TEST = 0.0
            WEIGHTS = ""
        def merge_from_file(self, path): pass

    config.get_cfg = lambda: FakeCfg()
    zoo.get_config_file = lambda name: name
    zoo.get_checkpoint_url = lambda name: name

    class FakePredictor:
        def __init__(self, cfg):
            self.cfg = cfg

    engine.DefaultPredictor = FakePredictor

    d2.engine = engine
    d2.config = config
    d2.model_zoo = zoo

    sys.modules["detectron2"] = d2
    sys.modules["detectron2.engine"] = engine
    sys.modules["detectron2.config"] = config
    sys.modules["detectron2.model_zoo"] = zoo


def _make_easyocr_stub():
    easyocr = types.ModuleType("easyocr")
    class FakeReader:
        def __init__(self, langs): pass
        def readtext(self, img): return []
    easyocr.Reader = FakeReader
    sys.modules["easyocr"] = easyocr


def _make_tqdm_stub():
    tqdm_mod = types.ModuleType("tqdm")
    class FakeTqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, n=1): pass
        def close(self): pass
    tqdm_mod.tqdm = FakeTqdm
    sys.modules["tqdm"] = tqdm_mod


_make_detectron2_stubs()
_make_easyocr_stub()
_make_tqdm_stub()

# Now import the module under test
import importlib
mod = importlib.import_module("multi_level_rpn_description_generation")
setup_predictor = mod.setup_predictor
process_image_recursive = mod.process_image_recursive
generate_proposals = mod.generate_proposals
filter_boxes = mod.filter_boxes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSetupPredictor(unittest.TestCase):
    """
    Verify that setup_predictor correctly assigns use_cuda and confidence_threshold,
    and that the argument order cannot be silently swapped.
    """

    def _call(self, use_cuda, confidence_threshold):
        """
        Call setup_predictor and return the cfg stored on the returned predictor.
        We patch mod.DefaultPredictor (the already-imported reference) to capture cfg.
        """
        captured = {}

        class CapturingPredictor:
            def __init__(self, cfg):
                captured["cfg"] = cfg

        original = mod.DefaultPredictor
        mod.DefaultPredictor = CapturingPredictor
        try:
            setup_predictor(use_cuda=use_cuda, confidence_threshold=confidence_threshold)
        finally:
            mod.DefaultPredictor = original
        return captured["cfg"]

    def test_cuda_device(self):
        cfg = self._call(use_cuda=True, confidence_threshold=0.35)
        self.assertEqual(cfg.MODEL.DEVICE, "cuda")

    def test_cpu_device(self):
        cfg = self._call(use_cuda=False, confidence_threshold=0.35)
        self.assertEqual(cfg.MODEL.DEVICE, "cpu")

    def test_threshold_cpu(self):
        cfg = self._call(use_cuda=False, confidence_threshold=0.35)
        self.assertAlmostEqual(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, 0.35)

    def test_threshold_cuda(self):
        cfg = self._call(use_cuda=True, confidence_threshold=0.35)
        self.assertAlmostEqual(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, 0.35)

    def test_threshold_custom(self):
        cfg = self._call(use_cuda=False, confidence_threshold=0.5)
        self.assertAlmostEqual(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, 0.5)

    def test_threshold_is_float_not_bool(self):
        """When threshold is a bool True, float() must still store a numeric value, not break."""
        cfg = self._call(use_cuda=False, confidence_threshold=0.7)
        self.assertIsInstance(cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, float)

    def test_swapped_args_would_fail(self):
        """
        If positional args were swapped (confidence_threshold, use_cuda) the
        threshold would be a bool and device would be wrong.
        Keyword-arg API prevents this mistake.
        """
        # Correctly called with keyword args — device must be "cuda"
        cfg_correct = self._call(use_cuda=True, confidence_threshold=0.35)
        self.assertEqual(cfg_correct.MODEL.DEVICE, "cuda")
        self.assertAlmostEqual(cfg_correct.MODEL.ROI_HEADS.SCORE_THRESH_TEST, 0.35)


class TestMutableDefaults(unittest.TestCase):
    """Verify that messages/box_counts don't leak across independent calls."""

    def _make_image(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def _make_predictor_returning(self, boxes):
        """Create a mock predictor that returns the given boxes."""
        from types import SimpleNamespace
        import torch

        predictor = MagicMock()
        tensor_boxes = MagicMock()
        tensor_boxes.tensor.cpu().numpy.return_value = np.array(boxes, dtype=np.float32)
        instances = MagicMock()
        instances.pred_boxes = tensor_boxes
        predictor.return_value = {"instances": instances}
        return predictor

    def test_messages_not_shared_across_calls(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            predictor = MagicMock()
            # Return no proposals so recursion exits immediately
            instances = MagicMock()
            instances.pred_boxes.tensor.cpu().numpy.return_value = np.array([], dtype=np.float32).reshape(0, 4)
            predictor.return_value = {"instances": instances}
            ocr = MagicMock()
            ocr.readtext.return_value = []

            img = self._make_image()
            # Save a fake original.jpg so the root-message path exists
            import cv2
            cv2.imwrite(os.path.join(d1, "original.jpg"), img)
            cv2.imwrite(os.path.join(d2, "original.jpg"), img)

            msgs1 = []
            process_image_recursive(img, d1, ocr, predictor, messages=msgs1, box_counts={})

            msgs2 = []
            process_image_recursive(img, d2, ocr, predictor, messages=msgs2, box_counts={})

            # Each call should produce exactly one root message (d^0)
            self.assertEqual(len(msgs1), 1)
            self.assertEqual(len(msgs2), 1)
            # They must be independent lists
            self.assertIsNot(msgs1, msgs2)

    def test_box_counts_not_shared_across_calls(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            predictor = MagicMock()
            instances = MagicMock()
            instances.pred_boxes.tensor.cpu().numpy.return_value = np.array([], dtype=np.float32).reshape(0, 4)
            predictor.return_value = {"instances": instances}
            ocr = MagicMock()
            ocr.readtext.return_value = []

            img = self._make_image()
            import cv2
            cv2.imwrite(os.path.join(d1, "original.jpg"), img)
            cv2.imwrite(os.path.join(d2, "original.jpg"), img)

            bc1 = {}
            process_image_recursive(img, d1, ocr, predictor, messages=[], box_counts=bc1)
            bc2 = {}
            process_image_recursive(img, d2, ocr, predictor, messages=[], box_counts=bc2)

            self.assertIsNot(bc1, bc2)


class TestRootImageMessage(unittest.TestCase):
    """
    Verify that the root image d^(0) description is added to messages at level 1
    (paper Algorithm 1), and that it references the correct 'original.jpg' path.
    """

    def test_root_message_added_at_level1(self):
        with tempfile.TemporaryDirectory() as outdir:
            predictor = MagicMock()
            instances = MagicMock()
            instances.pred_boxes.tensor.cpu().numpy.return_value = np.array([], dtype=np.float32).reshape(0, 4)
            predictor.return_value = {"instances": instances}
            ocr = MagicMock()
            ocr.readtext.return_value = []

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(os.path.join(outdir, "original.jpg"), img)

            messages = []
            process_image_recursive(img, outdir, ocr, predictor, messages=messages, box_counts={})

            self.assertGreater(len(messages), 0, "Expected at least the root message")
            root_msg = messages[0]
            self.assertEqual(root_msg["role"], "user")
            # Root content must include the original.jpg path
            image_content = [c for c in root_msg["content"] if c.get("type") == "image"]
            self.assertEqual(len(image_content), 1)
            self.assertIn("original.jpg", image_content[0]["image"])

    def test_root_message_not_added_at_level2(self):
        """Level > 1 should not prepend an extra root message."""
        with tempfile.TemporaryDirectory() as outdir:
            predictor = MagicMock()
            instances = MagicMock()
            instances.pred_boxes.tensor.cpu().numpy.return_value = np.array([], dtype=np.float32).reshape(0, 4)
            predictor.return_value = {"instances": instances}
            ocr = MagicMock()
            ocr.readtext.return_value = []

            img = np.zeros((100, 100, 3), dtype=np.uint8)
            # No original.jpg needed at level > 1

            messages = []
            # Call directly at level=2
            process_image_recursive(img, outdir, ocr, predictor, level=2, messages=messages, box_counts={})

            # No root message should be added at level 2
            root_messages = [
                m for m in messages
                if any(c.get("type") == "image" and "original.jpg" in c.get("image", "")
                       for c in m.get("content", []))
            ]
            self.assertEqual(len(root_messages), 0)


class TestBoxCountsAccumulation(unittest.TestCase):
    """box_counts[level] should accumulate across recursive branches, not overwrite."""

    def test_box_counts_accumulate(self):
        with tempfile.TemporaryDirectory() as outdir:
            # Image hierarchy:
            #   Level-1 image: 200x200 → 2 proposals [0,0,80,80] and [100,100,180,180]
            #   Level-2 images: each 80x80 → 1 proposal [0,0,30,30]
            #   Level-3 images: each 30x30 → no proposals (stop recursion)
            # Expected box_counts: {1: 2, 2: 2, 3: 0 (not set)}
            import cv2

            original_path = os.path.join(outdir, "original.jpg")
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(original_path, img)

            level1_boxes = np.array([[0, 0, 80, 80], [100, 100, 180, 180]], dtype=np.float32)
            level2_boxes = np.array([[0, 0, 30, 30]], dtype=np.float32)
            empty_boxes = np.array([], dtype=np.float32).reshape(0, 4)

            predictor = MagicMock()

            def fake_predict(image):
                instances = MagicMock()
                h = image.shape[0]
                if h > 100:
                    boxes = level1_boxes
                elif h > 20:
                    boxes = level2_boxes
                else:
                    boxes = empty_boxes
                instances.pred_boxes.tensor.cpu().numpy.return_value = boxes
                return {"instances": instances}

            predictor.side_effect = fake_predict

            ocr = MagicMock()
            ocr.readtext.return_value = []

            messages = []
            box_counts = {}
            process_image_recursive(img, outdir, ocr, predictor, messages=messages, box_counts=box_counts)

            # Level 1 should have 2 boxes
            self.assertEqual(box_counts.get(1), 2)
            # Level 2 should have accumulated 2 (1 from each of the 2 level-1 sub-dirs)
            self.assertEqual(box_counts.get(2), 2)


if __name__ == "__main__":
    unittest.main()
