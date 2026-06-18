import unittest
import sys
import types

sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault("torch", types.SimpleNamespace(Tensor=()))

from kwj_impact_nodes import SEGSFilterClosestMask


class SEGSFilterClosestMaskTests(unittest.TestCase):
    def test_empty_segments_preserve_source_shape(self):
        node = SEGSFilterClosestMask()

        filtered_segs, best_score = node.doit(
            ((512, 768), []),
            object(),
            "IoU",
        )

        self.assertEqual(filtered_segs, ((512, 768), []))
        self.assertEqual(best_score, 0.0)

    def test_malformed_segments_fall_back_to_zero_shape(self):
        node = SEGSFilterClosestMask()

        filtered_segs, best_score = node.doit(
            (),
            object(),
            "IoU",
        )

        self.assertEqual(filtered_segs, ((0, 0), []))
        self.assertEqual(best_score, 0.0)


if __name__ == "__main__":
    unittest.main()
