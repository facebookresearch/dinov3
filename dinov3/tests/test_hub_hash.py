# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from torch import nn

from dinov3.hub import classifiers, depthers, dinotxt


class HubHashTest(unittest.TestCase):
    def test_classifier_rejects_incorrect_head_hash(self):
        backbone = nn.Identity()
        backbone.embed_dim = 2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "head-00000000.pth"
            torch.save(nn.Linear(4, 1000).state_dict(), weights)
            with patch.object(classifiers, "dinov3_vit7b16", return_value=backbone):
                with patch("torch.hub.get_dir", return_value=str(root / "unchecked")):
                    classifiers.dinov3_vit7b16_lc(weights=str(weights), check_hash=False)
                with patch("torch.hub.get_dir", return_value=str(root / "checked")):
                    with self.assertRaisesRegex(RuntimeError, "invalid hash value"):
                        classifiers.dinov3_vit7b16_lc(weights=str(weights), check_hash=True)

    def test_depther_forwards_hash_check_to_backbone(self):
        for check_hash in (False, True):
            with self.subTest(check_hash=check_hash):
                backbone = Mock()
                with (
                    patch.dict(depthers._BACKBONE_DICT, {"dinov3_vit7b16": backbone}),
                    patch.object(depthers, "make_depther_from_config"),
                    patch.object(depthers, "_safe_load_state_dict_from_url", return_value={}),
                ):
                    depthers.dinov3_vit7b16_dd(check_hash=check_hash)
                backbone.assert_called_once()
                self.assertEqual(backbone.call_args.kwargs.get("check_hash", False), check_hash)

    def test_chmv2_forwards_hash_check_to_backbone(self):
        for check_hash in (False, True):
            with self.subTest(check_hash=check_hash):
                with (
                    patch.object(depthers, "dinov3_vitl16") as backbone,
                    patch.object(depthers, "make_depther_from_config"),
                    patch.object(depthers, "_safe_load_state_dict_from_url", return_value={}),
                ):
                    depthers.dinov3_vitl16_chmv2(check_hash=check_hash)
                backbone.assert_called_once()
                self.assertEqual(backbone.call_args.kwargs.get("check_hash", False), check_hash)

    def test_dinotxt_forwards_hash_check_to_backbone(self):
        for check_hash in (False, True):
            with self.subTest(check_hash=check_hash):
                with (
                    patch.object(dinotxt, "dinov3_vitl16") as backbone,
                    patch("dinov3.eval.text.dinotxt_model.DINOTxt"),
                    patch("dinov3.eval.text.text_transformer.TextTransformer"),
                    patch("dinov3.eval.text.tokenizer.get_tokenizer"),
                    patch.object(dinotxt, "_safe_load_state_dict_from_url", return_value={}),
                ):
                    dinotxt.dinov3_vitl16_dinotxt_tet1280d20h24l(check_hash=check_hash)
                backbone.assert_called_once()
                self.assertEqual(backbone.call_args.kwargs.get("check_hash", False), check_hash)


if __name__ == "__main__":
    unittest.main()
