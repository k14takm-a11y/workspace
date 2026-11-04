"""
pytorch実装のFRCNNで物体検出を行うモジュール
"""
import json
import sys
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from wrs_detector.detector_utils import get_bbox_dict


class FasterRcnnDetector:
    """
    Faster R-CNNで物体検出を行うクラス
    """
    def __init__(self, class_config_path, checkpoint_path=None):
        """frcnnモデルを準備する"""
        # classリストを読み出す
        with open(class_config_path) as json_file:
            self.class_info = json.load(json_file)["classes"]

        # デフォルトモデルをロード
        self.device = torch.device('cuda')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)

        # パラメータをロード
        if checkpoint_path:
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                in_features, len(self.class_info)
            )
            model_state_dict = torch.load(checkpoint_path)
            try:
                self.model.load_state_dict(model_state_dict)
            except RuntimeError:
                print("Failed to load checkpoint file without conversion. "
                      "Remap key name of dictionary data, and try to load the model again.",
                      file=sys.stderr)
                self.model.load_state_dict(self.remap_to_old_name(model_state_dict))

        # pytorchモデルをインファレンスモードに変更
        self.model.to(self.device)
        self.model.eval()

        # 画像入力のための変換を定義
        self.transform = transforms.Compose([transforms.ToTensor()])

    def predict(self, img_np, threshold=0.5):
        """
        Faster R-CNNで物体検出を実行する

        Note
        --------
        * 画像の入力形式はuint8/BRGとする。
        """
        img_tensor = self.transform(img_np).to(self.device)
        outputs = self.model([img_tensor])
        boxes = outputs[0]["boxes"].data.cpu().numpy()
        scores = outputs[0]["scores"].data.cpu().numpy()
        labels = outputs[0]["labels"].data.cpu().numpy()

        # threshold以上のBBoxを抽出
        boxes = boxes[scores >= threshold].astype(np.int32)
        labels = labels[scores >= threshold]
        scores = scores[scores >= threshold]

        result_array = []
        for box, label_id, score in zip(boxes, labels, scores):
            bbox_info = get_bbox_dict(self.class_info[label_id]["name"], score, box)
            result_array.append(bbox_info)

        return result_array

    @staticmethod
    def remap_to_old_name(pth):
        """命名がpytorch1.10から1.12の間で変化したため、強制的に名前を変更する"""
        remapping = [
            ("backbone.fpn.inner_blocks.0.0.weight", "backbone.fpn.inner_blocks.0.weight"),
            ("backbone.fpn.inner_blocks.0.0.bias", "backbone.fpn.inner_blocks.0.bias"),
            ("backbone.fpn.inner_blocks.1.0.weight", "backbone.fpn.inner_blocks.1.weight"),
            ("backbone.fpn.inner_blocks.1.0.bias", "backbone.fpn.inner_blocks.1.bias"),
            ("backbone.fpn.inner_blocks.2.0.weight", "backbone.fpn.inner_blocks.2.weight"),
            ("backbone.fpn.inner_blocks.2.0.bias", "backbone.fpn.inner_blocks.2.bias"),
            ("backbone.fpn.inner_blocks.3.0.weight", "backbone.fpn.inner_blocks.3.weight"),
            ("backbone.fpn.inner_blocks.3.0.bias", "backbone.fpn.inner_blocks.3.bias"),
            ("backbone.fpn.layer_blocks.0.0.weight", "backbone.fpn.layer_blocks.0.weight"),
            ("backbone.fpn.layer_blocks.0.0.bias", "backbone.fpn.layer_blocks.0.bias"),
            ("backbone.fpn.layer_blocks.1.0.weight", "backbone.fpn.layer_blocks.1.weight"),
            ("backbone.fpn.layer_blocks.1.0.bias", "backbone.fpn.layer_blocks.1.bias"),
            ("backbone.fpn.layer_blocks.2.0.weight", "backbone.fpn.layer_blocks.2.weight"),
            ("backbone.fpn.layer_blocks.2.0.bias", "backbone.fpn.layer_blocks.2.bias"),
            ("backbone.fpn.layer_blocks.3.0.weight", "backbone.fpn.layer_blocks.3.weight"),
            ("backbone.fpn.layer_blocks.3.0.bias", "backbone.fpn.layer_blocks.3.bias"),
            ("rpn.head.conv.0.0.weight", "rpn.head.conv.weight"),
            ("rpn.head.conv.0.0.bias", "rpn.head.conv.bias")]

        for names in remapping:
            pth[names[1]] = pth[names[0]]
            del pth[names[0]]

        return pth
