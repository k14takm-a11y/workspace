"""
mmdetectionで物体検出を行うモジュール
"""

import numpy as np
import torch
from mmdet.apis import init_detector, inference_detector
from wrs_detector.detector_utils import get_bbox_dict


class MmdetDetector():
    """
    Faster-RCNNで物体検出を行うクラス
    """
    def __init__(self, config_path, checkpoint_path):
        self.device = torch.device('cuda')
        self.model = init_detector(config_path, checkpoint_path, device=self.device)

    def predict(self, img_np, threshold=0.5):
        """
        mmdetectionで物体検出を実行する

        Note
        --------
        * 画像の入力形式はuint8/BRGとする。
        """
        result = inference_detector(self.model, img_np)

        # mmdetectionからの実行結果をパースして、bbox情報を抽出
        if isinstance(result, tuple):
            bbox_result, _ = result
        else:
            bbox_result = result
        bbox_score = np.vstack(bbox_result)
        boxes = bbox_score[:, 0:4]
        scores = bbox_score[:, 4]

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        # 閾値以下のbboxは削除
        boxes = boxes[scores >= threshold].astype(np.int32)
        labels = labels[scores >= threshold]
        scores = scores[scores >= threshold]

        # json形式に変換
        if hasattr(self.model, "module"):
            class_idx = self.model.module.CLASSES
        else:
            class_idx = self.model.CLASSES
        result_array = []
        for box, label_id, score in zip(boxes, labels, scores):
            bbox_info = get_bbox_dict(class_idx[label_id], score, box)
            result_array.append(bbox_info)

        return result_array
