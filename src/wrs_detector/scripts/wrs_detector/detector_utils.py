"""
物体検出関連のユーティリティモジュール
"""


def get_bbox_dict(label, score, bbox):
    """
    bbox情報をディクショナリ形式に変換する
    """
    ret = {
        "label": label,
        "score": score,
        "bbox": {
            "x": bbox[0], "y": bbox[1],
            "w": abs(bbox[0]-bbox[2]), "h": abs(bbox[1]-bbox[3])
        }
    }

    return ret
