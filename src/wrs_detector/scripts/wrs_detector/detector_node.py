#!/usr/bin/env python
"""
物体検出を行う基本モジュール
"""

import os
import importlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rospy
import rospkg
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import Image as MsgImage
from detector_msgs.msg import BBox, BBoxArray
from detector_msgs.srv import GetObjectDetection, GetObjectDetectionResponse


class Detector:
    """
    機械学習パッケージを使用して物体検出を行うノードの実装クラス
    """
    DEFAULT_DETECTION_FREQ = 1
    DEFAULT_DETECTOR = "frcnn"
    DEFAULT_THRESHOLD = 0.1
    MODE_CONTINUOUS = "continuous"
    MODE_REQUIRE_REQUEST = "with_request"

    def __init__(self):
        self.previous_detection = rospy.Time.now()
        self.image_req_time = rospy.Time.now()
        self.current_msg = None

        # 動作モードの取得
        self.mode = rospy.get_param("~mode")
        if self.mode not in [self.MODE_CONTINUOUS, self.MODE_REQUIRE_REQUEST]:
            err_msg = "unknown type of Mode {}".format(self.mode)
            raise RuntimeError(err_msg)

        # 検出器のタイプを取得
        detector_str = rospy.get_param("~detector", default=Detector.DEFAULT_DETECTOR)
        if detector_str == "frcnn_default":
            coco_class_file_path = os.path.join(
                rospkg.RosPack().get_path("wrs_detector"), "config", "coco_classes.json")
            detector_module = importlib.import_module("wrs_detector.frcnn_detector")
            self.detector = detector_module.FasterRcnnDetector(coco_class_file_path)
        elif detector_str == "frcnn_finetuned":
            detector_module = importlib.import_module("wrs_detector.frcnn_detector")
            self.detector = detector_module.FasterRcnnDetector(
                rospy.get_param("~frcnn_finetuned/class_config_path"),
                rospy.get_param("~frcnn_finetuned/checkpoint_path"))
        elif detector_str == "mmdet":
            detector_module = importlib.import_module("wrs_detector.mmdet_detector")
            rospy.get_param("~mode")

            self.detector = detector_module.MmdetDetector(
                rospy.get_param("~mmdet/config"), rospy.get_param("~mmdet/checkpoint"))
        else:
            err_msg = "unknown type of Detector {}".format(detector_str)
            raise RuntimeError(err_msg)

        # continuousモードの際は、最大検出周波数を取得
        if self.mode == self.MODE_CONTINUOUS:
            self.detection_freq = float(rospy.get_param("~freq", self.DEFAULT_DETECTION_FREQ))

        # その他のパラメータを取得
        self.threshold = float(rospy.get_param("~threshold", self.DEFAULT_THRESHOLD))
        self.publish_image = bool(rospy.get_param("~publish_image", False))

        # ROS関連通信の初期化
        self.result_pub = rospy.Publisher("result", BBoxArray, queue_size=1)
        if self.publish_image:
            self.result_image_pub = rospy.Publisher("result_image", MsgImage, queue_size=1)
        if self.mode == self.MODE_REQUIRE_REQUEST:
            self.detection_request_srv = rospy.Service(
                "get_object_detection", GetObjectDetection, self.detection_request)
        self.image_sub = rospy.Subscriber("in", numpy_msg(MsgImage), self.image_cb, queue_size=1)

    def image_cb(self, msg):
        """
        認識対象画像を受信するコールバック関数
        """
        if self.mode == self.MODE_REQUIRE_REQUEST:
            if self.current_msg is None and msg.header.stamp > self.image_req_time:
                self.current_msg = msg

        elif self.mode == self.MODE_CONTINUOUS:
            if rospy.Time.now() - self.previous_detection < rospy.Duration(secs=1/self.detection_freq):
                return
            self.previous_detection = rospy.Time.now()
            self.detection(msg)

    def detection_request(self, _):
        """
        認識処理のリクエストを処理するコールバック
        """
        self.wait_latest_image()
        success, bboxes, _ = self.detection(self.current_msg)

        res = GetObjectDetectionResponse()
        if success:
            res.bboxes = bboxes
        res.successed = success

        return res

    def wait_latest_image(self):
        """
        最新の画像が到着するのを待つ
        """
        self.image_req_time = rospy.Time.now()
        self.current_msg = None
        while not rospy.is_shutdown():
            if self.current_msg is not None:
                break
            rospy.sleep(0.1)

    @staticmethod
    def create_result_image(img_np, bboxes):
        """
        バウンディングボックスの焼き込み画像を生成する
        """
        img_flip_rb = img_np[:, :, [2, 1, 0]]
        image_pil = Image.fromarray(img_flip_rb.astype(np.uint8))

        draw = ImageDraw.Draw(image_pil)
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        for bbox in sorted(bboxes, key=lambda x: x["score"]):
            label_str = "{},{:.2%}".format(bbox["label"], bbox["score"])
            box = bbox["bbox"]
            draw.rectangle(
                [(box["x"], box["y"]), (box["x"] + box["w"], box["y"] + box["h"])],
                outline="red", width=3)
            text_w, text_h = fnt.getsize(label_str)
            draw.rectangle([box["x"], box["y"]-text_h, box["x"]+text_w, box["y"]], fill="red")
            draw.text((box["x"], box["y"]-text_h), label_str, font=fnt, fill="white")

        return np.array(image_pil)[:, :, ::-1]

    def detection(self, msg):
        """
        物体認識を行って、ROSのメッセージ形式で返す
        """
        # 受信メッセージ型の確認
        if msg.encoding != "bgr8":
            rospy.logwarn("%s node recieved unsupported type image. "
                          "Type must be bgr8.", rospy.get_name())
            return False, None, None

        # 物体検出の実行
        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        bboxes = self.detector.predict(img_np, threshold=self.threshold)

        # bbox情報の生成
        bboxes_msg = BBoxArray(header=msg.header)
        for bbox in bboxes:
            box_msg = BBox(label=bbox["label"], score=bbox["score"])
            box_msg.x = bbox["bbox"]["x"]
            box_msg.y = bbox["bbox"]["y"]
            box_msg.w = bbox["bbox"]["w"]
            box_msg.h = bbox["bbox"]["h"]
            bboxes_msg.bboxes.append(box_msg)

        # bbox焼き込み画像の生成
        result_image = self.create_result_image(img_np, bboxes)
        contig_arr = np.ascontiguousarray(result_image.astype(np.uint8))
        bbox_img = MsgImage(header=msg.header)
        bbox_img.height = msg.height
        bbox_img.width = msg.width
        bbox_img.encoding = "bgr8"
        bbox_img.is_bigendian = 0
        bbox_img.step = contig_arr.strides[0]
        bbox_img.data = contig_arr.tostring()

        self.result_pub.publish(bboxes_msg)
        if self.publish_image:
            self.result_image_pub.publish(bbox_img)

        return True, bboxes_msg, bbox_img


if __name__ == "__main__":
    try:
        rospy.init_node("detector")

        detector = Detector()
        rospy.loginfo("initializing node[%s] is completed", rospy.get_name())

        rospy.spin()
    except rospy.ROSInterruptException:
        pass
