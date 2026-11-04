#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PCDとバウンディングボックス情報からTFを出力する
"""

# Import libraries
import traceback
import numpy as np

# Import ROS packages
import rospy
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from detector_msgs.srv import SetTransformFromBBox, SetTransformFromBBoxResponse
from detector_msgs.srv import AddTransformToManager, AddTransformToManagerRequest


class PcdToTf():
    """
    PCDとバウンディングボックス情報からTFを出力するクラス
    """
    AVG_NUM_OF_PCD = 5              # 平均する際のPCDフレーム数
    RATE_OF_ERODE = 0.5             # バウンディングボックスの収縮割合
    HEAD_TO_TF_PREFIX = "head_to"   # 計算用に出力するTFに設定するprefix
    PC2_HEADER = Header(frame_id='head_rgbd_sensor_rgb_frame')
    PC2_FIELDS = [
        # 点の座標(x, y, z)
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    def __init__(self):
        self._req_time = rospy.Time()
        self._latest_pcd = []
        self._tf_list = []

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._tf_br = tf2_ros.TransformBroadcaster()

        self._object_pcd_pub = rospy.Publisher('object_point_cloud', PointCloud2, queue_size=1)

        pcd_topic_name = "/hsrb/head_rgbd_sensor/depth_registered/rectified_points"
        self._pcd_sub = rospy.Subscriber(
            pcd_topic_name, PointCloud2, callback=self.pcd_cb)

        add_tf_to_manager_name = "add_tf_to_manager"
        rospy.wait_for_service(add_tf_to_manager_name)
        self._add_tf_to_manager_clt = rospy.ServiceProxy(
            add_tf_to_manager_name, AddTransformToManager)

        self._set_tf_from_bbox_srv = rospy.Service(
            "set_tf_from_bbox", SetTransformFromBBox, self.set_tf_from_bbox_cb)

    def set_tf_from_bbox_cb(self, req):
        """
        point_cloudからTFを出力するROSサービスのコールバック
        @param[in] req ROSサービスのリクエスト
        @return TFが出力できた場合はTrueを返す
        """
        rospy.loginfo("tf requested: %s", req.frame)

        # 新規にデータを取り込む
        self._req_time = rospy.Time.now()
        self._latest_pcd = []
        while len(self._latest_pcd) < self.AVG_NUM_OF_PCD:
            rospy.sleep(0.1)

        # bboxを縮小する
        bbox_x = req.bbox.x + req.bbox.w * (1 - self.RATE_OF_ERODE)/2
        bbox_y = req.bbox.y + req.bbox.h * (1 - self.RATE_OF_ERODE)/2
        bbox_w = req.bbox.w * self.RATE_OF_ERODE
        bbox_h = req.bbox.h * self.RATE_OF_ERODE
        rospy.loginfo(
            "check pointcloud which fit for bbox(x,y,w,h = %.2f, %.2f, %.2f, %.2f)",
            bbox_x, bbox_y, bbox_w, bbox_h)

        # uv座標のリストを作成
        uv_list = []
        for point_u in range(int(bbox_x), int(bbox_x + bbox_w)):
            for point_v in range(int(bbox_y), int(bbox_y + bbox_h)):
                uv_list.append((point_u, point_v))

        # bbox内のpointを抽出して、numpyで計算
        pcd_list = []
        for pcd in self._latest_pcd:
            pcd_list.extend(pc2.read_points(
                pcd, skip_nans=True, field_names=("x", "y", "z"), uvs=uv_list))
        if len(pcd_list) <= 0:
            rospy.logerr("Cannot get any pointcloud which fits for specified bbox.")
            return SetTransformFromBBoxResponse(result=False)
        np_point_list = np.array(pcd_list)
        avg_point = np.mean(np_point_list, axis=0)
        rospy.logdebug("tf between head_rgbd_camera and object(x,y,z): %.3f,%.3f,%.3f",
                       avg_point[0], avg_point[1], avg_point[2])

        # 計算に使用したpoint_cloudをpublish
        point_cloud = pc2.create_cloud(self.PC2_HEADER, self.PC2_FIELDS, pcd_list)
        self._object_pcd_pub.publish(point_cloud)

        # 座標系を変換
        tf_time = rospy.Time.now()
        source_frame = self._latest_pcd[0].header.frame_id
        tf_head_to_obj = TransformStamped()
        tf_head_to_obj.header.stamp = tf_time
        tf_head_to_obj.header.frame_id = source_frame
        tf_head_to_obj.child_frame_id = self.HEAD_TO_TF_PREFIX + req.frame
        tf_head_to_obj.transform.translation.x = avg_point[0]
        tf_head_to_obj.transform.translation.y = avg_point[1]
        tf_head_to_obj.transform.translation.z = avg_point[2]
        tf_head_to_obj.transform.rotation.x = 0.0
        tf_head_to_obj.transform.rotation.y = 0.0
        tf_head_to_obj.transform.rotation.z = 0.0
        tf_head_to_obj.transform.rotation.w = 1.0
        try:
            # 一度headカメラからのtfで出力して、mapからの絶対tfを取得する
            self._tf_br.sendTransform(tf_head_to_obj)
            tf_map_to_obj = self._tf_buffer.lookup_transform(
                'map', self.HEAD_TO_TF_PREFIX + req.frame,
                tf_time, rospy.Duration(0.5))

            # grasp用の回転を取得
            tf_to_grasp_standard = self._tf_buffer.lookup_transform(
                'map', "grasp_standard", tf_time, rospy.Duration(0.5))
            tf_map_to_obj.transform.rotation = tf_to_grasp_standard.transform.rotation

            # ブロードキャストするtfに追加
            tf_map_to_obj.child_frame_id = req.frame
            tf_request = AddTransformToManagerRequest(transform=tf_map_to_obj)
            self._add_tf_to_manager_clt(tf_request)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr(traceback.format_exc())
            return SetTransformFromBBoxResponse(result=False)

        return SetTransformFromBBoxResponse(result=True)

    def pcd_cb(self, msg):
        """
        point_cloudを受信するコールバック
        @param[in] msg point_cloudデータ
        @return None
        """
        # 設定されたデータ数に達するまで保存する
        if len(self._latest_pcd) < self.AVG_NUM_OF_PCD:
            if msg.header.stamp > self._req_time:
                self._latest_pcd.append(msg)


def main():
    """
    PCDとバウンディングボックス情報からTFを出力するノードを起動する
    """
    rospy.init_node("pcd_to_tf_node")

    try:
        unused_pcd_to_tf = PcdToTf()
        rospy.loginfo("initializing node[%s] is completed", rospy.get_name())

        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
