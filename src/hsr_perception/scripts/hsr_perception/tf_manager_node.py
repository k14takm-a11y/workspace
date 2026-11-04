#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一定数のTFを出力するノード
"""

import traceback

import rospy
import tf2_ros
from detector_msgs.srv import AddTransformToManager, AddTransformToManagerResponse


class TfManager():
    """
    一定数のTFを出力するノード
    """
    DEFAULT_MAX_NUM_OF_TF = 10  # 出力するTFの最大数
    DEFAULT_TF_PUBLISH_FREQ = 10  # TFを出力する周波数

    def __init__(self):
        self._tf_list = []

        self.max_num_of_tf = int(
            rospy.get_param("~max_num_of_tf", self.DEFAULT_MAX_NUM_OF_TF))
        self.tf_publish_freq = float(
            rospy.get_param("~tf_publish_freq", self.DEFAULT_TF_PUBLISH_FREQ))

        self._tf_br = tf2_ros.TransformBroadcaster()
        self._add_tf_to_manager_srv = rospy.Service(
            "add_tf_to_manager", AddTransformToManager, self.add_tf_to_manager_cb)

        # display settings
        setting_str = "node setting [{}]".format(rospy.get_name())
        setting_str += "\n - Max number of TF    : {}".format(self.max_num_of_tf)
        setting_str += "\n - Frequency to publish: {:.1f}".format(self.tf_publish_freq)
        rospy.loginfo(setting_str)

    def add_tf_to_manager_cb(self, req):
        """
        出力するTFを追加するサービスのコールバック

        @param[in] req ROSサービスのリクエスト
        @return TFが出力できた場合はTrueを返す
        """
        try:
            # 重複しているtfがあれば削除
            for idx, trans in enumerate(self._tf_list):
                if trans.child_frame_id == req.transform.child_frame_id:
                    self._tf_list.pop(idx)
                    break

            # ブロードキャストするtfに追加
            self._tf_list.append(req.transform)

            # 保持するtfが最大値を超えたら、古いものから削除
            if len(self._tf_list) > self.max_num_of_tf:
                self._tf_list.pop(0)

            rospy.loginfo("%d tf(s) are broadcasting" % len(self._tf_list))

        except KeyError:
            rospy.logerr(traceback.format_exc())
            return AddTransformToManagerResponse(result=False)

        return AddTransformToManagerResponse(result=True)

    def run(self):
        """
        一定数のTFを出力する処理を実行する

        @return None
        """
        rate = rospy.Rate(self.tf_publish_freq)
        while not rospy.is_shutdown():
            for trans in self._tf_list:
                trans.header.stamp = rospy.Time.now()
                try:
                    self._tf_br.sendTransform(trans)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException, KeyError):
                    rospy.logerr(traceback.format_exc())
            rate.sleep()


def main():
    """
    一定数のTFを出力するROSノードを実行する
    """
    rospy.init_node("tf_manager")
    try:
        node = TfManager()
        rospy.loginfo("initializing node[%s] is completed", rospy.get_name())

        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
