# -*- coding: utf-8 -*-
"""
HSRのハンドを制御するためのユーティリティモジュール
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import moveit_commander
import rospy


# moveitでの制御対象としてハンドを指定
GRIPPER_CMD = moveit_commander.MoveGroupCommander(str("gripper"))


def command(value):
    """
    ハンドを制御

    Parameters
    ----------
        value (float): ハンドの開き具合 (0：閉じる、1:開く)

    Return
    ------
        正しく動作すればTrue, そうでなければFalse

    """

    GRIPPER_CMD.set_joint_value_target(str("hand_motor_joint"), value)
    success = GRIPPER_CMD.go()
    rospy.sleep(6)
    return success


def get_opening():
    """現在のhand_motor_joint値を返す。把持中は対象物で閉じ切れない。"""
    joint_values = GRIPPER_CMD.get_current_joint_values()
    return joint_values[0] if joint_values else 0.0
