#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
from select import select
import traceback
from turtle import pos
import rospy
import rospkg
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math


class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    # 向きボーナスがない実験でも、これらの物体自体の収納点は得られるため除外しない。
    IGNORE_LIST = []
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    # Task 1では最大30個の物体を処理する。各探索位置で空になるまで再検出する。
    MAX_TASK1_OBJECTS = 30
    TROFAST_Y_OFFSET = 0.2
    MAX_GRASP_REACH = 0.78
    # 0.042以下の空掴みログと、保持候補0.089の実ログを分離する。
    GRIPPER_OBJECT_THRESHOLD = 0.05
    TASK2A_CLEARANCE = 0.75
    # Corridor width permits several lateral routes; the collision check
    # filters out lanes too close to detected floor objects.
    TASK2A_LANES = (1.35, 1.7, 2.15, 2.6, 2.95)
    TASK2A_WAYPOINT_Y = (2.5, 2.9, 3.3)
    TASK1_FOOD_LABELS = {
        "cracker_box", "sugar_box", "pudding_box", "gelatin_box",
        "potted_meat_can", "master_chef_can", "tuna_fish_can", "chips_can",
        "mustard_bottle", "tomato_soup_can", "banana", "strawberry", "apple",
        "lemon", "peach", "pear", "orange", "plum"
    }
    TASK1_KITCHEN_LABELS = {
        "windex_bottle", "bleach_cleanser", "sponge", "pitcher_base",
        "pitcher_lid", "plate", "bowl", "fork", "spoon", "spatula",
        "wine_glass", "mug"
    }
    TASK1_ORIENTATION_LABELS = {"large_marker", "small_marker", "fork", "spoon"}
    TASK1_TOOL_LABELS = {"padlock", "bolt_and_nut", "clamp"}
    TASK1_SHAPE_LABELS = {
        "credit_card_blank", "mini_soccer_ball", "softball", "baseball",
        "tennis_ball", "racquetball", "golf_ball", "marble", "cup",
        "foam_brick", "dice", "rope", "chain"
    }
    TASK1_TASK_LABELS = {
        "rubiks_cube", "colored_wood_block", "nine_hole_peg_test",
        "toy_airplane", "lego_duplo", "magazine", "black_t_shirt", "timer"
    }

    def __init__(self):
        # 変数の初期化
        self.instruction_list = []
        self.detection_list   = []
        self.task1_deposit_count = 0
        self.carrying_object = False
        self.safe_stop = False
        self.task2a_active = False
        self.task2a_navigating = False

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses       = self.load_json(self.get_path(["config", "poses.json"]))

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber("/message",    String, self.instruction_cb, queue_size=10)
        self.detection_sub   = rospy.Subscriber("/detect_msg", String, self.detection_cb,   queue_size=10)

    @staticmethod
    def get_path(pathes, package="wrs_algorithm"):
        """
        ROSパッケージ名とファイルまでのパスを指定して、ファイルのパスを取得する
        """
        if not pathes:  # check if the list is empty
            rospy.logerr("Can NOT resolve file path.")
            raise ValueError("You must specify the path to file.")
        pkg_path = rospkg.RosPack().get_path(package)
        path = os.path.join(*pathes)
        return os.path.join(pkg_path, path)

    @staticmethod
    def load_json(path):
        """
        jsonファイルを辞書型で読み込む
        """
        with open(path, "r") as json_file:
            return json.load(json_file)

    def instruction_cb(self, msg):
        """
        指示文を受信する
        """
        rospy.loginfo("instruction received. [%s]", msg.data)
        self.instruction_list.append(msg.data)

    def detection_cb(self, msg):
        """
        検出結果を受信する
        """
        if self.task2a_navigating and not self.safe_stop:
            self.safe_stop = True
            rospy.logerr("Collision event [%s]; canceling base goal and stopping.", msg.data)
            if omni_base.NAVCLIENT.get_state() == GoalStatus.ACTIVE:
                omni_base.NAVCLIENT.cancel_goal()
            omni_base.cmd_vel(0.0, 0.0, 0.0)
        else:
            rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(parent, child,rospy.Time.now(),rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name, abort_if=None):
        """
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            pos = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, pos[0], pos[1], pos[2])
            return omni_base.go_abs(pos[0], pos[1], pos[2], abort_if=abort_if)

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, pos, abort_if=None):
        """
        waypoint名で指定された場所に移動する
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", pos[0], pos[1], pos[2])
        return omni_base.go_abs(pos[0], pos[1], pos[2], abort_if=abort_if)

    def change_pose(self, name):
        """
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    def is_holding_object(self):
        """グリッパ開口から保持状態を更新する。"""
        opening = gripper.get_opening()
        self.carrying_object = opening >= self.GRIPPER_OBJECT_THRESHOLD
        rospy.logdebug("gripper opening %.3f; carrying=%s",
                       opening, self.carrying_object)
        return self.carrying_object

    def secure_grip_for_transport(self):
        """搬送姿勢でグリッパを閉じ直し、保持を確認してから走行する。"""
        if not self.is_holding_object():
            rospy.logerr("No object held before transport grip check.")
            return False
        if not gripper.command(0):
            rospy.logerr("Could not re-close gripper before transport.")
            return False
        rospy.sleep(0.5)
        if not self.is_holding_object():
            rospy.logerr("Object not retained after transport grip check.")
            return False
        return True

    def check_positions(self):
        """
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    def get_latest_detection(self):
        """
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(            SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        rospy.sleep(1.0)  # tfが安定するのを待つ
        return self.get_relative_coordinate("map", self.GRASP_TF_NAME).translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None: return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str  = ""
        for obj in obj_list:
            info_str = "{:<15}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h)
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += "- extracted: {:07.3f} ".format(score) + info_str

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(extracted, key=lambda x: x["score"], reverse=True):
            obj      = obj_info["bbox"]
            info_str = "{} ({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(obj.label, obj.score, obj.x, obj.y, obj.w, obj.h )
            rospy.loginfo("selected bbox: " + info_str)
            return obj_info

        # objが一つもない場合は、Noneを返す
        return None

    @classmethod
    def calc_score_bbox(cls, bbox):
        """
        detector_msgs/BBoxのスコアを計算する
        """
        gravity_x = bbox.x + bbox.w / 2
        gravity_y = bbox.y + bbox.h / 2
        xy_diff   = abs(320- gravity_x) / 320 + abs(360 - gravity_y) / 240

        return 1 / xy_diff

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """
        label名が一致するオブジェクトの中から最も把持すべき物体のbboxを返す
        """
        match_objs = [obj for obj in obj_list if obj.label in label]
        if not match_objs:
            rospy.logwarn("Cannot find a object which labeled with similar name.")
            return None
        return cls.get_most_graspable_bbox(match_objs)

    @staticmethod
    def extract_target_obj_and_person(instruction):
        """
        指示文（例: "potted_meat_can to person right"）から対象物と人を抽出する
        """
        rospy.loginfo("[extract_target_obj_and_person] instruction: %s", instruction)
        parts = instruction.strip().lower().split()
        if not parts:
            return None, None

        target_obj = parts[0]
        target_person = None
        for index, token in enumerate(parts):
            if token in ("left", "right"):
                target_person = token
                break
            if token in ("person_a", "person_b"):
                target_person = token
                break

        return target_obj, target_person

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in [ "+y", "-y", "+x", "-x" ]: raise RuntimeError("unnkown graps preliminary type [{}]".format(preliminary))

        rospy.loginfo("move hand to grasp (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        grasp_back_safe = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_back = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_pos = {"x": pos_x, "y": pos_y, "z": pos_z}

        if "+" in preliminary:
            sign = 1
        elif "-" in preliminary:
            sign = -1

        if "x" in preliminary:
            grasp_back_safe["x"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["x"] += sign * self.GRASP_BACK["xy"]
        elif "y" in preliminary:
            grasp_back_safe["y"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["y"] += sign * self.GRASP_BACK["xy"]

        if not gripper.command(1):
            rospy.logwarn("Could not open gripper before grasp.")
            return False
        if not whole_body.move_end_effector_pose(
                grasp_back_safe["x"], grasp_back_safe["y"],
                grasp_back_safe["z"], yaw, pitch, roll):
            rospy.logwarn("Could not reach safe pre-grasp pose.")
            return False
        if not whole_body.move_end_effector_pose(
                grasp_back["x"], grasp_back["y"], grasp_back["z"],
                yaw, pitch, roll):
            rospy.logwarn("Could not reach pre-grasp pose.")
            return False
        if not whole_body.move_end_effector_pose(
                grasp_pos["x"], grasp_pos["y"], grasp_pos["z"],
                yaw, pitch, roll):
            rospy.logwarn("Could not reach grasp pose; leaving gripper open.")
            return False
        if not gripper.command(0):
            rospy.logwarn("Gripper close command failed.")
            # 動作が一部だけ実行された可能性があるため、空掴みと決めつけない。
            self.safe_stop = True
            return False
        # 閉じた後に退避し、開口幅から保持状態を確認する。
        rospy.sleep(0.5)
        opening = gripper.get_opening()
        self.carrying_object = opening >= self.GRIPPER_OBJECT_THRESHOLD
        rospy.loginfo("gripper opening after grasp: %.3f (grasped=%s)",
                      opening, self.carrying_object)
        if not self.carrying_object:
            return False
        if not whole_body.move_end_effector_pose(
                grasp_back_safe["x"], grasp_back_safe["y"],
                grasp_back_safe["z"], yaw, pitch, roll):
            rospy.logerr("Holding an object but arm retreat failed; stopping task.")
            self.safe_stop = True
            return False
        opening_after_retreat = gripper.get_opening()
        self.carrying_object = opening_after_retreat >= self.GRIPPER_OBJECT_THRESHOLD
        if not self.carrying_object:
            rospy.logerr("Object was lost during arm retreat (opening %.3f); stopping.",
                         opening_after_retreat)
            self.safe_stop = True
            return False
        return True

    def grasp_from_front_side(self, grasp_pos):
        """
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        return self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos):
        """
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        return self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")

    def exec_graspable_method(self, grasp_pos, label=""):
        """
        task1専用:posの位置によって把持方法を判定し実行する。
        """
        method = None
        graspable_y = 1.85  # これ以上奥は把持できない
        desk_y = 1.5
        desk_z = 0.35

        # 把持禁止判定
        if graspable_y < grasp_pos.y and desk_z > grasp_pos.z:
            return False

        if label in ["cup", "frisbee", "bowl"]:
            # bowlの張り付き対策
            method = self.grasp_from_upper_side
        else:
            if desk_y < grasp_pos.y and desk_z > grasp_pos.z:
                # 机の下である場合
                method = self.grasp_from_front_side
            else:
                method = self.grasp_from_upper_side

        return bool(method(grasp_pos))

    def approach_and_refresh_target(self, target_obj, target_pos, search_pose):
        """対象物が腕の届く距離にない場合、近づいてから同じラベルを再認識する。"""
        base = self.get_relative_coordinate("map", "base_footprint")
        if base is None:
            return target_pos

        dx = target_pos.x - base.translation.x
        dy = target_pos.y - base.translation.y
        distance = math.hypot(dx, dy)
        if distance <= self.MAX_GRASP_REACH:
            return target_pos

        stand_off = 0.62
        goal_x = target_pos.x - stand_off * dx / distance
        goal_y = target_pos.y - stand_off * dy / distance
        goal_yaw = math.degrees(math.atan2(dy, dx))
        rospy.loginfo("approach [%s]: distance=%.2f, goal=(%.2f, %.2f, %.1f)",
                      target_obj, distance, goal_x, goal_y, goal_yaw)
        if not self.goto_pos([goal_x, goal_y, goal_yaw]):
            return None

        self.change_pose(search_pose)
        detected = self.get_latest_detection()
        matches = [bbox for bbox in detected.bboxes if bbox.label == target_obj]
        if not matches:
            rospy.logwarn("Target [%s] not visible after approach.", target_obj)
            return None

        # 同一ラベルが複数ある場合は、画像上で最も把持しやすいものを選ぶ。
        bbox = self.get_most_graspable_bbox(matches)
        return self.get_grasp_coordinate(bbox)

    def put_in_place(self, place, into_pose):
        """保持したまま投入地点へ移動し、到達確認後にだけ手を開く。"""
        if not self.secure_grip_for_transport():
            rospy.logwarn("Grip check failed; refusing to drive to a deposit.")
            self.safe_stop = True
            return False

        # 目標未到達時にその場で開放すると、物体を散らかすため必ず成否を確認する。
        if not self.goto_name(place, abort_if=self.is_holding_object):
            if not self.is_holding_object():
                rospy.logerr("Object was lost en route to [%s]; base goal was canceled.", place)
                self.safe_stop = True
                return False
            rospy.logwarn("Could not reach deposit [%s]; keeping object grasped.", place)
            return False
        if not self.is_holding_object():
            rospy.logerr("Object was lost while moving to [%s]; stopping task.", place)
            self.safe_stop = True
            return False
        if not self.change_pose(into_pose):
            rospy.logerr("Could not reach release pose at [%s]; keeping object grasped.", place)
            return False
        if not self.is_holding_object():
            rospy.logerr("Object was lost before release at [%s]; stopping task.", place)
            self.safe_stop = True
            return False
        if not gripper.command(1):
            rospy.logerr("Could not release object at [%s].", place)
            return False
        self.carrying_object = False
        rospy.sleep(1.0)
        return True

    def get_task1_deposit(self, label):
        """既存実装で使われていたBin A/Bと投入姿勢を交互に使う。

        独自に追加したトレイ・容器・引き出し座標は実測校正がなく、
        採点器の中心座標判定から外れる可能性があるため使わない。
        """
        place = "bin_a_place" if self.task1_deposit_count % 2 == 0 else "bin_b_place"
        self.task1_deposit_count += 1
        return place, "put_in_bin"

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        # trofastの引き出しを引き出す
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        a = True  # TODO 不要な変数
        gripper.command(1)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw,  pitch, roll)
        gripper.command(1)
        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y+=self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose( pos_x, pos_y +    self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(  pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(            pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    def deliver_to_target(self, target_obj, target_person):
        """
        棚で取得したものを人に渡す。
        """
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")

        rospy.loginfo("target_obj: " + target_obj + "  target_person: " + target_person)
        # 物体検出結果から、把持するbboxを決定
        detected_objs = self.get_latest_detection()
        grasp_bbox = self.get_most_graspable_bboxes_by_label(detected_objs.bboxes, target_obj)
        if grasp_bbox is None:
            rospy.logwarn("Cannot find object to grasp. task2b is aborted.")
            return

        # BBoxの3次元座標を取得して、その座標で把持する
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)
        if grasp_pos is None or not self.change_pose("grasp_on_shelf"):
            rospy.logwarn("Cannot safely start shelf grasp; aborting delivery.")
            return
        if not self.grasp_from_front_side(grasp_pos):
            if self.carrying_object or self.safe_stop:
                self.safe_stop = True
                rospy.logerr("Shelf grasp state is uncertain; stopping without moving base.")
                return
            rospy.logwarn("Failed to grasp requested food [%s]; aborting Task 2b delivery.", target_obj)
            return
        if not self.change_pose("all_neutral"):
            rospy.logerr("Could not enter carry pose after shelf grasp; stopping.")
            self.safe_stop = True
            return
        if not self.secure_grip_for_transport():
            rospy.logerr("Shelf object is not secure for transport; stopping before base motion.")
            self.safe_stop = True
            return

        # target_personの前に持っていく
        if target_person in ("left", "person_a"):
            person_waypoint = "person_a"
        else:
            person_waypoint = "person_b"
        if not self.goto_name(person_waypoint, abort_if=self.is_holding_object):
            if not self.is_holding_object():
                rospy.logerr("Object was lost en route to [%s]; base goal was canceled.",
                             person_waypoint)
                self.safe_stop = True
                return
            rospy.logwarn("Could not reach [%s]; keeping object grasped.", person_waypoint)
            return
        if not self.is_holding_object():
            rospy.logerr("Object was lost while moving to person; stopping.")
            self.safe_stop = True
            return
        if not self.change_pose("deliver_to_human"):
            rospy.logwarn("Could not reach handover pose; keeping object grasped.")
            return
        rospy.sleep(10.0)
        if not self.is_holding_object():
            rospy.logerr("Object was lost before handover; stopping.")
            self.safe_stop = True
            return
        if not gripper.command(1):
            rospy.logerr("Could not release requested object to person.")
            return
        self.carrying_object = False
        self.change_pose("all_neutral")

    def execute_avoid_blocks(self):
        """検出物体を迂回し、各区間を再確認しながらゴールへ進む。"""
        obstacles = self.get_task2a_obstacles()
        base = self.get_relative_coordinate("map", "base_footprint")
        if base is None:
            rospy.logerr("Cannot localize before Task 2a route planning.")
            return False
        current = (base.translation.x, base.translation.y)
        goal = self.coordinates["positions"]["go_throw_2a"]

        # Task 2aの物体群は通路中央に置かれる。障害物検出が空でも
        # 中央を直進せず、列の手前から側面を回る迂回路を使う。
        candidate_routes = [
            [(1.25, 1.50, 90.0), (1.25, 3.30, 90.0), goal],
            [(3.10, 1.50, 90.0), (3.10, 3.30, 90.0), goal],
        ]
        scored_routes = []
        for candidate in candidate_routes:
            start = current
            clearances = []
            distance = 0.0
            for waypoint in candidate:
                end = (waypoint[0], waypoint[1])
                clearances.append(self.segment_clearance(start, end, obstacles))
                distance += math.hypot(end[0] - start[0], end[1] - start[1])
                start = end
            minimum = min(clearances)
            if minimum >= self.TASK2A_CLEARANCE:
                scored_routes.append((-minimum, distance, candidate))

        if not scored_routes:
            rospy.logerr("No detour clears floor objects by %.2fm; not moving.",
                         self.TASK2A_CLEARANCE)
            return False
        _, _, route = min(scored_routes)
        rospy.loginfo("Task 2a detour selected: %s", route)

        known_obstacles = list(obstacles)
        for waypoint in route:
            # その場で新たに見えた物体も地図へ足し、消えた検出を
            # 「障害物なし」と誤解しないよう、既知物体は保持する。
            for obstacle in self.get_task2a_obstacles():
                if all(math.hypot(obstacle[0] - known[0], obstacle[1] - known[1]) > 0.20
                       for known in known_obstacles):
                    known_obstacles.append(obstacle)

            base = self.get_relative_coordinate("map", "base_footprint")
            if base is None:
                rospy.logerr("Cannot localize while following Task 2a detour.")
                return False
            start = (base.translation.x, base.translation.y)
            end = (waypoint[0], waypoint[1])
            clearance = self.segment_clearance(start, end, known_obstacles)
            if clearance < self.TASK2A_CLEARANCE:
                rospy.logerr("Detour segment blocked (clearance %.2fm); stopping.", clearance)
                return False
            self.task2a_navigating = True
            moved = self.goto_pos(waypoint)
            self.task2a_navigating = False
            if not moved:
                rospy.logerr("Task 2a detour waypoint failed; stopping route.")
                return False
            if self.safe_stop:
                return False
        return True

    def get_task2a_obstacles(self):
        """床上の小物体だけをmap座標へ変換する。"""
        detected = self.get_latest_detection()
        obstacles = []
        for bbox in detected.bboxes:
            pos = self.get_grasp_coordinate(bbox)
            if pos is None:
                continue
            if (pos.z <= 0.45 and 1.2 <= pos.x <= 3.2 and
                    2.0 <= pos.y <= 3.8):
                obstacles.append((pos.x, pos.y))
        rospy.loginfo("Task 2a floor obstacles mapped: %s", obstacles)
        return obstacles

    @staticmethod
    def segment_clearance(start, end, obstacles):
        """線分と障害物中心の最小距離。"""
        if not obstacles:
            return float("inf")
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        length_sq = dx * dx + dy * dy
        clearance = float("inf")
        for ox, oy in obstacles:
            if length_sq == 0.0:
                t = 0.0
            else:
                t = max(0.0, min(1.0,
                    ((ox - sx) * dx + (oy - sy) * dy) / length_sq))
            px, py = sx + t * dx, sy + t * dy
            clearance = min(clearance, math.hypot(ox - px, oy - py))
        return clearance

    def select_next_waypoint(self, current_stp, pos_bboxes):
        """障害物との距離を評価し、現在位置から安全なレーンを選択する。"""
        base = self.get_relative_coordinate("map", "base_footprint")
        if base is None:
            rospy.logerr("Cannot localize robot for Task 2a route planning.")
            return None

        start = (base.translation.x, base.translation.y)
        y = self.TASK2A_WAYPOINT_Y[current_stp]
        goal = self.coordinates["positions"]["go_throw_2a"]
        candidates = []
        for x in self.TASK2A_LANES:
            waypoint = (x, y)
            clearance = min(
                self.segment_clearance(start, waypoint, pos_bboxes),
                self.segment_clearance(waypoint, (goal[0], goal[1]), pos_bboxes))
            if clearance >= self.TASK2A_CLEARANCE:
                travel = math.hypot(x - start[0], y - start[1])
                candidates.append((travel, -clearance, x))

        if not candidates:
            rospy.logwarn("All Task 2a lanes blocked at y=%.2f; no movement commanded.", y)
            return None

        _, neg_clearance, x = min(candidates)
        rospy.loginfo("Task 2a selected lane x=%.2f (clearance %.2fm)", x, -neg_clearance)
        return [x, y, 90.0]

    def execute_task1(self):
        """
        task1を実行する
        """
        rospy.loginfo("#### start Task 1 ####")
        hsr_position = [
            ("tall_table", "look_at_tall_table"),
            ("near_long_table_l", "look_at_near_floor"),
            ("long_table_r", "look_at_long_table"),
        ]

        total_cnt = 0
        for plc, pose in hsr_position:
            failed_labels = set()
            while total_cnt < self.MAX_TASK1_OBJECTS:
                # Asia1st同様、収納後に認識場所へ戻って次の物体を再検出する。
                if not self.goto_name(plc):
                    rospy.logwarn("Could not reach search waypoint [%s]; skipping it.", plc)
                    break
                if not self.change_pose(pose):
                    rospy.logwarn("Could not set search pose [%s]; skipping it.", pose)
                    break
                # 把持前にグリッパを開く。
                gripper.command(0)

                # 把持対象の有無チェック
                detected_objs = self.get_latest_detection()
                candidates = [bbox for bbox in detected_objs.bboxes
                              if bbox.label not in failed_labels]
                graspable_obj = self.get_most_graspable_obj(candidates)

                if graspable_obj is None:
                    rospy.loginfo("No more untried objects at [%s].", plc)
                    break
                label = graspable_obj["label"]
                grasp_bbox = graspable_obj["bbox"]
                # TODO ラベル名を確認するためにコメントアウトを外す
                # rospy.loginfo("grasp the " + label)

                # Asia1st同様、対象物の近くまで移動してから把持を行う。
                grasp_pos = self.get_grasp_coordinate(grasp_bbox)
                if grasp_pos is None:
                    rospy.logwarn("Could not estimate grasp position for [%s].", label)
                    failed_labels.add(label)
                    continue
                grasp_pos = self.approach_and_refresh_target(label, grasp_pos, pose)
                if grasp_pos is None:
                    rospy.logwarn("Could not approach target [%s].", label)
                    failed_labels.add(label)
                    continue

                if not self.change_pose("grasp_on_table"):
                    rospy.logwarn("Could not reach grasp posture for [%s].", label)
                    failed_labels.add(label)
                    continue
                grasped = self.exec_graspable_method(grasp_pos, label)
                if not grasped:
                    if self.carrying_object or self.safe_stop:
                        rospy.logerr("Arm/gripper state is uncertain after grasp; stopping safely.")
                        self.safe_stop = True
                        return
                    rospy.logwarn("Object [%s] is not graspable from [%s]; moving to next search position.", label, plc)
                    failed_labels.add(label)
                    continue

                if not self.change_pose("all_neutral"):
                    rospy.logerr("Could not enter carry pose; keeping object grasped and stopping.")
                    self.safe_stop = True
                    return

                deposit, deposit_pose = self.get_task1_deposit(label)
                rospy.loginfo("Task 1 deposit for [%s]: [%s]", label, deposit)
                if not self.put_in_place(deposit, deposit_pose):
                    if not self.safe_stop and self.is_holding_object():
                        alternate = "bin_b_place" if deposit == "bin_a_place" else "bin_a_place"
                        rospy.logwarn("Retrying delivery at alternate deposit [%s].", alternate)
                        if not self.put_in_place(alternate, deposit_pose):
                            self.safe_stop = True
                    if self.safe_stop or self.is_holding_object():
                        rospy.logerr("Could not deliver object safely; stopping remaining tasks.")
                        self.safe_stop = True
                        return
                    # 把持が失われた場合も、物体に接触する次動作を続けない。
                    rospy.logerr("Object was lost during delivery; stopping remaining tasks.")
                    self.safe_stop = True
                    return
                total_cnt += 1

        rospy.loginfo("Task 1 finished after handling %d objects.", total_cnt)

    def execute_task2a(self):
        """
        task2aを実行する
        """
        rospy.loginfo("#### start Task 2a ####")
        self.task2a_active = True
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.change_pose("look_at_near_floor")
        if self.safe_stop:
            self.task2a_active = False
            rospy.logerr("Collision reported before Task 2a navigation; stopping without moving.")
            return False
        # Task 1終了位置から障害物を回避する。中央のstandby座標へ
        # 先に向かうと、障害物地図を作る前に物体列へ入るため経由しない。
        if not self.execute_avoid_blocks():
            rospy.logerr("Task 2a route is not verified safe; aborting subsequent tasks.")
            self.safe_stop = True
            self.task2a_active = False
            return False

        if self.safe_stop:
            self.task2a_active = False
            return False
        whole_body.move_to_go()
        self.task2a_active = False
        return True

    def execute_task2b(self):
        """
        task2bを実行する
        """
        rospy.loginfo("#### start Task 2b ####")

        # 命令文を取得
        if self.instruction_list:
            latest_instruction = self.instruction_list[-1]
            rospy.loginfo("recieved instruction: %s", latest_instruction)
        else:
            rospy.logwarn("instruction_list is None")
            return

        # 命令内容を解釈
        target_obj, target_person = self.extract_target_obj_and_person(latest_instruction)

        # 指定したオブジェクトを指定した配達先へ
        if target_obj and target_person:
            self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")
        # シミュレータのリセット後も競技開始位置から始める。
        if not self.goto_name("home"):
            self.safe_stop = True
            return
        self.execute_task1()
        if self.safe_stop:
            rospy.logerr("Safe stop after Task 1; skipping remaining tasks.")
            return
        if not self.execute_task2a() or self.safe_stop:
            rospy.logerr("Safe stop after Task 2a; skipping Task 2b.")
            return
        self.execute_task2b()


def main():
    """
    WRS環境内でタスクを実行するためのメインノードを起動する
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
