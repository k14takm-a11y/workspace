#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import copy
import heapq
import json
import math
import os
import re
import traceback
import numpy as np
import actionlib
import rospy
import rospkg
import tf2_ros
from control_msgs.msg import (
    FollowJointTrajectoryAction, FollowJointTrajectoryGoal, JointTrajectoryControllerState)
from nav_msgs.msg import OccupancyGrid
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import JointState, PointCloud2
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper


class GridPlanner(object):
    """62411499 髙野幹太"""
    # nav_msgs/OccupancyGridの上で経路を計画するクラス
    WALL_RADIUS = 0.28      # 壁と台車中心の最小距離[m]
    OBSTACLE_RADIUS = 0.34  # 床の障害物と台車中心の最小距離[m]
    PREFERRED_MARGIN = 0.55  # この距離までは離れるほど良いとする[m]
    MARGIN_WEIGHT = 4.0     # 離れることをどれだけ重視するか
    BLOCKED_COST = 100.0    # 通行不可セルのコスト。安全な経路が無いときだけ通る
    SEARCH_MARGIN = 1.5     # 始点と終点の外側を探索する幅[m]

    def __init__(self, occupancy_grid):
        """62411499 髙野幹太"""
        info = occupancy_grid.info
        self.resolution = info.resolution
        self.origin = (info.origin.position.x, info.origin.position.y)
        self.grid = np.array(occupancy_grid.data).reshape(info.height, info.width)

    def to_cell(self, point):
        """62411499 髙野幹太"""
        # map座標を(行, 列)に変換する
        return (int((point[1] - self.origin[1]) / self.resolution),
                int((point[0] - self.origin[0]) / self.resolution))

    def to_point(self, cell):
        """62411499 髙野幹太"""
        # (行, 列)をセル中心のmap座標に変換する
        return (self.origin[0] + (cell[1] + 0.5) * self.resolution,
                self.origin[1] + (cell[0] + 0.5) * self.resolution)

    def _cost_map(self, window, obstacles):
        """62411499 髙野幹太"""
        # 探索範囲内の各セルの通行コストを計算する
        rows = np.arange(window[0], window[1])
        cols = np.arange(window[2], window[3])
        cell_x = self.origin[0] + (cols[np.newaxis, :] + 0.5) * self.resolution
        cell_y = self.origin[1] + (rows[:, np.newaxis] + 0.5) * self.resolution

        cost = np.zeros((len(rows), len(cols)))
        walls = np.argwhere(self.grid[window[0]:window[1], window[2]:window[3]] != 0)
        for row, col in walls:
            dist = np.hypot(cell_x - cell_x[0, col], cell_y - cell_y[row, 0])
            cost = np.maximum(cost, self.BLOCKED_COST * (dist < self.WALL_RADIUS))

        for obs_x, obs_y in obstacles:
            dist = np.hypot(cell_x - obs_x, cell_y - obs_y)
            near = np.clip(self.PREFERRED_MARGIN - dist, 0.0, None) * self.MARGIN_WEIGHT
            cost = np.maximum(cost, near)
            cost = np.maximum(cost, self.BLOCKED_COST * (dist < self.OBSTACLE_RADIUS))
        return cost

    def plan(self, start, goal, obstacles):
        """62411499 髙野幹太"""
        # startからgoalまでの経路をmap座標のリストで返す。到達できなければ空リストを返す
        begin = self.to_cell(start)
        end = self.to_cell(goal)
        pad = int(self.SEARCH_MARGIN / self.resolution)
        window = (max(min(begin[0], end[0]) - pad, 0),
                  min(max(begin[0], end[0]) + pad, self.grid.shape[0]),
                  max(min(begin[1], end[1]) - pad, 0),
                  min(max(begin[1], end[1]) + pad, self.grid.shape[1]))
        cost = self._cost_map(window, obstacles)

        def heuristic(cell):
            return math.hypot(cell[0] - end[0], cell[1] - end[1])

        best = {begin: 0.0}
        parent = {}
        queue = [(heuristic(begin), begin)]
        while queue:
            _, cell = heapq.heappop(queue)
            if cell == end:
                return self._restore(parent, cell, start, goal)
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                 (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = (cell[0] + d_row, cell[1] + d_col)
                if not (window[0] <= nxt[0] < window[1] and window[2] <= nxt[1] < window[3]):
                    continue
                step = math.hypot(d_row, d_col) * (
                    1.0 + cost[nxt[0] - window[0], nxt[1] - window[2]])
                if best[cell] + step < best.get(nxt, float("inf")):
                    best[nxt] = best[cell] + step
                    parent[nxt] = cell
                    heapq.heappush(queue, (best[nxt] + heuristic(nxt), nxt))
        return []

    def wall_distance(self, point, limit=0.6):
        """62411499 髙野幹太"""
        # pointから最も近い壁や家具(静的地図の占有セル)までの距離を返す。limitより遠ければlimitを返す
        row, col = self.to_cell(point)
        pad = int(limit / self.resolution) + 1
        top, left = max(row - pad, 0), max(col - pad, 0)
        cells = np.argwhere(self.grid[top:row + pad + 1, left:col + pad + 1] != 0)
        if len(cells) == 0:
            return limit
        cell_x = self.origin[0] + (cells[:, 1] + left + 0.5) * self.resolution
        cell_y = self.origin[1] + (cells[:, 0] + top + 0.5) * self.resolution
        return min(limit, float(np.min(np.hypot(cell_x - point[0], cell_y - point[1]))))

    def _restore(self, parent, cell, start, goal):
        """62411499 髙野幹太"""
        # 探索結果から経路を復元する。始点と終点は要求された座標そのものにする
        cells = [cell]
        while cells[-1] in parent:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return [start] + [self.to_point(item) for item in cells[1:-1]] + [goal]


def path_clearance(path, obstacles):
    """62411499 髙野幹太"""
    # 経路上の点から最も近い障害物までの距離を返す
    best = float("inf")
    for point in path:
        for obstacle in obstacles:
            best = min(best, math.hypot(point[0] - obstacle[0], point[1] - obstacle[1]))
    return best


def segment_distance(point, start, end):
    """62411499 髙野幹太"""
    # pointから線分start-endまでの距離を返す
    seg_x, seg_y = end[0] - start[0], end[1] - start[1]
    length2 = seg_x ** 2 + seg_y ** 2
    ratio = 0.0 if length2 == 0.0 else max(0.0, min(1.0, (
        (point[0] - start[0]) * seg_x + (point[1] - start[1]) * seg_y) / length2))
    return math.hypot(point[0] - start[0] - ratio * seg_x, point[1] - start[1] - ratio * seg_y)


def cut_path(path, length):
    """62411499 髙野幹太"""
    # 経路の先頭からlength[m]までの部分を返す
    moved = 0.0
    for index in range(1, len(path)):
        moved += math.hypot(path[index][0] - path[index - 1][0],
                            path[index][1] - path[index - 1][1])
        if moved >= length:
            return path[:index + 1]
    return path


class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = ["small_marker", "large_marker", "lego_duplo", "spatula", "nine_hole_peg_test"]
    GRASP_TF_NAME = "object_grasping"
    # 腕の実測値[m]: 手のひらの高さ = リフト + 基準 + ARM_LENGTH * cos(arm_flex)
    ARM_LENGTH = 0.345
    PALM_HEIGHT_DOWN = 0.20     # 手を真下に向けたときの基準の高さ
    PALM_HEIGHT_FORWARD = 0.321  # 手を水平に向けたときの基準の高さ
    PALM_FORWARD_DOWN = 0.122   # 手を真下に向けたときの、腕の長さを除く前方距離
    PALM_FORWARD_FORWARD = 0.277  # 手を水平に向けたときの、腕の長さを除く前方距離
    PALM_LATERAL = 0.106        # 手のひらの横方向のずれ
    MAX_LIFT = 0.65
    MIN_ARM_FLEX = -2.6
    # 把持の実測値[m]
    HOVER_HEIGHT = 0.15         # 物体の上で位置合わせする高さ
    PALM_ABOVE_OBJECT = 0.07    # 上からつかむときの手のひらと物体中心の高さの差
    PALM_ABOVE_SURFACE = 0.105  # 床や机の面より手のひらを低くしない高さ(閉じた指先が面に当たらない)
    # 床の物体では手を少し床に近づける(0.105では閉じた指先が床から1.1〜1.9 cmで、小さな物体を
    # 挟みにくかった。0.093では指先が床に当たった)
    FLOOR_PALM_ABOVE_SURFACE = 0.098
    # tall_table(0.60 m)の上では腕を高く上げるので手が下がりやすく、0.105では閉じた指先が
    # 天板に押し付けられる(平たいclampで指先が天板から約1.5 cm)。2.5 cm程度空ける
    TALL_PALM_ABOVE_SURFACE = 0.12
    TALL_TABLE_HEIGHT = 0.60
    GRASP_DESCENT_STEPS = 6     # 上からつかむときに腕を下ろす分割数
    SIDE_GRASP_MIN_Z = 0.09     # これより低い物体は寝ているか背が低いので上からつかむ
    TABLE_HEIGHTS = (0.40, 0.60)
    PALM_BEHIND_OBJECT = 0.05   # 横からつかむときの手のひらと物体の手前の面の距離
    SIDE_APPROACH = 0.15        # 横からつかむ前に手前で止まる距離
    GRASP_RAISE = 0.04          # つかんだ後に持ち上げる高さ
    RELEASE_RAISE = 0.06        # 物体を落とした後、下がる前に手を持ち上げる高さ
    RIM_BELOW_FORWARD_PALM = 0.10  # 収納用の姿勢(手が水平)で、手のひらが縁より高い分[m]
    DROP_CLEARANCE = 0.03       # 下に向けた手で置くとき、物体の下端と縁の隙間[m]
    SLOT_ALIGN_TOLERANCE = 0.006  # 置き場所のある収納先で、伸ばした手を合わせる許容誤差[m]
    SLOT_ALIGN_TRIALS = 5
    RELEASE_WAIT = 3.0          # 手を開いてから物体が落ち着くまで待つ時間[s]
    RETREAT_SPEED = 0.03        # 物体を落とした後に下がる速さ[m/s]
    LONG_OBJECT_RATIO = 1.3     # bboxの幅が高さのこの倍率を超えたら横長とみなす
    # 長い方向を点群から測り、指をそれと直角に閉じる物体。clampは柄のV字の隙間と、先端がつながった
    # 顎の隙間が長い方向に並ぶので、長い方向に指を閉じると指が顎の輪に入り、放しても手に掛かる
    AXIS_LABELS = ("clamp",)
    POINTS_TOPIC = "/hsrb/head_rgbd_sensor/depth_registered/rectified_points"
    AXIS_MIN_HEIGHT = 0.012     # 面からこれより高い点を物体の点とする[m]
    AXIS_MIN_POINTS = 30
    # 床の物体をつかむ向きの候補[deg]。正面(90)に近い順に、腕の動きが妨げられない向きを選ぶ
    GRASP_YAWS = (90, 60, 120, 30, 150, 0, 180, -30, -150, -60, -120, -90)
    BASE_CLEARANCE = 0.30       # 台車中心と他の床の物体の最小距離[m]
    WALL_CLEARANCE = 0.27       # 台車中心と壁や家具の最小距離[m]
    ARM_CORRIDOR = 0.07         # 台車から物体までの腕と手の通り道の半幅[m]
    FINGER_SPAN = 0.10          # 開いた指が届く範囲(指が開く方向の半幅)[m]
    FINGER_WIDTH = 0.05         # 開いた指の範囲(指が開く方向と直角の半幅)[m]
    SAME_OBJECT_GAP = 0.06      # 手の範囲を調べるとき、これより近い検出は同じ物体とみなす[m]
    OWN_RADIUS_MAX = 0.12       # 大きい物体で同じ物体とみなす半径の上限[m]
    TABLE_MARGIN = 0.05         # 手と机の天板の縁の最小距離[m]
    BASE_RADIUS = 0.25          # 台車の半径(天板の下に入らない距離)[m]
    MAX_APPROACH = 0.40         # 手前からつかむとき、手を差し入れる最大の距離[m]
    FRONT_PALM_ABOVE_SURFACE = 0.06  # 水平な手のひらを面に近づける最低の高さ[m]
    PLAN_TOLERANCE = -0.03      # つかみ方の余裕がこれ未満しか無ければ、つかまずに飛ばす[m]
    # 手と腕が動く空間の大きさ(机の上の物体や天板に当たらないかを調べる)[m]
    HAND_RADIUS = 0.08          # 手の水平方向の半径
    HAND_CLEARANCE = 0.11       # 上から手を下ろすとき、手の中心と上にある天板の縁の最小距離[m]
    ARM_RADIUS = 0.06           # 腕の水平方向の半幅と上下の半分の厚さ
    OBJECT_RADIUS = 0.04        # 見えた物体の大きさ(中心からの半径)
    SHOULDER_HEIGHT = 0.34      # リフトが0のときの腕の付け根の高さ
    ARM_ABOVE_PALM = 0.14       # 手を下に向けたときの、腕(水平)の手のひらからの高さ
    TABLE_ITEM_HEIGHT = 0.12    # 天板の上に物体が置かれうる高さ
    MIN_HOVER = 0.03            # 机の近くで手を構える最低の高さ
    # 背が高く上からつかめない物体は横からつかむ
    # 寝ていたら縦に起こして置く物体(寝たままでは収納先に入らない。windexは長さ0.27 m、
    # Containerの内側は0.18 x 0.26 m)
    STAND_UP_LABELS = ("windex_bottle", "bleach_cleanser", "mustard_bottle")
    END_RATIO = 0.3             # 物体の両端として測る、検出枠の長い方向の割合
    END_HEIGHT_GAP = 0.02       # 両端の高さの差がこれ未満なら、頭と底を見分けられないとみなす[m]
    # 小さいはずの物体のラベルと最大の大きさ[m]。これより大きく見えたら別のラベルを使う
    # (strawberryを近くからmarbleと誤認していた)
    LABEL_MAX_SIZE = {"marble": 0.03, "dice": 0.035}
    # 大きいはずの物体のラベルと最小の大きさ[m]。これより小さく見えたら別のラベルを使う
    # (一辺3.5 cmのcolored_wood_blocksをplumやpudding_boxと誤認していた)
    LABEL_MIN_SIZE = {"plum": 0.045, "peach": 0.05, "pear": 0.05, "apple": 0.06,
                      "orange": 0.06, "lemon": 0.045, "pudding_box": 0.07,
                      "gelatin_box": 0.06, "cracker_box": 0.1, "sugar_box": 0.09}
    CAMERA_FOCAL = 577.0        # カメラの焦点距離[pixel](640 pixel、水平画角58度)
    CAMERA_FRAME = "head_rgbd_sensor_rgb_frame"
    SIDE_GRASP_LABELS = ("windex_bottle", "bleach_cleanser", "mustard_bottle", "chips_can",
                         "master_chef_can", "cracker_box", "sugar_box", "pitcher_base",
                         "potted_meat_can", "tomato_soup_can", "tuna_fish_can")
    MIN_DETECTION_SCORE = 0.3   # これ未満は誤検出が多いので把持しない
    # 小さく平たくてつかめていない物体は後回しにする(pitcherも同じラベルに誤認されるので高さで区別)
    SKIP_LABELS = ()
    SKIP_MAX_HEIGHT = 0.06
    ARM_STEP_NUM = 3            # 腕を大きく動かすときの分割数
    ARM_JOINTS = ("arm_lift_joint", "arm_flex_joint", "arm_roll_joint",
                  "wrist_flex_joint", "wrist_roll_joint")
    MAX_TASK1_OBJECTS = 30
    # 指先だけで挟む小さく軽い物体。手を水平に回すと抜けて落ちるので、上からだけつかみ、
    # 手を下に向けたまま運んで放す(toy_airplaneは腕をたたむときに落ちていた)
    CARRY_DOWN_LABELS = ("toy_airplane",)
    # 手を水平にして手前からだけつかむ物体。cupは天板の縁の近くにあり、上から下ろす手が天板に当たる
    FRONT_ONLY_LABELS = ("cup",)
    CARRY_PALM_HEIGHT = 0.75    # 手を下に向けて運ぶときの手のひらの高さ(机や箱より高い)[m]
    MAX_FAILS_PER_POINT = 3
    DEFER_LABELS = ("cup",)     # 倒れやすいので、他の物体を全部片付けてから最後に入れる
    FAILED_RADIUS = 0.10
    SAME_OBJECT_RADIUS = 0.15
    APPROACH_DISTANCE = 0.55
    NO_APPROACH_Z = 0.35    # これより高い物体は机の上にあるので、探索位置から動かず把持する
    # 引き出し: 手のひらから閉じた指先までの実測値、取っ手の板より奥の隙間までの距離など[m]
    PALM_TO_CLOSED_TIP = 0.09
    PINCH_X_OFFSET = 0.0
    HANDLE_GAP = 0.027
    HANDLE_ALIGN_TOLERANCE = 0.004  # 取っ手に手を合わせる許容誤差[m]
    HANDLE_ALIGN_TRIALS = 5
    # 取っ手の掴み方(左右のずらし[m], 手の高さの調整[m])を順に試す。指は水平に開いて板の
    # 左右から軸に回り込ませる(引っ掛かるので確実に引ける)。台車の位置合わせは1 cm程度ずれるので、
    # 掴み損ねたら左右にずらし、手前が下がって傾いた引き出しに備えて手も少し下げる
    HANDLE_STRATEGIES = ((0.0, 0.0), (0.012, -0.005))  # 最大2回(ユーザー指示)
    # fine_alignの引き出しの最後の位置合わせ。台車は1 cm動くたびに向きが±2.5°変わり、
    # 0.6 m先の手のひらが左右に振れるので、許容誤差に入るまで測り直して合わせ直す
    HANDLE_FINE_TOLERANCE = 0.006
    HANDLE_FINE_TRIALS = 10
    HANDLE_RETRY_BACK = 0.05    # 掴み直す前に取っ手から離れる距離[m]
    SETTLE_WAIT = 1.0           # 指を閉じる前に、伸ばした腕の揺れが収まるのを待つ時間[s]
    FINE_MOVE = 0.02            # これより短い台車の移動はゆっくり動かして行き過ぎを防ぐ[m]
    FINE_MOVE_DURATION = 2.0    # 短い移動にかける最短の時間[s]
    GRASP_PLAN_TRIALS = 2       # 把持に失敗したとき、別のつかみ方で試す回数(最初を含む)
    DRAWER_APPROACH = 0.15
    DRAWER_RELEASE_BACK = 0.15
    # 物体を掴むと指が閉じ切らないことを利用した把持判定の閾値[hand_motor_jointの値, rad]
    # 空振りで閉じると0.05以下で振動する。strawberryを持つと0.09〜0.105、pitcherは0.35
    GRIP_HOLD_THRESHOLD = 0.07
    GRIP_EMPTY_THRESHOLD = 0.03  # これ以下なら空。閾値との間は、元の位置に物体が残っているかを見る
    HEAD_HEIGHT = 1.0           # 腕をたたんだときのカメラの高さ(物体を見る首の角度の計算用)[m]
    TABLE_RETREAT = 0.25        # 机の縁の近くでつかんだ後、腕をたたむ前に下がる距離[m]
    # 手を下に向けたときの、手首と腕の上端の手のひらからの高さ[m]。
    # URDFでは手首の軸が手のひらの0.1405 m上、腕の厚さの半分が0.044 m
    WRIST_ABOVE_PALM = 0.19
    WRIST_FLEX_MIN = -1.9       # 手首を下に曲げられる限界(URDFの下限-1.92)[rad]
    FRONT_MIN_HEIGHT = 0.025    # 手前から水平につかむ物体の、面からの最低の高さ[m]
    FRONT_MAX_WIDTH = 0.10      # これより幅の広い物体は水平な手の指の間に入らない[m]
    TABLE_FRONT_MIN_HEIGHT = 0.03  # 机の上でこれより背のある物体は、横からつかむ方法を先に試す[m]
    TABLE_FRONT_MAX_WIDTH = 0.07   # ただし幅がこれ未満の細い物体だけ(softball 9.6 cmは上から)[m]
    TASK1_TIME_LIMIT = None     # これを過ぎたら新しい物体に手を付けない[s]。Noneなら制限しない
    GRIP_SAMPLES = 10
    # 台車の細かい位置合わせ: omni_base_controllerに送る目標姿勢の速さ
    SHIFT_SPEED = 0.05              # [m/s]
    SHIFT_YAW_SPEED = math.radians(10.0)  # [rad/s]
    # Task 2a: 障害物を探す範囲(map座標)、再検出までの走行距離、経路追従の速度と許容誤差
    OBSTACLE_AREA = {"x": (1.0, 3.3), "y": (1.9, 3.7), "z_max": 0.25}
    REPLAN_INTERVAL = 0.5
    # Task 1: 床の物体を押さないよう、見えた床の物体を避けて移動する
    FLOOR_OBJECT_Z = 0.25       # これより低い物体を床の物体とみなす[m]
    AVOID_ARRIVAL = 0.10        # 経路追従の後、これより離れていればmove_baseで仕上げる[m]
    FOLLOW_SPEED = 0.2
    FOLLOW_TOLERANCE = 0.03
    FOLLOW_PASS_TOLERANCE = 0.06  # 経路の途中の点を通過したとみなす距離[m]
    FOLLOW_TIMEOUT = 40.0
    ALIGN_TOLERANCE = 0.008
    ALIGN_YAW_TOLERANCE = 0.5
    ALIGN_TRIALS = 3
    ALIGN_YAW_DEADBAND = 2.0    # 台車はこれより小さな回転に追従しない[deg]
    ALIGN_YAW_STUCK = 0.5       # 回転を指令して向きの変化がこれ未満なら、追従しなかったとみなす[deg]
    ALIGN_YAW_LARGE = 4.0       # 追従しなかった後は、これより大きくずれたときだけ回す[deg]
    # Task 2b: 指示文の形式「<物体名> to person <left|right>」と配達先
    INSTRUCTION_PATTERN = re.compile(r"^(\w+) to person (left|right)$")
    PERSON_WAYPOINTS = {"left": "person_a", "right": "person_b"}
    INSTRUCTION_TIMEOUT = 15.0
    SHELF_GRASP_RETRY = 2
    SHELF_TILTS = (-0.3, -0.6, 0.0)

    def __init__(self):
        # 変数の初期化
        self.instruction_list = []
        self.detection_list = []

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses = self.load_json(self.get_path(["config", "poses.json"]))
        self.deposit_count = {}
        self.deferring = True    # DEFER_LABELSの物体を後回しにしている間はTrue
        self.deferred = []       # 後回しにした物体を見つけた探索位置[(場所, 姿勢)]
        self.closed_categories = set()  # 引き出しを開けられず、収納できないカテゴリ
        self.closed_drawers = set()     # 開けられなかった引き出しの名前
        self.last_seen = None    # 把持に失敗した後、元の位置に見えた物体の座標
        self.hang = self.PALM_TO_CLOSED_TIP  # 上からつかんだ物体が手のひらから下に出る長さ
        self.carry_down = False  # 手を下に向けたまま運んでいる間はTrue
        self.floor_objects = []  # Task 1で見えた床の物体のmap座標
        self.seen_objects = []   # Task 1で見えた物体の3次元のmap座標(手が当たるかの判定用)
        self.planner = None      # Task 1で床の物体を避けて移動するときの経路計画
        self.handle_palms = {}   # 取っ手を掴めたときの手のひらのmap座標(引き出しの名前ごと)

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.base_client = actionlib.SimpleActionClient(
            "/hsrb/omni_base_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
        self.base_client.wait_for_server(rospy.Duration(10.0))

        self.instruction_sub = rospy.Subscriber(
            "/message", String, self.instruction_cb, queue_size=10)
        self.detection_sub = rospy.Subscriber(
            "/detect_msg", String, self.detection_cb, queue_size=10)

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
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(
                parent, child, rospy.Time.now(), rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name):
        """
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            coord = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, coord[0], coord[1], coord[2])
            return self.goto_pos(coord)

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, coord):
        """
        座標[x, y, yaw]で指定された場所に移動する。
        Task 1で床の物体が見えていれば、それを避ける経路を計画して走る
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", coord[0], coord[1], coord[2])
        if self.planner is not None and self.floor_objects:
            if self.goto_avoiding(coord):
                return True
        return omni_base.go_abs(coord[0], coord[1], coord[2])

    def goto_avoiding(self, coord):
        """62411499 髙野幹太"""
        # 床の物体を避ける経路をA*で計画して追従し、最後に向きを合わせる。
        # move_baseは床の小物を地図に載せないので、物体を押したり台車の下に挟んだりする
        pose = self.get_base_pose()
        if pose is None:
            return False
        path = self.planner.plan(pose[:2], coord[:2], self.floor_objects)
        if not path:
            return False
        rospy.loginfo("avoiding path: %d points, clearance %.2f",
                      len(path), path_clearance(path, self.floor_objects))
        self.follow_path(path, yaw_deg=coord[2])
        pose = self.get_base_pose()
        if pose is None or math.hypot(coord[0] - pose[0], coord[1] - pose[1]) > self.AVOID_ARRIVAL:
            return False
        diff = math.radians(coord[2]) - pose[2]
        d_yaw = math.atan2(math.sin(diff), math.cos(diff))
        self.shift_base(0.0, 0.0, d_yaw if abs(math.degrees(d_yaw)) > self.ALIGN_YAW_DEADBAND
                        else 0.0)
        return True

    def remember_floor_object(self, point):
        """62411499 髙野幹太"""
        # 床の物体のmap座標を記憶する。既に記憶した物体の近くなら位置を更新する
        if point.z >= self.FLOOR_OBJECT_Z:
            return
        for index, (obj_x, obj_y) in enumerate(self.floor_objects):
            if math.hypot(point.x - obj_x, point.y - obj_y) < self.SAME_OBJECT_RADIUS:
                self.floor_objects[index] = (point.x, point.y)
                return
        self.floor_objects.append((point.x, point.y))

    def remember_object(self, point):
        """62411499 髙野幹太"""
        # 見えた物体の3次元座標を記憶する(床の物体なら台車の経路計画にも使う)
        self.remember_floor_object(point)
        for index, obj in enumerate(self.seen_objects):
            if (math.hypot(point.x - obj[0], point.y - obj[1]) < self.SAME_OBJECT_GAP
                    and abs(point.z - obj[2]) < self.SAME_OBJECT_GAP):
                self.seen_objects[index] = (point.x, point.y, point.z)
                return
        self.seen_objects.append((point.x, point.y, point.z))

    def forget_floor_object(self, point):
        """62411499 髙野幹太"""
        # つかんで運んだ床の物体を記憶から消す
        self.floor_objects = [(obj_x, obj_y) for obj_x, obj_y in self.floor_objects
                              if math.hypot(point.x - obj_x, point.y - obj_y)
                              >= self.SAME_OBJECT_RADIUS]
        self.seen_objects = [obj for obj in self.seen_objects
                             if math.hypot(point.x - obj[0], point.y - obj[1])
                             >= self.SAME_OBJECT_GAP or abs(point.z - obj[2]) >= 0.1]

    def change_pose(self, name):
        """
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

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
        BBox情報から把持座標を取得する。取得できなければNoneを返す
        """
        # BBox情報からtfを生成して、座標を取得
        res = self.tf_from_bbox_clt.call(
            SetTransformFromBBoxRequest(bbox=bbox, frame=self.GRASP_TF_NAME))
        if not res.result:
            return None
        rospy.sleep(1.0)  # tfが安定するのを待つ
        transform = self.get_relative_coordinate("map", self.GRASP_TF_NAME)
        return None if transform is None else transform.translation

    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None:
            return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str = ""
        for obj in obj_list:
            info_str = "{:<15}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n".format(
                obj.label, obj.score, obj.x, obj.y, obj.w, obj.h)
            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({"bbox": obj, "score": score, "label": obj.label})
                extract_str += "- extracted: {:07.3f} ".format(score) + info_str

        rospy.loginfo(extract_str + ignore_str)

        # objが一つもない場合は、Noneを返す
        if not extracted:
            return None

        # つかむべきかのscoreが一番高い物体を返す
        best = max(extracted, key=lambda x: x["score"])
        obj = best["bbox"]
        rospy.loginfo("selected bbox: %s (%.2f%%, %3d, %3d, %3d, %3d)",
                      obj.label, obj.score * 100, obj.x, obj.y, obj.w, obj.h)
        return best

    @classmethod
    def calc_score_bbox(cls, bbox):
        """
        detector_msgs/BBoxのスコアを計算する
        """
        gravity_x = bbox.x + bbox.w / 2
        gravity_y = bbox.y + bbox.h / 2
        xy_diff = abs(320 - gravity_x) / 320 + abs(360 - gravity_y) / 240

        return 1 / max(xy_diff, 1e-6)

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """
        label名が一致するオブジェクトの中から最も把持すべき物体のbboxを返す
        """
        match_objs = [obj for obj in obj_list if obj.label == label]
        if not match_objs:
            rospy.logwarn("Cannot find a object which labeled with similar name.")
            return None
        return cls.get_most_graspable_bbox(match_objs)

    @classmethod
    def extract_target_obj_and_person(cls, instruction):
        """
        指示文「<物体名> to person <left|right>」から対象物と人を抽出する。
        形式が合わない文(例: done)に対しては(None, None)を返す
        """
        match = cls.INSTRUCTION_PATTERN.match(instruction.strip())
        if match is None:
            return None, None
        return match.group(1), match.group(2)

    def wait_for_instruction(self, timeout):
        """62414107 長澤桃峨"""
        # 有効な指示文を新しい順に探して返す。timeout[s]待っても無ければ(None, None)を返す
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown():
            for instruction in reversed(self.instruction_list):
                target = self.extract_target_obj_and_person(instruction)
                if target[0] is not None:
                    return target
            if rospy.Time.now() > deadline:
                break
            rospy.sleep(0.5)
        return None, None

    @staticmethod
    def set_gripper(value, wait=2.0):
        """62414107 長澤桃峨"""
        # ハンドを動かす。gripper.commandの固定6秒待ちを短くして全体の時間を節約する
        gripper.GRIPPER_CMD.set_joint_value_target(str("hand_motor_joint"), value)
        gripper.GRIPPER_CMD.go()
        rospy.sleep(wait)

    def is_holding(self):
        """62414107 長澤桃峨"""
        # 閉じた指の開き具合(hand_motor_joint)から物体を保持しているかを判定する
        return self.grip_opening() >= self.GRIP_HOLD_THRESHOLD

    def grip_opening(self):
        """62414107 長澤桃峨"""
        # 閉じた指の開き具合(hand_motor_joint)を返す。
        # 空振りでも値が細かく振動するので、数回の平均を返す
        values = []
        for _ in range(self.GRIP_SAMPLES):
            try:
                msg = rospy.wait_for_message("/hsrb/joint_states", JointState, 1.0)
            except rospy.ROSException:
                continue
            if "hand_motor_joint" in msg.name:
                values.append(msg.position[msg.name.index("hand_motor_joint")])
        opening = sum(values) / len(values) if values else 0.0
        rospy.loginfo("gripper opening: %.3f", opening)
        return opening

    def object_remains(self, point):
        """62414107 長澤桃峨"""
        # 腕をたたんだ後に元の位置を見て、残っている物体の座標を返す。無ければNoneを返す
        base = self.get_base_pose()
        if base is None:
            return None
        dist = math.hypot(point.x - base[0], point.y - base[1])
        whole_body.move_head_tilt(-math.atan2(self.HEAD_HEIGHT - point.z, max(dist, 0.3)))
        rospy.sleep(1.0)
        for bbox in self.get_latest_detection().bboxes:
            seen = self.get_grasp_coordinate(bbox)
            if (seen is not None and abs(seen.z - point.z) < 0.1
                    and math.hypot(seen.x - point.x, seen.y - point.y) < self.SAME_OBJECT_RADIUS):
                return seen
        return None

    def check_grasp(self, point):
        """62414107 長澤桃峨"""
        # 腕をたたんだ後に、物体を保持しているかを返す。指の開き具合ではっきりしなければ
        # 元の位置を見て、物体が無くなっていれば保持していると判断する(つかめた物体を落とさない)
        self.last_seen = None
        opening = self.grip_opening()
        if opening >= self.GRIP_HOLD_THRESHOLD:
            return True
        # 空なら掴み直しに使うため、はっきりしなければ保持の判断のため、元の位置を見る
        self.last_seen = self.object_remains(point)
        if opening <= self.GRIP_EMPTY_THRESHOLD:
            return False
        rospy.loginfo("grip is unclear (%.3f): the object %s", opening,
                      "is still there" if self.last_seen else "is gone, so it is held")
        return self.last_seen is None

    def near_table(self, point, distance):
        """62401543 池畑一志"""
        # 物体が、それより高い机の天板の縁からdistance以内(天板の下を含む)ならTrueを返す
        return any(self.rect_distance((point.x, point.y), table) < distance
                   and point.z < table["top"] - self.TABLE_MARGIN
                   for table in self.coordinates.get("tables", []))

    def retreat_from_tables(self, point, yaw_deg):
        """62401543 池畑一志"""
        # 机の縁の近くでつかんだら、腕をたたむときに手が天板の裏に当たらないよう台車を下げる
        if self.near_table(point, self.TABLE_RETREAT):
            head, _ = self.heading(yaw_deg)
            self.shift_base(-self.TABLE_RETREAT * head[0], -self.TABLE_RETREAT * head[1], 0.0)

    def release_object(self):
        """62414107 長澤桃峨"""
        # 手を開いて物体を放す
        self.set_gripper(1, wait=self.RELEASE_WAIT)

    def arm_joints(self, palm_z, hand_down):
        """62414107 長澤桃峨"""
        # 手のひらを高さpalm_zに置く腕の関節値と、台車中心から手のひらまでの前方距離を返す。
        # 手は真下(hand_down=True)または水平に向ける。高さが足りるうちはリフトだけで調整する
        base_height = self.PALM_HEIGHT_DOWN if hand_down else self.PALM_HEIGHT_FORWARD
        lift = palm_z - base_height
        flex = -math.pi / 2
        if lift < 0.0:
            ratio = max(-1.0, lift / self.ARM_LENGTH)
            flex = max(self.MIN_ARM_FLEX, -math.acos(ratio))
            lift = 0.0
        lift = min(lift, self.MAX_LIFT)
        wrist = -math.pi - flex if hand_down else -math.pi / 2 - flex
        forward = self.PALM_FORWARD_DOWN if hand_down else self.PALM_FORWARD_FORWARD
        joints = {"arm_lift_joint": lift, "arm_flex_joint": flex, "arm_roll_joint": 0.0,
                  "wrist_flex_joint": wrist, "wrist_roll_joint": 0.0}
        return joints, forward + self.ARM_LENGTH * abs(math.sin(flex))

    def move_arm_slowly(self, joints, steps=None):
        """62414107 長澤桃峨"""
        # 腕の関節値を少しずつ変えて動かす。
        # move_to_joint_positionsは1秒で動き切るので、大きく動かすと腕を振り回してしまう。
        # 腕の制御器は一部の関節だけの指令を受け付けないので、指定の無い関節は今の値で送る
        names = whole_body.ARM_CMD.get_active_joints()
        start = dict(zip(names, whole_body.ARM_CMD.get_current_joint_values()))
        goal = {name: joints.get(name, start[name]) for name in self.ARM_JOINTS if name in start}
        steps = steps or self.ARM_STEP_NUM
        for step in range(1, steps + 1):
            ratio = step / steps
            whole_body.move_to_joint_positions(
                {name: start[name] + (value - start[name]) * ratio
                 for name, value in goal.items()})

    def reach_out(self, joints):
        """62414107 長澤桃峨"""
        # 手が周囲に当たらないよう、高い位置で腕を前に向けてから、目標の姿勢へ腕を伸ばす。
        # 手を下に向ける姿勢では、腕に直角な手のまま振り下ろすと、途中で指先が目標より
        # 7〜12 cm前に出て机の縁や天板の上の物体に当たる。そこで手首を下限まで下に曲げたまま
        # 手が真下を向く角度まで腕を倒し、その後は手を真下に向けたまま倒す(前に出るのは2 cm以内)
        self.move_arm_slowly({"arm_lift_joint": joints["arm_lift_joint"]}, steps=1)
        hand_down = abs(joints["arm_flex_joint"] + joints["wrist_flex_joint"] + math.pi) < 0.05
        if not hand_down:
            self.move_arm_slowly({"arm_roll_joint": 0.0, "arm_flex_joint": 0.0,
                                  "wrist_flex_joint": -math.pi / 2, "wrist_roll_joint": 0.0})
            self.move_arm_slowly(joints)
            return
        roll = joints.get("wrist_roll_joint", 0.0)
        self.move_arm_slowly({"arm_roll_joint": 0.0, "arm_flex_joint": 0.0,
                              "wrist_flex_joint": self.WRIST_FLEX_MIN, "wrist_roll_joint": roll})
        flex_down = -math.pi - self.WRIST_FLEX_MIN  # 手首が下限のまま手が真下を向く腕の角度
        if joints["arm_flex_joint"] < flex_down:
            self.move_arm_slowly({"arm_flex_joint": flex_down,
                                  "wrist_flex_joint": self.WRIST_FLEX_MIN})
        self.move_arm_slowly(joints)

    def fold_arm(self):
        """62414107 長澤桃峨"""
        # reach_outと逆の順序で腕をたたみ、搬送姿勢に戻す
        self.move_arm_slowly({"arm_flex_joint": 0.0, "wrist_flex_joint": -math.pi / 2,
                              "wrist_roll_joint": 0.0})
        self.move_arm_slowly({"arm_roll_joint": -math.pi / 2}, steps=2)
        self.change_pose("all_neutral")

    def object_axis(self, target):
        """62401543 池畑一志"""
        # 検出枠の中の点群から、面より高い点(物体)を取り出し、長い方向の角度[rad](map座標)と
        # 中心(x, y)と長さ・幅[m]を返す。測れなければNoneを返す
        bbox = target.get("bbox")
        if bbox is None:
            return None
        try:
            cloud = rospy.wait_for_message(self.POINTS_TOPIC, PointCloud2, timeout=5.0)
        except rospy.ROSException:
            return None
        transform = self.get_relative_coordinate("map", cloud.header.frame_id)
        if transform is None:
            return None
        uvs = [(u, v) for u in range(int(bbox.x), int(bbox.x + bbox.w))
               for v in range(int(bbox.y), int(bbox.y + bbox.h))]
        points = np.array(list(pc2.read_points(cloud, field_names=("x", "y", "z"),
                                               skip_nans=True, uvs=uvs)))
        if len(points) < self.AXIS_MIN_POINTS:
            return None
        rot = transform.rotation
        quat_x, quat_y, quat_z, quat_w = rot.x, rot.y, rot.z, rot.w
        matrix = np.array([
            [1 - 2 * (quat_y ** 2 + quat_z ** 2), 2 * (quat_x * quat_y - quat_z * quat_w),
             2 * (quat_x * quat_z + quat_y * quat_w)],
            [2 * (quat_x * quat_y + quat_z * quat_w), 1 - 2 * (quat_x ** 2 + quat_z ** 2),
             2 * (quat_y * quat_z - quat_x * quat_w)],
            [2 * (quat_x * quat_z - quat_y * quat_w), 2 * (quat_y * quat_z + quat_x * quat_w),
             1 - 2 * (quat_x ** 2 + quat_y ** 2)]])
        trans = transform.translation
        points = points.dot(matrix.T) + np.array([trans.x, trans.y, trans.z])
        surface = self.surface_of(target["point"].z)
        points = points[points[:, 2] > surface + self.AXIS_MIN_HEIGHT]
        if len(points) < self.AXIS_MIN_POINTS:
            return None
        center = points[:, :2].mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((points[:, :2] - center).T))
        major = vectors[:, int(np.argmax(values))]
        angle = math.atan2(major[1], major[0])
        along = (points[:, :2] - center).dot(major)
        across = (points[:, :2] - center).dot(np.array([-major[1], major[0]]))
        length = np.percentile(along, 95) - np.percentile(along, 5)
        width = np.percentile(across, 95) - np.percentile(across, 5)
        rospy.loginfo("%s axis %.0f deg, center (%.3f, %.3f), %.3f x %.3f m (%d points)",
                      target["label"], math.degrees(angle), center[0], center[1],
                      length, width, len(points))
        return {"angle": angle, "center": (float(center[0]), float(center[1])),
                "length": float(length), "width": float(width)}

    def axis_roll(self, axis, yaw_deg):
        """62401543 池畑一志"""
        # 手を下に向けたとき、指が物体の長い方向と直角に閉じる手首の回転角(-90〜90度)を返す。
        # 指の開く向きは、台車の向きから(90度 - 手首の回転角)の方向(URDFの順運動学で確認)
        roll = math.radians(yaw_deg) - axis["angle"]
        return (roll + math.pi / 2) % math.pi - math.pi / 2

    @classmethod
    def wrist_roll_for(cls, bbox):
        """62401543 池畑一志"""
        # 上からつかむときの手首の回転角を返す。指は横に開くので、
        # 画像上で横長の物体は手首を90度回して短い方向を挟む
        return math.pi / 2 if bbox.w > bbox.h * cls.LONG_OBJECT_RATIO else 0.0

    def surface_limit(self, height):
        """62401543 池畑一志"""
        # 物体の高さから、その下の面(床か机)に手のひらが当たらない最低の高さを返す
        tables = [table for table in self.TABLE_HEIGHTS if table < height]
        if not tables:
            return self.FLOOR_PALM_ABOVE_SURFACE
        if max(tables) >= self.TALL_TABLE_HEIGHT:
            return max(tables) + self.TALL_PALM_ABOVE_SURFACE
        return max(tables) + self.PALM_ABOVE_SURFACE

    @staticmethod
    def heading(yaw_deg):
        """62411499 髙野幹太"""
        # 台車の向きyaw_degの前方と左方向の単位ベクトルを返す
        rad = math.radians(yaw_deg)
        return (math.cos(rad), math.sin(rad)), (-math.sin(rad), math.cos(rad))

    def base_for_palm(self, palm, yaw_deg, forward):
        """62411499 髙野幹太"""
        # 向きyaw_degで手のひらをpalm(x, y)に置くための台車の位置[x, y, yaw]を返す。
        # forwardは台車中心から手のひらまでの前方距離。手のひらは台車中心より左にずれている
        head, left = self.heading(yaw_deg)
        return [palm[0] - forward * head[0] - self.PALM_LATERAL * left[0],
                palm[1] - forward * head[1] - self.PALM_LATERAL * left[1], yaw_deg]

    def above_grasp_joints(self, point):
        """62401543 池畑一志"""
        # 上からつかむときの腕の関節値と、台車中心から手のひらまでの前方距離を返す
        palm_z = max(point.z + self.PALM_ABOVE_OBJECT, self.surface_limit(point.z))
        return self.arm_joints(palm_z, hand_down=True)

    def front_grasp_joints(self, point):
        """62414107 長澤桃峨"""
        # 手を水平にして手前からつかむときの腕の関節値と、台車中心から手のひらまでの前方距離を返す
        surface = self.surface_of(point.z) + self.FRONT_PALM_ABOVE_SURFACE
        return self.arm_joints(max(point.z, surface), hand_down=False)

    def hover_height(self, point):
        """62401543 池畑一志"""
        # 上からつかむとき、物体の上で手を構える高さ(手のひらと、つかむ高さとの差)を返す。
        # 物体より高い机の近くでは、構えた手首が天板とその上の物体に当たらない高さまで下げる
        palm_z = max(point.z + self.PALM_ABOVE_OBJECT, self.surface_limit(point.z))
        hover = self.HOVER_HEIGHT
        for table in self.coordinates.get("tables", []):
            if point.z >= table["top"] - self.TABLE_MARGIN:
                continue
            if self.rect_distance((point.x, point.y), table) < self.HAND_RADIUS + 0.1:
                # 判定(hand_margin)で天板の下に入ると見なされるよう、わずかに低くする
                hover = min(hover, table["top"] - self.TABLE_MARGIN - self.WRIST_ABOVE_PALM
                            - palm_z - 0.002)
        return max(self.MIN_HOVER, hover)

    def table_backoff(self, point, yaw_deg):
        """62401543 池畑一志"""
        # 上からつかむとき、手を構える位置を物体から手前に戻す距離を返す。
        # 物体の上に天板がかかっていると、上から下ろす手が天板を通り抜けて当たるので、
        # 手を天板の外(縁からHAND_CLEARANCE)で天板より低く構え、台車で物体の上まで進める。
        # 戻しても天板の外に出られなければNoneを返す
        tables = [table for table in self.coordinates.get("tables", [])
                  if point.z < table["top"] - self.TABLE_MARGIN]
        head, _ = self.heading(yaw_deg)
        back = 0.0
        while back <= self.MAX_APPROACH:
            palm = (point.x - back * head[0], point.y - back * head[1])
            if all(self.rect_distance(palm, table) >= self.HAND_CLEARANCE for table in tables):
                return back
            back += 0.01
        return None

    def grasp_from_above(self, point, wrist_roll=0.0, yaw_deg=90.0):
        """62401543 池畑一志"""
        # 台車を止めて腕だけで、物体の真上から下ろしてつかむ。台車は物体からyaw_degの向きに置く
        joints, forward = self.above_grasp_joints(point)
        joints["wrist_roll_joint"] = wrist_roll
        hover = dict(joints, arm_lift_joint=joints["arm_lift_joint"] + self.hover_height(point))
        # 物体は面に立っていたので、手のひらから面までの高さだけ下に出ている
        palm_z = max(point.z + self.PALM_ABOVE_OBJECT, self.surface_limit(point.z))
        self.hang = max(self.PALM_TO_CLOSED_TIP, palm_z - self.surface_of(point.z))

        head, _ = self.heading(yaw_deg)
        back = self.table_backoff(point, yaw_deg) or 0.0
        start = (point.x - back * head[0], point.y - back * head[1])
        tall = self.surface_of(point.z) >= self.TALL_TABLE_HEIGHT
        if tall:
            # tall_tableの上の物体は、机から離れた位置で先にリフトを一番上まで上げて腕を伸ばし、
            # 高く構えたまま台車を進めてから下ろす(机の近くで腕を上げると天板に当たる)
            hover = dict(hover, arm_lift_joint=self.MAX_LIFT)
            self.set_gripper(1, wait=1.0)
            self.reach_out(hover)
            self.goto_pos(self.base_for_palm(start, yaw_deg, forward))
        else:
            self.goto_pos(self.base_for_palm(start, yaw_deg, forward))
            self.set_gripper(1, wait=1.0)
            self.reach_out(hover)
        if back > 0.0:
            # 天板の外で天板より低く構えた手を、台車で物体の上まで進める
            rospy.loginfo("hold the hand %.2f m before the table edge", back)
        # 台車は物体の上で手を合わせるときだけ動かし、つかむ動作は腕だけで行う
        self.align_hand_over((point.x, point.y), yaw_deg)
        self.move_arm_slowly(joints, steps=self.GRASP_DESCENT_STEPS)
        rospy.sleep(self.SETTLE_WAIT)
        self.set_gripper(0)
        self.move_arm_slowly(hover)
        if tall:
            # 一番上まで上げたまま机から離れてから腕をたたむ
            self.shift_base(-self.TABLE_RETREAT * head[0], -self.TABLE_RETREAT * head[1], 0.0)
        else:
            self.retreat_from_tables(point, yaw_deg)
        if self.carry_down:
            # 手首を回さず、手を下に向けたまま腕を水平にして高く上げる
            joints, _ = self.arm_joints(self.CARRY_PALM_HEIGHT, hand_down=True)
            self.move_arm_slowly(dict(joints, wrist_roll_joint=wrist_roll))
        else:
            self.fold_arm()
        return self.check_grasp(point)

    def grasp_from_front(self, point, yaw_deg=90.0, approach=None):
        """62414107 長澤桃峨"""
        # 手を水平にして手前から近づけてつかむ。腕の高さは固定で台車を前進させる。
        # 台車は物体からyaw_degの向きに置き、手はapproach[m]手前で腕を伸ばし切ってから差し入れる
        joints, forward = self.front_grasp_joints(point)
        head, _ = self.heading(yaw_deg)
        approach = approach or self.SIDE_APPROACH + self.front_overshoot(point)
        # 物体は面に立っていたので、手のひらから面までの高さだけ下に出ている
        palm_z = max(point.z, self.surface_of(point.z) + self.FRONT_PALM_ABOVE_SURFACE)
        self.hang = palm_z - self.surface_of(point.z)
        palm = (point.x - self.PALM_BEHIND_OBJECT * head[0],
                point.y - self.PALM_BEHIND_OBJECT * head[1])
        start = (palm[0] - approach * head[0], palm[1] - approach * head[1])

        self.goto_pos(self.base_for_palm(start, yaw_deg, forward))
        self.set_gripper(1, wait=1.0)
        self.reach_out(joints)
        self.align_hand_over(start, yaw_deg)
        self.align_hand_over(palm, yaw_deg)
        rospy.sleep(self.SETTLE_WAIT)
        self.set_gripper(0)
        self.move_arm_slowly({"arm_lift_joint": joints["arm_lift_joint"] + self.GRASP_RAISE},
                             steps=1)
        self.align_hand_over(start, yaw_deg)
        self.retreat_from_tables(point, yaw_deg)
        self.fold_arm()
        return self.check_grasp(point)

    def height_above_surface(self, point):
        """62401543 池畑一志"""
        # 物体の中心の、下の面(床か机)からの高さを返す
        return point.z - max([t for t in self.TABLE_HEIGHTS if t < point.z] or [0.0])

    def is_skipped(self, label, point):
        """62401543 池畑一志"""
        # つかむのが難しいので後回しにする物体ならTrueを返す
        return label in self.SKIP_LABELS and self.height_above_surface(point) < self.SKIP_MAX_HEIGHT

    def is_front_grasp(self, target):
        """62401543 池畑一志"""
        # 背の高い物体なら、手を水平にして手前からつかむのでTrueを返す
        return (target["label"] in self.SIDE_GRASP_LABELS
                and self.height_above_surface(target["point"]) >= self.SIDE_GRASP_MIN_Z)

    def surface_of(self, height):
        """62414107 長澤桃峨"""
        # 高さheightの物体が載っている面(床か机)の高さを返す
        return max([t for t in self.TABLE_HEIGHTS if t < height] or [0.0])

    def table_over(self, point):
        """62401543 池畑一志"""
        # 物体の上に天板がある机(物体が天板の下にある机)を返す。無ければNoneを返す
        for table in self.coordinates.get("tables", []):
            if (table["x"][0] <= point.x <= table["x"][1]
                    and table["y"][0] <= point.y <= table["y"][1]
                    and point.z < table["top"] - self.TABLE_MARGIN):
                return table
        return None

    @staticmethod
    def rect_distance(xy, table):
        """62414107 長澤桃峨"""
        # 点xyから机の天板の範囲までの水平距離を返す。範囲の内側なら負の値(縁までの距離)を返す
        out_x = max(table["x"][0] - xy[0], xy[0] - table["x"][1])
        out_y = max(table["y"][0] - xy[1], xy[1] - table["y"][1])
        if out_x <= 0.0 and out_y <= 0.0:
            return max(out_x, out_y)
        return math.hypot(max(out_x, 0.0), max(out_y, 0.0))

    def front_overshoot(self, point):
        """62414107 長澤桃峨"""
        # 手を水平にして低い物体をつかむとき、腕を伸ばす途中で手が目標より前に出る距離を返す。
        # 腕を上から振り下ろすので、腕が水平になったときに手が最も前に出る
        flex = self.front_grasp_joints(point)[0]["arm_flex_joint"]
        if flex >= -math.pi / 2:
            return 0.0
        return self.ARM_LENGTH * (1.0 - math.sin(-flex))

    def front_approach(self, target, yaw_deg):
        """62401543 池畑一志"""
        # 手前からつかむとき、腕を伸ばし切る位置から手を差し入れる距離を返す。
        # 腕を伸ばす途中で手が前に出る分(front_overshoot)だけ遠くで腕を伸ばす。
        # 物体が天板の下にあれば、前に出た指先も天板の外に収まる距離にする
        point = target["point"]
        overshoot = self.front_overshoot(point)
        approach = self.SIDE_APPROACH + overshoot
        table = self.table_over(point)
        if table is None:
            return approach if approach <= self.MAX_APPROACH else None
        head, _ = self.heading(yaw_deg)
        palm = (point.x - self.PALM_BEHIND_OBJECT * head[0],
                point.y - self.PALM_BEHIND_OBJECT * head[1])
        # 指先(手のひらの先PALM_TO_CLOSED_TIP)が天板の縁からTABLE_MARGIN外に出るまで戻す
        while approach <= self.MAX_APPROACH:
            tip = (palm[0] - (approach - overshoot - self.PALM_TO_CLOSED_TIP) * head[0],
                   palm[1] - (approach - overshoot - self.PALM_TO_CLOSED_TIP) * head[1])
            if self.rect_distance(tip, table) >= self.TABLE_MARGIN:
                return approach
            approach += 0.02
        return None

    def grasp_reach(self, target, plan):
        """62401543 池畑一志"""
        # つかむ位置に置いた台車の中心から、物体までの前方距離を返す
        point = target["point"]
        if plan[0] == "front":
            forward = self.front_grasp_joints(point)[1]
            return forward + self.PALM_BEHIND_OBJECT + plan[3]
        return self.above_grasp_joints(point)[1]

    def base_margin(self, target, plan):
        """62401543 池畑一志"""
        # 床の物体をつかむときの、台車と腕の余裕[m]を返す。負なら当たる。
        # 台車の位置(つかむ位置と見直す位置)と壁・机の天板・他の床の物体の距離、
        # 台車から物体までの腕の通り道と他の床の物体の距離を調べる
        point = target["point"]
        yaw_deg = plan[1]
        base = self.base_for_palm((point.x, point.y), yaw_deg, self.grasp_reach(target, plan))
        head, _ = self.heading(yaw_deg)
        view = (point.x - self.APPROACH_DISTANCE * head[0],
                point.y - self.APPROACH_DISTANCE * head[1])
        others = [obj for obj in self.floor_objects
                  if math.hypot(obj[0] - point.x, obj[1] - point.y) >= self.SAME_OBJECT_RADIUS]
        margins = [self.planner.wall_distance(xy) - self.WALL_CLEARANCE
                   for xy in (base[:2], view)]
        for table in self.coordinates.get("tables", []):
            margins.extend(self.rect_distance(xy, table) - self.BASE_RADIUS
                           for xy in (base[:2], view))
        for obj in others:
            margins.append(math.hypot(obj[0] - base[0], obj[1] - base[1]) - self.BASE_CLEARANCE)
            margins.append(segment_distance(obj, base[:2], (point.x, point.y))
                           - self.ARM_CORRIDOR)
        return min(margins)

    def hand_margin(self, target, plan):
        """62401543 池畑一志"""
        # つかむときに手が動く範囲と、同じ面に載った他の物体や机の天板との余裕[m]を返す。
        # 負なら当たる。上からつかむときは、物体の真上に下ろした手の指が開く範囲を調べる。
        # 手前からつかむときは、水平な手が物体まで進む通り道を調べる
        style, yaw_deg, roll, approach = plan
        point = target["point"]
        head, left = self.heading(yaw_deg)
        surface = self.surface_of(point.z)
        own = self.own_radius(target)
        margins = [1.0]
        # 手を下に向けたとき、指は台車の向きから(90度 - 手首の回転角)の方向に開く
        open_angle = math.radians(yaw_deg) + math.pi / 2 - roll
        axis_open = (math.cos(open_angle), math.sin(open_angle))
        axis_side = (math.sin(open_angle), -math.cos(open_angle))
        if style == "above":
            # 開いた指の範囲(長方形)の角が、物体より高い天板の下に入れば当たる
            corners = [(point.x + s_open * self.FINGER_SPAN * axis_open[0]
                        + s_side * self.FINGER_WIDTH * axis_side[0],
                        point.y + s_open * self.FINGER_SPAN * axis_open[1]
                        + s_side * self.FINGER_WIDTH * axis_side[1])
                       for s_open in (-1, 1) for s_side in (-1, 1)]
            hand_area = corners + [(point.x, point.y)]
        else:
            # 物体まで進んだ水平な手の指先
            reach = self.PALM_TO_CLOSED_TIP - self.PALM_BEHIND_OBJECT
            hand_area = [(point.x + reach * head[0], point.y + reach * head[1])]
        under = self.table_over(point)
        if style == "above":
            palm_z = max(point.z + self.PALM_ABOVE_OBJECT, self.surface_limit(point.z))
            hand_top = palm_z + self.hover_height(point) + self.WRIST_ABOVE_PALM
        else:
            # 水平な手は腕の厚さの分だけ手のひらより上に出る
            hand_top = (max(point.z, self.surface_of(point.z) + self.FRONT_PALM_ABOVE_SURFACE)
                        + self.ARM_RADIUS)
        for table in self.coordinates.get("tables", []):
            if table is under or point.z >= table["top"] - self.TABLE_MARGIN:
                continue
            if hand_top < table["top"] - self.TABLE_MARGIN:
                continue  # 手と手首が天板より低ければ、天板の下に入っても当たらない
            margins.extend(self.rect_distance(xy, table) for xy in hand_area)
        if (style == "above" and under is not None
                and hand_top >= under["top"] - self.TABLE_MARGIN):
            margins.append(-1.0)
        if style == "above" and self.table_backoff(point, yaw_deg) is None:
            margins.append(-1.0)  # 手を天板の外で構えられない
        for obj in self.seen_objects:
            d_x, d_y = obj[0] - point.x, obj[1] - point.y
            if math.hypot(d_x, d_y) < own or self.surface_of(obj[2]) != surface:
                continue
            along_head = d_x * head[0] + d_y * head[1]
            along_left = d_x * left[0] + d_y * left[1]
            if style == "above":
                opening = d_x * axis_open[0] + d_y * axis_open[1]
                side = d_x * axis_side[0] + d_y * axis_side[1]
                margins.append(max(abs(opening) - self.FINGER_SPAN,
                                   abs(side) - self.FINGER_WIDTH))
            else:
                start = -(self.PALM_BEHIND_OBJECT + approach + self.PALM_TO_CLOSED_TIP)
                margins.append(max(abs(along_left) - self.FINGER_SPAN,
                                   start - along_head, along_head - self.PALM_BEHIND_OBJECT))
        margins.append(self.sweep_margin(target, plan))
        return min(margins)

    def sweep_volumes(self, target, plan):
        """62401543 池畑一志"""
        # つかむ間に手と腕が動く空間を返す。手は[(x, y)の線分, 下端, 上端]、
        # 腕は台車から手までの[(x, y, 下端, 上端)の点列]で表す
        style, yaw_deg, _, approach = plan
        point = target["point"]
        head, _ = self.heading(yaw_deg)
        if style == "above":
            palm_low = max(point.z + self.PALM_ABOVE_OBJECT, self.surface_limit(point.z))
            palm_high = palm_low + self.hover_height(point)
            hand = ([(point.x, point.y), (point.x, point.y)],
                    palm_low - self.PALM_TO_CLOSED_TIP, palm_high + self.WRIST_ABOVE_PALM)
            states = [(max(0.0, palm - self.PALM_HEIGHT_DOWN) + self.SHOULDER_HEIGHT,
                       palm + self.ARM_ABOVE_PALM) for palm in (palm_low, palm_high)]
            palm_xy = (point.x, point.y)
        else:
            palm_z = max(point.z, self.surface_of(point.z) + self.FRONT_PALM_ABOVE_SURFACE)
            palm_xy = (point.x - self.PALM_BEHIND_OBJECT * head[0],
                       point.y - self.PALM_BEHIND_OBJECT * head[1])
            start = (palm_xy[0] - approach * head[0], palm_xy[1] - approach * head[1])
            hand = ([start, palm_xy], palm_z - self.ARM_RADIUS, palm_z + self.ARM_RADIUS)
            lift = max(0.0, palm_z - self.PALM_HEIGHT_FORWARD)
            states = [(lift + self.SHOULDER_HEIGHT, palm_z)]
        base = self.base_for_palm(palm_xy, yaw_deg, self.grasp_reach(target, plan))
        arm = []
        # 手先側の2割は手首と手で、手の空間として調べる
        for step in range(9):
            ratio = step / 10.0
            heights = [shoulder + (wrist - shoulder) * ratio for shoulder, wrist in states]
            arm.append((base[0] + (palm_xy[0] - base[0]) * ratio,
                        base[1] + (palm_xy[1] - base[1]) * ratio,
                        min(heights) - self.ARM_RADIUS, max(heights) + self.ARM_RADIUS))
        return hand, arm

    def sweep_margin(self, target, plan):
        """62401543 池畑一志"""
        # つかむ間に手と腕が動く空間と、見えた物体(机の上も含む)や机の天板との余裕[m]を返す。
        # 負なら当たる。同じ面の低い物体は、指が開く範囲の判定(hand_margin)に任せる
        point = target["point"]
        (segment, hand_low, hand_high), arm = self.sweep_volumes(target, plan)
        palm_low = hand_low + self.PALM_TO_CLOSED_TIP if plan[0] == "above" else hand_low
        own = self.own_radius(target)
        margins = [1.0]
        for obj in self.seen_objects:
            if math.hypot(obj[0] - point.x, obj[1] - point.y) < own:
                continue
            low, high = obj[2] - self.OBJECT_RADIUS, obj[2] + self.OBJECT_RADIUS
            if high > palm_low and low < hand_high:
                margins.append(segment_distance(obj, segment[0], segment[1])
                               - self.HAND_RADIUS - self.OBJECT_RADIUS)
            for arm_x, arm_y, arm_low, arm_high in arm:
                if high > arm_low and low < arm_high:
                    margins.append(math.hypot(obj[0] - arm_x, obj[1] - arm_y)
                                   - self.ARM_RADIUS - self.OBJECT_RADIUS)
        support = self.surface_of(point.z)
        for table in self.coordinates.get("tables", []):
            if abs(table["top"] - support) < 0.05:
                continue  # 物体が載っている机。机の上の物体は上で調べた
            low, high = table["top"] - 0.03, table["top"] + self.TABLE_ITEM_HEIGHT
            if hand_high > low and hand_low < high:
                margins.extend(self.rect_distance(xy, table) - self.HAND_RADIUS
                               for xy in segment)
            for arm_x, arm_y, arm_low, arm_high in arm:
                if arm_high > low and arm_low < high:
                    margins.append(self.rect_distance((arm_x, arm_y), table) - self.ARM_RADIUS)
        return min(margins)

    def grasp_plans(self, target, prefer_yaw):
        """62401543 池畑一志"""
        # つかみ方(方法, 台車の向き, 手首の回転, 手を差し入れる距離)の候補を試す順に返す。
        # 台車を動かさずに済むよう、同じ向きで手首の回転や方法を変える候補を先にする。
        # 天板の下の物体は手前からだけ、机の上の物体は正面からだけ
        point = target["point"]
        floor = point.z < self.NO_APPROACH_Z and self.planner is not None
        yaws = sorted(self.GRASP_YAWS if floor else (90,),
                      key=lambda yaw: (abs(math.degrees(math.atan2(
                          math.sin(math.radians(yaw - prefer_yaw)),
                          math.cos(math.radians(yaw - prefer_yaw))))), abs(yaw - 90)))
        roll = 0.0 if target.get("bbox") is None else self.wrist_roll_for(target["bbox"])
        near_table = any(self.rect_distance((point.x, point.y), table)
                         < self.FINGER_SPAN + self.TABLE_MARGIN
                         and point.z < table["top"] - self.TABLE_MARGIN
                         for table in self.coordinates.get("tables", []))
        on_table = point.z >= self.NO_APPROACH_Z
        if target["label"] in self.CARRY_DOWN_LABELS:
            styles = ("above",)  # 水平な手で差し入れると倒してしまう
        elif target["label"] in self.FRONT_ONLY_LABELS:
            styles = ("front",)
        elif self.table_over(point) is not None:
            styles = ("front",)
        elif (on_table and self.height_above_surface(point) >= self.TABLE_FRONT_MIN_HEIGHT
              and 0.0 < self.object_width(target) < self.TABLE_FRONT_MAX_WIDTH):
            # 机の上の細くて背のある物体は、上から指先で挟むと運ぶ途中で落ちるので、
            # 水平な手で横から差し入れて指の奥で挟む方法を先に試す
            styles = ("front", "above")
        elif self.is_front_grasp(target):
            styles = ("front", "above")
        elif (near_table and self.height_above_surface(point) >= self.FRONT_MIN_HEIGHT
              and self.object_width(target) < self.FRONT_MAX_WIDTH):
            # 天板の縁の近くでは、指が天板に当たらないよう手前から水平につかむ方法も試す
            styles = ("above", "front")
        else:
            styles = ("above",)
        plans = []
        for yaw in yaws:
            for style in styles:
                if style == "above":
                    if target.get("axis") is not None:
                        # 長い方向に指を閉じると物体が手に掛かるので、直角に閉じる向きだけにする。
                        # 失敗したら同じ向きで、見直した位置でつかみ直す
                        plan = ("above", float(yaw), self.axis_roll(target["axis"], yaw), 0.0)
                        plans.extend((plan, plan))
                        continue
                    plans.extend(("above", float(yaw), value, 0.0)
                                 for value in (roll, math.pi / 2 - roll))
                    continue
                approach = self.front_approach(target, yaw)
                if approach is not None:
                    plans.append(("front", float(yaw), 0.0, approach))
        return plans

    def choose_grasp_plans(self, target, prefer_yaw=90.0, count=1):
        """62401543 池畑一志"""
        # 台車、腕、手が壁や机や他の物体に当たらないつかみ方を、試す順に最大count個返す。
        # 当たらないものが足りなければ、余裕がPLAN_TOLERANCE以上のものを余裕の大きい順に足す。
        # どれも当たるなら空のリストを返す(つかまずに飛ばす)
        floor = target["point"].z < self.NO_APPROACH_Z and self.planner is not None
        safe, risky = [], []
        for plan in self.grasp_plans(target, prefer_yaw):
            margin = self.hand_margin(target, plan)
            if floor:
                margin = min(margin, self.base_margin(target, plan))
            if margin >= 0.0:
                safe.append(plan)
                if len(safe) >= count:
                    break
            elif margin >= self.PLAN_TOLERANCE:
                risky.append((margin, plan))
        plans = safe + [plan for _, plan in sorted(risky, reverse=True)]
        plans = plans[:count]
        if plans:
            rospy.loginfo("grasp plans: %s", plans)
        else:
            rospy.logwarn("no grasp plan without collision")
        return plans

    def choose_grasp_plan(self, target, prefer_yaw=90.0):
        """62401543 池畑一志"""
        # 最も良いつかみ方を1つ返す。どれも当たるならNoneを返す
        plans = self.choose_grasp_plans(target, prefer_yaw)
        return plans[0] if plans else None

    def object_ends(self, bbox):
        """62401543 池畑一志"""
        # 検出枠を長い方向に分け、両端の部分の3次元座標を返す。測れなければNoneを返す
        ends = []
        for side in (0, 1):
            part = copy.deepcopy(bbox)
            if bbox.w >= bbox.h:
                part.w = max(1, int(bbox.w * self.END_RATIO))
                part.x = bbox.x if side == 0 else bbox.x + bbox.w - part.w
            else:
                part.h = max(1, int(bbox.h * self.END_RATIO))
                part.y = bbox.y if side == 0 else bbox.y + bbox.h - part.h
            point = self.get_grasp_coordinate(part)
            if point is None:
                return None
            ends.append(point)
        return ends

    def stand_up_plan(self, target):
        """62401543 池畑一志"""
        # 寝ている背の高い物体を縦に起こして置くつかみ方を返す。対象でなければNoneを返す。
        # 上からつかんだ手を水平にすると、台車に近い側の端が下になるので、底(低い側の端)の側に
        # 台車を置き、頭(高い側の端。windexはスプレーの頭)の方向を向いてつかむ
        point = target["point"]
        if (target["label"] not in self.STAND_UP_LABELS or target.get("bbox") is None
                or self.height_above_surface(point) >= self.SIDE_GRASP_MIN_Z):
            return None
        ends = self.object_ends(target["bbox"])
        if ends is None:
            return None
        bottom, top = sorted(ends, key=lambda end: end.z)
        length = math.hypot(top.x - bottom.x, top.y - bottom.y) / (1.0 - self.END_RATIO)
        orders = [(bottom, top), (top, bottom)]
        if top.z - bottom.z < self.END_HEIGHT_GAP:
            # 両端の高さの差が小さいと頭と底を見分けられないので、正面に近い向きを先にする
            def off_front(order):
                diff = math.atan2(order[1].y - order[0].y, order[1].x - order[0].x) - math.pi / 2
                return abs(math.atan2(math.sin(diff), math.cos(diff)))
            orders.sort(key=off_front)
        rospy.loginfo("stand up the %s: ends (%.2f, %.2f, %.3f) (%.2f, %.2f, %.3f)",
                      target["label"], bottom.x, bottom.y, bottom.z, top.x, top.y, top.z)
        for near, far in orders:
            # 台車に近い側(near)の端が下になる
            yaw = math.degrees(math.atan2(far.y - near.y, far.x - near.x))
            plan = ("above", round(yaw, 1), 0.0, 0.0)
            margin = self.hand_margin(target, plan)
            if self.planner is not None:
                margin = min(margin, self.base_margin(target, plan))
            if margin >= self.PLAN_TOLERANCE:
                return plan, length
            rospy.logwarn("cannot stand up the %s from %.0f deg (margin %.2f)",
                          target["label"], yaw, margin)
        return None

    def exec_grasp_plan(self, target, plan):
        """62401543 池畑一志"""
        # task1専用:選んだつかみ方で物体をつかみ、保持できたかを返す
        style, yaw_deg, roll, approach = plan
        if style == "front":
            return self.grasp_from_front(target["point"], yaw_deg, approach)
        self.carry_down = target["label"] in self.CARRY_DOWN_LABELS
        return self.grasp_from_above(target["point"], roll, yaw_deg)

    def find_shelf_target(self, target_obj):
        """62414107 長澤桃峨"""
        # 頭の角度を変えながら棚を見て、把持する物体のbboxを返す。
        # 指定物体が見えなければ、誤配達でも得点になるので見えた物体を代わりに返す
        substitute = None
        for tilt in self.SHELF_TILTS:
            whole_body.move_head_tilt(tilt)
            rospy.sleep(1.0)
            bboxes = self.get_latest_detection().bboxes
            bbox = self.get_most_graspable_bboxes_by_label(bboxes, target_obj)
            if bbox is not None:
                return bbox
            substitute = substitute or self.get_most_graspable_bbox(bboxes)
        rospy.logwarn("[%s] is not found. Use a substitute object.", target_obj)
        return substitute

    def grasp_from_shelf(self, target_obj):
        """62414107 長澤桃峨"""
        # 棚の前で物体を掴む。保持できたらTrueを返す
        for trial in range(self.SHELF_GRASP_RETRY):
            rospy.loginfo("shelf grasp trial %d", trial + 1)
            self.goto_name("shelf")
            self.change_pose("look_at_shelf")
            bbox = self.find_shelf_target(target_obj)
            grasp_pos = None if bbox is None else self.get_grasp_coordinate(bbox)
            if grasp_pos is None:
                continue
            if self.grasp_from_front(grasp_pos):
                return True
            self.set_gripper(1)
        return False

    def deliver_to_target(self, target_obj, target_person):
        """
        棚で取得したものを人に渡す。
        """
        rospy.loginfo("target_obj: %s  target_person: %s", target_obj, target_person)
        if not self.grasp_from_shelf(target_obj):
            rospy.logwarn("Cannot grasp any object. task2b is aborted.")
            return

        # target_personの前に持っていく
        self.change_pose("look_at_near_floor")
        self.goto_name(self.PERSON_WAYPOINTS[target_person])
        self.change_pose("deliver_to_human")
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    def detect_floor_obstacles(self, known):
        """62411499 髙野幹太"""
        # カメラに写った床上の物体のmap座標をknownに追加する
        area = self.OBSTACLE_AREA
        for bbox in self.get_latest_detection().bboxes:
            point = self.get_grasp_coordinate(bbox)
            if point is None:
                continue
            rospy.loginfo("seen %s at (%.2f, %.2f, %.2f)", bbox.label, point.x, point.y, point.z)
            in_area = (area["x"][0] <= point.x <= area["x"][1]
                       and area["y"][0] <= point.y <= area["y"][1]
                       and point.z <= area["z_max"])
            is_new = all(math.hypot(point.x - x, point.y - y) > 0.15 for x, y in known)
            if in_area and is_new:
                known.append((point.x, point.y))
        rospy.loginfo("known obstacles: %s", known)

    def get_base_pose(self):
        """62411499 髙野幹太"""
        # 台車のmap座標(x, y, yaw[rad])を返す。取得できなければNoneを返す
        transform = self.get_relative_coordinate("map", "base_footprint")
        if transform is None:
            return None
        rot = transform.rotation
        yaw = math.atan2(2 * (rot.w * rot.z + rot.x * rot.y),
                         1 - 2 * (rot.y ** 2 + rot.z ** 2))
        return transform.translation.x, transform.translation.y, yaw

    def follow_path(self, path, yaw_deg=90.0, tolerance=None):
        """62411499 髙野幹太"""
        # 経路上の点を順に速度指令で追従する。向きを保ったまま全方向移動台車で平行移動する。
        # move_baseは床の小物を避けられず軌跡も膨らむため、計画した経路どおりに走らせる
        rate = rospy.Rate(20)
        deadline = rospy.Time.now() + rospy.Duration(self.FOLLOW_TIMEOUT)
        limit = tolerance or self.FOLLOW_TOLERANCE
        for index, target in enumerate(path[1:], start=2):
            is_last = index == len(path)
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                pose = self.get_base_pose()
                if pose is None:
                    break
                err_x, err_y = target[0] - pose[0], target[1] - pose[1]
                dist = math.hypot(err_x, err_y)
                diff = math.radians(yaw_deg) - pose[2]
                yaw_err = math.degrees(math.atan2(math.sin(diff), math.cos(diff)))
                yaw_ok = not is_last or tolerance is None or abs(yaw_err) < self.ALIGN_YAW_TOLERANCE
                # 途中の点は止まらずに通過し、最後の点だけ減速して合わせる
                point_limit = limit if is_last else max(limit, self.FOLLOW_PASS_TOLERANCE)
                if dist < point_limit and yaw_ok:
                    break
                speed = 0.0 if dist < limit else (
                    min(self.FOLLOW_SPEED, 0.05 + dist) if is_last else self.FOLLOW_SPEED)
                vel_x, vel_y = speed * err_x / max(dist, 1e-3), speed * err_y / max(dist, 1e-3)
                omni_base.cmd_vel(math.cos(pose[2]) * vel_x + math.sin(pose[2]) * vel_y,
                                  -math.sin(pose[2]) * vel_x + math.cos(pose[2]) * vel_y,
                                  max(-30.0, min(30.0, 3.0 * yaw_err)))
                rate.sleep()
        omni_base.cmd_vel(0.0, 0.0, 0.0)

    def execute_avoid_blocks(self):
        """
        床の障害物を記憶しながら、少し進むごとに再検出して経路を計画し直す
        """
        planner = GridPlanner(rospy.wait_for_message("/map", OccupancyGrid))
        goal = self.coordinates["positions"]["go_throw_2a"][:2]
        obstacles = []
        while not rospy.is_shutdown():
            self.detect_floor_obstacles(obstacles)
            pose = self.get_base_pose()
            path = planner.plan(pose[:2], goal, obstacles)
            if not path:
                rospy.logwarn("No path to the goal of task2a.")
                return
            rospy.loginfo("planned path: %d points, clearance %.2f",
                          len(path), path_clearance(path, obstacles))
            part = cut_path(path, self.REPLAN_INTERVAL)
            self.follow_path(part)
            if len(part) == len(path):
                return

    def get_category(self, label):
        """62414107 長澤桃峨"""
        # 検出ラベルからTask 1のカテゴリ名を返す。どれにも属さなければotherを返す。
        # label_depositsに書いたラベル(cup)は、そこに書いた収納先のカテゴリにする
        override = self.coordinates.get("label_deposits", {}).get(label)
        if override is not None:
            return override
        for category, labels in self.coordinates["categories"].items():
            if label in labels:
                return category
        return "other"

    def get_odom_joints(self):
        """62411499 髙野幹太"""
        # omni_base_controllerが制御している台車の姿勢[odom_x, odom_y, odom_t]を返す
        try:
            state = rospy.wait_for_message(
                "/hsrb/omni_base_controller/state", JointTrajectoryControllerState, 2.0)
        except rospy.ROSException:
            rospy.logwarn("omni_base_controller state is not available")
            return None
        odom = dict(zip(state.joint_names, state.actual.positions))
        return [odom["odom_x"], odom["odom_y"], odom["odom_t"]]

    def send_base_goal(self, goal, duration):
        """62411499 髙野幹太"""
        # omni_base_controllerに台車の目標姿勢(odom座標)を送り、動き終わるまで待つ
        point = JointTrajectoryPoint(positions=goal, velocities=[0.0] * 3,
                                     time_from_start=rospy.Duration(duration))
        traj = JointTrajectory(joint_names=["odom_x", "odom_y", "odom_t"], points=[point])
        self.base_client.send_goal(FollowJointTrajectoryGoal(trajectory=traj))
        self.base_client.wait_for_result(rospy.Duration(duration + 3.0))

    def hold_base(self):
        """62411499 髙野幹太"""
        # 台車の目標姿勢を今の姿勢に置き直す。
        # 台車の制御は数度の誤差を残したまま止まり、腕を動かした振動などで急に回って誤差を
        # 取り戻すので、目標と今の姿勢を一致させて、つかむ間に台車が動かないようにする
        rospy.sleep(0.5)
        current = self.get_odom_joints()
        if current is not None:
            self.send_base_goal(current, 0.3)

    def shift_base(self, d_x, d_y, d_yaw, speed=None):
        """62411499 髙野幹太"""
        # 台車をmap座標でd_x, d_y[m]平行移動し、d_yaw[rad]回す。
        # 速度指令で止めると、制御器が少しずれた姿勢を保持し直して数度回ってしまうので、
        # omni_base_controllerに目標姿勢(odom座標)を直接送る
        pose = self.get_base_pose()
        odom = self.get_odom_joints()
        if pose is None or odom is None:
            return False
        rot = odom[2] - pose[2]  # map座標からodom座標への回転
        goal = [odom[0] + math.cos(rot) * d_x - math.sin(rot) * d_y,
                odom[1] + math.sin(rot) * d_x + math.cos(rot) * d_y,
                odom[2] + d_yaw]
        distance = math.hypot(d_x, d_y)
        shortest = self.FINE_MOVE_DURATION if distance < self.FINE_MOVE else 1.0
        duration = max(shortest, distance / (speed or self.SHIFT_SPEED),
                       abs(d_yaw) / self.SHIFT_YAW_SPEED)
        self.send_base_goal(goal, duration)
        self.hold_base()
        return True

    def align_hand_over(self, target, yaw_deg, tolerance=None, trials=None, rotate=True,
                        verify=False):
        """62411499 髙野幹太"""
        # 手のひらのTFを実測し、手のひらがtargetの真上に来て台車がyaw_degを向くように台車を動かす。
        # 回転すると手のひらも振れるので、回転後の手のひらの位置から台車の移動量を求める。
        # rotateがFalseなら向きは直さず、平行移動だけで手のひらを合わせる。
        # verifyがTrueなら最後は動かずに測り直し、許容誤差に入らなければFalseを返す
        self.hold_base()
        tolerance = tolerance or self.ALIGN_TOLERANCE * 1.5
        trials = trials or self.ALIGN_TRIALS
        last_yaw = None  # 直前に回転を指令したときの向き
        deadband = self.ALIGN_YAW_DEADBAND
        for trial in range(trials):
            palm = self.get_relative_coordinate("map", "hand_palm_link")
            pose = self.get_base_pose()
            if palm is None or pose is None:
                return False
            if (verify and last_yaw is not None
                    and abs(math.degrees(pose[2] - last_yaw)) < self.ALIGN_YAW_STUCK):
                # 回転を指令しても向きが変わらなければ(台車の不感帯)、回転の分だけ横移動が
                # 打ち消されて手が動かなくなるので、以降は大きくずれたときだけ回す
                deadband = self.ALIGN_YAW_LARGE
            diff = math.radians(yaw_deg) - pose[2]
            d_yaw = math.atan2(math.sin(diff), math.cos(diff))
            # 小さな回転は台車が追従しないので、向きの誤差は許容して手のひらの位置だけ合わせる
            if abs(math.degrees(d_yaw)) < deadband or not rotate:
                d_yaw = 0.0
            rel_x, rel_y = palm.translation.x - pose[0], palm.translation.y - pose[1]
            new_x = math.cos(d_yaw) * rel_x - math.sin(d_yaw) * rel_y
            new_y = math.sin(d_yaw) * rel_x + math.cos(d_yaw) * rel_y
            d_x, d_y = target[0] - pose[0] - new_x, target[1] - pose[1] - new_y
            rospy.loginfo("align hand: error (%.3f, %.3f) m, %.1f deg",
                          target[0] - palm.translation.x, target[1] - palm.translation.y,
                          math.degrees(d_yaw))
            if (math.hypot(target[0] - palm.translation.x, target[1] - palm.translation.y)
                    < tolerance and d_yaw == 0.0):
                return True
            if verify and trial == trials - 1:
                return False
            last_yaw = pose[2] if d_yaw != 0.0 else None
            self.shift_base(d_x, d_y, d_yaw)
        return True

    def deposit_joints(self, entry, style):
        """62414107 長澤桃峨"""
        # 収納先の縁のすぐ上で腕を水平に伸ばす関節値と、手のひらを合わせる位置を返す。
        # 上からつかんだ物体は手を下に向けたまま、床にあったときの向きで置く(手のひらの真下に
        # 落ちるので位置が正確で、寝ていた物体は寝たまま置ける)。
        # 手前からつかんだ立った物体と縦に起こす物体(stand)は手を水平にし、
        # 下端が縁のすぐ上に来る高さで放す
        pose = self.poses[entry["pose"]]
        # 縁の高さ。設定に無ければ収納用の姿勢(縁の0.10 m上に手のひら)から求める
        rim = entry.get("rim", pose["arm_lift_joint"] + self.PALM_HEIGHT_FORWARD
                        - self.RIM_BELOW_FORWARD_PALM)
        hang = self.hang
        if style == "above":
            joints, forward = self.arm_joints(rim + self.DROP_CLEARANCE + hang, hand_down=True)
            return joints, tuple(entry.get("drop", entry["target"])), forward
        # 手前からつかんだ物体も縦に起こした物体も、下端が縁のすぐ上に来る高さで放す
        joints, forward = self.arm_joints(rim + self.DROP_CLEARANCE + hang, hand_down=False)
        head, _ = self.heading(entry["base"][2])
        # 物体は指先にあるので、手のひらは落としたい位置より手前に合わせる
        return joints, (entry["target"][0] - self.PALM_TO_CLOSED_TIP * head[0],
                        entry["target"][1] - self.PALM_TO_CLOSED_TIP * head[1]), forward

    def align_base_for_palm(self, palm, yaw_deg, forward, trials=3, tolerance=0.01):
        """62414107 長澤桃峨"""
        # 腕をたたんだまま、腕を伸ばしたときの手のひらがpalm(map座標)に来るよう台車を合わせる。
        # 台車が追従しない小さな向きのずれ(2度で手のひらが2 cm横にずれる)は残るので、
        # 手のひらの位置は今の向きで予測する(台車のTFで測る)
        for _ in range(trials):
            pose = self.get_base_pose()
            if pose is None:
                return
            diff = math.radians(yaw_deg) - pose[2]
            d_yaw = math.atan2(math.sin(diff), math.cos(diff))
            if abs(math.degrees(d_yaw)) < self.ALIGN_YAW_DEADBAND:
                d_yaw = 0.0
            base = self.base_for_palm(palm, math.degrees(pose[2] + d_yaw), forward)
            d_x, d_y = base[0] - pose[0], base[1] - pose[1]
            rospy.loginfo("align base: error (%.3f, %.3f) m, %.1f deg",
                          d_x, d_y, math.degrees(d_yaw))
            if math.hypot(d_x, d_y) < tolerance and d_yaw == 0.0:
                return
            self.shift_base(d_x, d_y, d_yaw)

    def slot_entry(self, entry, label):
        """62414107 長澤桃峨"""
        # 収納先のslot_labelsに書いた物体(pitcher、cup、toy_airplane)は、その置き場所(slots)を
        # 落とす位置にした収納先を返す。それ以外の物体は、置き場所と分けた元の収納先
        # (target、drop)を返す
        plain = {key: value for key, value in entry.items()
                 if key not in ("slots", "slot_labels")}
        slots = entry.get("slots")
        index = entry.get("slot_labels", {}).get(label)
        if not slots or index is None:
            return plain
        return dict(entry, target=list(slots[index]), drop=list(slots[index]))

    def deposit_object(self, label, style="above"):
        """62414107 長澤桃峨"""
        # 掴んでいる物体をカテゴリの収納先の縁のすぐ上まで運び、水平に腕を伸ばして落とす。
        # 手を箱の中に入れないので、箱や引き出しに腕が引っ掛からない
        category = self.get_category(label)
        entries = self.coordinates["deposits"][category]
        if any("drawer" in entry for entry in entries):
            # 引き出しの収納先(drawerに名前を書いたもの)は順に回さず、開けられた最初の
            # 引き出しを使う(trofast_3を開けると真下のtrofast_2には手が入らないので、
            # toolはtrofast_3、trofast_3を開けられなければtrofast_2)
            entries = [entry for entry in entries
                       if entry.get("drawer") not in self.closed_drawers][:1]
            if not entries:
                category = "other"
        if category in self.closed_categories:
            category = "other"
        if category == "other" and self.get_category(label) != "other":
            # 引き出しが開いていなければ、床に落とすより箱に入れる
            rospy.logwarn("the drawer for %s is closed. put it in the bin for others",
                          self.get_category(label))
            entries = self.coordinates["deposits"][category]
        count = self.deposit_count.get(category, 0)
        self.deposit_count[category] = count + 1
        entry = self.slot_entry(entries[count % len(entries)], label)
        rospy.loginfo("deposit %s (%s) at %s", label, category, entry["target"])

        joints, palm, forward = self.deposit_joints(entry, style)
        self.goto_pos(entry["base"])
        slotted = bool(entry.get("slots"))
        if slotted:
            # 隣の置き場所の物体(Containerのwindex)の上を、伸ばした腕と物体で横切らないよう、
            # 腕をたたんだまま台車を合わせてから腕を伸ばす
            self.align_base_for_palm(palm, entry["base"][2], forward)
        if self.carry_down and style == "above":
            # 手を下に向けたまま運んできたので、手首を回さずに縁の上まで下ろす
            # (手首の回転はつかんだときのまま。指定しない関節は今の値で送られる)
            self.move_arm_slowly({name: value for name, value in joints.items()
                                  if name != "wrist_roll_joint"})
        else:
            self.reach_out(joints)
        self.carry_down = False
        if slotted:
            # 回すと手のひらが左右に振れるので、平行移動だけで細かく合わせる
            self.align_hand_over(palm, entry["base"][2], tolerance=self.SLOT_ALIGN_TOLERANCE,
                                 trials=self.SLOT_ALIGN_TRIALS, rotate=False)
        else:
            self.align_hand_over(palm, entry["base"][2])
        self.release_object()
        # 手が物体や縁に引っ掛からないよう、ゆっくり持ち上げてからゆっくり真っすぐ下がる
        lift = joints["arm_lift_joint"] + self.RELEASE_RAISE
        self.move_arm_slowly(dict(joints, arm_lift_joint=min(lift, self.MAX_LIFT)))
        pose = self.get_base_pose()
        if pose is not None:
            self.shift_base(entry["base"][0] - pose[0], entry["base"][1] - pose[1], 0.0,
                            speed=self.RETREAT_SPEED)
        self.fold_arm()

    def open_drawer(self, drawer):
        """62414107 長澤桃峨"""
        # 取っ手の軸を指で挟み、腕は固定したまま台車を後退させて引き出しを開ける。
        # 関節姿勢を固定して手のひらのTFで位置合わせするので、全身IKのように台車の向きが変わらない
        handle_x, handle_y = drawer["handle"][:2]
        palm_x = handle_x - self.PINCH_X_OFFSET
        palm_y = handle_y - self.HANDLE_GAP + self.PALM_TO_CLOSED_TIP
        reference = drawer.get("reference")
        if reference and reference["name"] in self.handle_palms:
            # 同じ形の引き出しを掴めたときの手のひらの位置から、同じ相対位置で掴む。
            # map座標は自己位置推定の誤差で数cmずれるが、近くで続けて測れば誤差はほぼ共通
            ref_x, ref_y = self.handle_palms[reference["name"]]
            palm_x = ref_x + reference["offset"][0]
            palm_y = ref_y + reference["offset"][1]
            handle_x = palm_x + self.PINCH_X_OFFSET
            handle_y = palm_y + self.HANDLE_GAP - self.PALM_TO_CLOSED_TIP
            rospy.loginfo("grasp the handle of %s at the palm (%.3f, %.3f) relative to %s",
                          drawer.get("name"), palm_x, palm_y, reference["name"])

        self.change_pose("all_neutral")
        self.goto_pos([handle_x - 0.1, palm_y + self.DRAWER_APPROACH + 0.6, -90])
        self.set_gripper(1, wait=1.0)
        self.reach_out(self.poses[drawer["pose"]])
        self.align_hand_over((palm_x, palm_y + self.DRAWER_APPROACH), -90.0)
        grasped = False
        pose = self.poses[drawer["pose"]]
        # fine_alignの引き出しは、台車が動くたびに向きが変わって0.6 m先の手のひらが振れるので、
        # 最後に必ず測り直し、許容誤差に入るまで合わせ直す(以前は回転した直後に測らず閉じ、
        # 横にずれたまま閉じていた。向きを直さないと、ずれが溜まって台車が斜めになる)(trofast_2)
        fine = drawer.get("fine_align", False)
        if fine:
            # 前進して指先が取っ手の板の横を通るよう、手前で左右を合わせておく
            self.align_hand_over((palm_x, palm_y + self.DRAWER_APPROACH), -90.0,
                                 tolerance=self.HANDLE_FINE_TOLERANCE,
                                 trials=self.HANDLE_FINE_TRIALS, verify=True)
        for trial, (shift, lower) in enumerate(self.HANDLE_STRATEGIES):
            if trial > 0 and lower != 0.0:
                self.move_arm_slowly(dict(pose, arm_lift_joint=max(
                    0.0, pose["arm_lift_joint"] + lower)), steps=1)
            # 開いた指先と取っ手の板の左右の隙間は2.5 cmしかないので、精度を上げて合わせ、
            # 伸ばした腕の揺れが収まってから閉じる
            if not fine:
                self.align_hand_over((palm_x + shift, palm_y), -90.0,
                                     tolerance=self.HANDLE_ALIGN_TOLERANCE,
                                     trials=self.HANDLE_ALIGN_TRIALS)
            elif not self.align_hand_over((palm_x + shift, palm_y), -90.0,
                                          tolerance=self.HANDLE_FINE_TOLERANCE,
                                          trials=self.HANDLE_FINE_TRIALS, verify=True):
                rospy.logwarn("the hand is not aligned with the handle")
            rospy.sleep(self.SETTLE_WAIT)
            closing = self.get_relative_coordinate("map", "hand_palm_link")
            self.set_gripper(0)
            grasped = self.is_holding()
            rospy.loginfo("drawer handle grasped: %s (trial %d)", grasped, trial + 1)
            if grasped:
                if closing is not None:
                    self.handle_palms[drawer.get("name")] = (closing.translation.x,
                                                             closing.translation.y)
                break
            # 掴み損ねたら手を開いて少し離れ、合わせ直してから掴み直す
            self.set_gripper(1, wait=1.0)
            self.shift_base(0.0, self.HANDLE_RETRY_BACK, 0.0)
        if not grasped:
            rospy.logwarn("cannot open the drawer %s for %s", drawer.get("name"),
                          drawer.get("category"))
            self.closed_drawers.add(drawer.get("name"))
            # この引き出しを収納先にしているカテゴリは、すべて別の箱に回す。
            # 収納先にdrawerを書いたカテゴリは、deposit_objectで開いた別の引き出しを選ぶ
            # (trofast_2とtrofast_3はxが同じなので、xでは区別できない)
            for category, entries in self.coordinates["deposits"].items():
                if any("drawer" in entry for entry in entries):
                    continue
                if any(abs(entry["target"][0] - handle_x) < 0.1 for entry in entries):
                    self.closed_categories.add(category)
            self.shift_base(0.0, self.DRAWER_RELEASE_BACK, 0.0)
            self.fold_arm()
            return False

        # 取っ手を掴んだまま台車が回らないよう、目標姿勢を送って真っすぐ後退する
        self.shift_base(0.0, drawer["pull"], 0.0)
        if drawer.get("close_after", False):
            # 開けたら、取っ手を掴んだまま同じ距離だけ真っすぐ前進して閉じる(trofast_3、
            # ユーザー指示)。開いたままだと真下のtrofast_2に手が入らない。
            # 閉じた引き出しには入れないので、収納先の候補から外す
            rospy.sleep(self.SETTLE_WAIT)
            self.shift_base(0.0, -drawer["pull"], 0.0)
            self.closed_drawers.add(drawer.get("name"))
            rospy.loginfo("closed the drawer %s again", drawer.get("name"))
        self.set_gripper(1)
        # 指が取っ手の板に引っ掛からないよう、左右の中心に合わせてから下がる
        palm = self.get_relative_coordinate("map", "hand_palm_link").translation
        self.align_hand_over((handle_x, palm.y), -90.0)
        self.shift_base(0.0, self.DRAWER_RELEASE_BACK, 0.0)
        self.fold_arm()
        return True

    def select_grasp_target(self, failed):
        """62411499 髙野幹太"""
        # 視野内で把持できそうな物体を中央に近い順に調べ、ラベル・3次元座標・bboxを返す。
        # 失敗済みの位置と、把持できない位置の物体は除く。無ければNoneを返す
        bboxes = self.get_latest_detection().bboxes
        candidates = [bbox for bbox in bboxes if bbox.label not in self.IGNORE_LIST
                      and bbox.score >= self.MIN_DETECTION_SCORE]
        candidates.sort(key=self.calc_score_bbox, reverse=True)
        points = [(bbox, self.get_grasp_coordinate(bbox)) for bbox in candidates]
        points = [(bbox, point) for bbox, point in points if point is not None]
        for _, point in points:
            self.remember_object(point)
        for bbox, point in points:
            rospy.loginfo("candidate %s (%.0f%%) at (%.2f, %.2f, %.2f)",
                          bbox.label, bbox.score * 100, point.x, point.y, point.z)
            is_failed = any(math.hypot(point.x - x, point.y - y) < self.FAILED_RADIUS
                            for x, y in failed)
            out_of_room = point.y < 0.0 or point.y > 2.1 or point.x < -0.3 or point.x > 3.2
            if self.is_skipped(bbox.label, point):
                rospy.loginfo("skip the %s: it is too small to grasp", bbox.label)
                continue
            if not (is_failed or out_of_room or self.is_ungraspable(point)):
                return {"label": bbox.label, "point": point, "bbox": bbox}
        return None

    @staticmethod
    def is_ungraspable(point):
        """62411499 髙野幹太"""
        # 机の奥の床にあり腕が届かない位置ならTrueを返す
        graspable_y = 1.85
        desk_z = 0.35
        return point.y > graspable_y and point.z < desk_z

    def refine_target(self, target, pose, yaw_deg=90.0):
        """62411499 髙野幹太"""
        # 床の物体に、つかむときと同じ向きで近づいてから見直し、精度の良い結果を返す。
        # 見失ったら元の結果を返す
        point = target["point"]
        if point.z >= self.NO_APPROACH_Z:
            return target
        head, _ = self.heading(yaw_deg)
        self.goto_pos([point.x - self.APPROACH_DISTANCE * head[0],
                       point.y - self.APPROACH_DISTANCE * head[1], yaw_deg])
        self.change_pose(pose)
        rospy.sleep(1.0)
        # 同じ物体に複数のラベルが付くことがあるので、遠くから見た結果も含めて信頼度の高い順に
        # 並べる。位置は近くから見た結果を使い、ラベルは大きさが合うものを採用する
        labels = [(target["bbox"].score, target["label"], target["bbox"], point)]
        best = None
        for bbox in self.get_latest_detection().bboxes:
            new_point = self.get_grasp_coordinate(bbox)
            if new_point is None:
                continue
            self.remember_object(new_point)
            dist = math.hypot(new_point.x - point.x, new_point.y - point.y)
            if dist < self.SAME_OBJECT_RADIUS:
                labels.append((bbox.score, bbox.label, bbox, new_point))
                if best is None or bbox.score > best["bbox"].score:
                    best = {"label": bbox.label, "point": new_point, "bbox": bbox}
        if best is None:
            return target
        for _, label, bbox, seen in sorted(labels, key=lambda item: item[0], reverse=True):
            if self.label_fits_size(label, bbox, seen):
                best["label"] = label
                break
        else:
            # どのラベルも大きさが合わなければ分類しない(どの物体も数えられるBin Bに入れる)
            best["label"] = "unknown"
        return best

    def own_radius(self, target):
        """62411499 髙野幹太"""
        # つかむ物体自身の検出とみなす半径を返す。大きい物体(pitcherは幅約15 cm)はラベルごとに
        # 検出位置がずれるので、ずれた検出を隣の物体と誤って当たり判定しないよう広げる
        return min(self.OWN_RADIUS_MAX,
                   max(self.SAME_OBJECT_GAP, self.object_width(target) * 0.75))

    def object_width(self, target):
        """62411499 髙野幹太"""
        # 検出枠とカメラからの距離から、物体の短い方向の幅[m]を見積もる。分からなければ0を返す
        bbox, point = target.get("bbox"), target["point"]
        if bbox is None:
            return 0.0
        camera = self.get_relative_coordinate("map", self.CAMERA_FRAME)
        if camera is None:
            return 0.0
        cam = camera.translation
        dist = math.sqrt((point.x - cam.x) ** 2 + (point.y - cam.y) ** 2 + (point.z - cam.z) ** 2)
        return min(bbox.w, bbox.h) * dist / self.CAMERA_FOCAL

    def label_fits_size(self, label, bbox, point):
        """62411499 髙野幹太"""
        # 小さいはずのラベルの物体が、検出枠の大きさから見て大きすぎなければTrueを返す
        largest = self.LABEL_MAX_SIZE.get(label)
        smallest = self.LABEL_MIN_SIZE.get(label)
        camera = self.get_relative_coordinate("map", self.CAMERA_FRAME)
        if (largest is None and smallest is None) or camera is None:
            return True
        cam = camera.translation
        dist = math.sqrt((point.x - cam.x) ** 2 + (point.y - cam.y) ** 2 + (point.z - cam.z) ** 2)
        size = max(bbox.w, bbox.h) * dist / self.CAMERA_FOCAL
        if largest is not None and size > largest:
            rospy.loginfo("too large (%.3f m) to be a %s", size, label)
            return False
        if smallest is not None and size < smallest:
            rospy.loginfo("too small (%.3f m) to be a %s", size, label)
            return False
        return True

    def pick_and_deposit(self, place, pose, failed):
        """62401543 池畑一志"""
        # 探索位置で物体を1つ掴んで収納する。
        # 収納できたらdone、失敗したらfailed、物体が無ければnone、後回しにしたらdeferredを返す
        self.goto_name(place)
        self.change_pose(pose)
        self.set_gripper(0, wait=1.0)
        target = self.select_grasp_target(failed)
        if target is None:
            return "none"

        failed.append((target["point"].x, target["point"].y))
        plan = self.choose_grasp_plan(target)
        if plan is None:
            rospy.logwarn("skip the %s: no way to grasp it without collision", target["label"])
            return "failed"
        target = self.refine_target(target, pose, plan[1])
        label, point = target["label"], target["point"]
        failed.append((point.x, point.y))
        if self.is_skipped(label, point):
            rospy.loginfo("skip the %s: it is too small to grasp", label)
            return "failed"
        if label in self.AXIS_LABELS:
            # 長い方向を点群から測り、物体だけの点の中心(clampは軸の付近)をつかむ
            axis = self.object_axis(target)
            if axis is not None:
                point = copy.deepcopy(point)
                point.x, point.y = axis["center"]
                target = dict(target, point=point, axis=axis)
        if self.deferring and label in self.DEFER_LABELS:
            rospy.loginfo("put off the %s until the other objects are put away", label)
            if (place, pose) not in self.deferred:
                self.deferred.append((place, pose))
            return "deferred"
        # 近づいて見えた周りの物体も含めて、手が当たらないつかみ方を選び直す。
        # 失敗したら、手首の回転や向きが異なる別のつかみ方で試す
        stand = self.stand_up_plan(target)
        plans = self.choose_grasp_plans(target, plan[1], self.GRASP_PLAN_TRIALS)
        if stand is not None:
            plans = [stand[0]] + plans[:self.GRASP_PLAN_TRIALS - 1]
        if not plans:
            rospy.logwarn("skip the %s: no way to grasp it without collision", label)
            return "failed"
        for plan in plans:
            rospy.loginfo("grasp the %s at (%.2f, %.2f, %.2f) by %s",
                          label, point.x, point.y, point.z, plan)
            if self.exec_grasp_plan(target, plan):
                break
            rospy.logwarn("failed to grasp the %s", label)
            self.set_gripper(1)
            if self.carry_down:
                self.carry_down = False
                self.fold_arm()
            if self.last_seen is not None:
                # 小さい物体は検出位置が数cmずれるので、近くから見直した位置で掴み直す
                point = self.last_seen
                target = dict(target, point=point)
        else:
            return "failed"
        self.forget_floor_object(point)
        if stand is not None and plan == stand[0]:
            # 手を水平にして縦に起こした物体を、下端が縁に当たらない高さで置く
            self.hang = stand[1] / 2.0
            self.deposit_object(label, "stand")
        else:
            self.deposit_object(label, plan[0])
        del failed[-2:]
        return "done"

    def execute_task1(self):
        """
        task1を実行する。探索位置ごとに、物体が無くなるか失敗が続くまで収納を繰り返す
        """
        rospy.loginfo("#### start Task 1 ####")
        start = rospy.Time.now()
        self.planner = GridPlanner(rospy.wait_for_message("/map", OccupancyGrid))
        for drawer in self.coordinates["drawers"]:
            if drawer.get("enabled", True):
                self.open_drawer(drawer)

        total = 0
        for place, pose in self.coordinates["search_points"]:
            total += self.clear_search_point(place, pose, total, start)
        # 後回しにした物体(cup)を、見つけた探索位置に戻って最後に入れる
        self.deferring = False
        for place, pose in self.deferred:
            total += self.clear_search_point(place, pose, total, start)
        self.planner = None
        rospy.loginfo("task1 finished: %d objects", total)

    def clear_search_point(self, place, pose, total, start):
        """62411499 髙野幹太"""
        # 探索位置で、物体が無くなるか失敗が続くまで収納を繰り返し、収納した数を返す。
        # 後回しにした物体(deferred)は失敗に数えない
        failed = []
        fails = 0
        done = 0
        while total + done < self.MAX_TASK1_OBJECTS and fails < self.MAX_FAILS_PER_POINT:
            if (self.TASK1_TIME_LIMIT is not None
                    and (rospy.Time.now() - start).to_sec() > self.TASK1_TIME_LIMIT):
                rospy.logwarn("task1 time is up")
                break
            result = self.pick_and_deposit(place, pose, failed)
            if result == "none":
                break
            if result == "done":
                done += 1
            elif result == "failed":
                fails += 1
        return done

    def execute_task2a(self):
        """
        task2aを実行する
        """
        rospy.loginfo("#### start Task 2a ####")
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.change_pose("look_at_near_floor")
        self.goto_name("standby_2a")

        # 落ちているブロックを避けて移動
        self.execute_avoid_blocks()

        whole_body.move_to_go()

    def execute_task2b(self):
        """
        task2bを実行する
        """
        rospy.loginfo("#### start Task 2b ####")

        # 指示文は棚の前で受信できるので、先に棚へ移動する
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        target_obj, target_person = self.wait_for_instruction(self.INSTRUCTION_TIMEOUT)
        if target_obj is None:
            rospy.logwarn("No valid instruction. task2b is aborted.")
            return

        # 指定したオブジェクトを指定した配達先へ
        self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")
        tasks = rospy.get_param("~tasks", "task1,task2a,task2b").split(",")
        if "task1" in tasks:
            self.execute_task1()
        if "task2a" in tasks:
            self.execute_task2a()
        if "task2b" in tasks:
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
