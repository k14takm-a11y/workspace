#!/bin/bash -eu

seeds=(1 25 31 36 59 86)
res=0

# seedsに一致する場合はseed変更処理へ
# 期待する値でなければ、何も返さず終了
if [ $# != 1 ]; then
    res=0
else
    for i in "${seeds[@]}" ; do
        if [ $i = $1 ]; then
            res=$i
        fi
    done
fi

if [ $res -eq 0 ]; then
    exit 1
fi

# バックアップがなければ作成
# バックアップから文字列置換し対象のファイルを書き換え
docker compose exec simulator bash -c "\
cd /opt/ros/noetic/share/hsrb_wrs_gazebo_launch/launch && \
org_file=wrs_practice0_easy_tmc.launch && \
bk_file=wrs_practice0_easy_tmc_bk.launch && \
if [ ! -e \$bk_file ]; then cp -f \$org_file \$bk_file; fi
sed 's/name=\"seed\" default=\"[1-9]*\"/name=\"seed\" default=\"'$res'\"/' \$bk_file > \$org_file && \
grep seed wrs_practice0_easy_tmc.launch
"
