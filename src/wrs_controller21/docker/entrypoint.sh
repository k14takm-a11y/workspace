#!/bin/bash -e
# copy from https://github.com/devrt/ros-devcontainer-vscode/blob/dfad41c87d8e78614856e2d1598d2ac7a6f16b9d/.devcontainer/entrypoint.sh
# edited by smilab

USER_ID=$(id -u)
GROUP_ID=$(id -g)

sudo usermod -u $USER_ID -o -m -d /home/developer developer > /dev/null 2>&1
sudo groupmod -g $GROUP_ID developer > /dev/null 2>&1
sudo chown -R developer:developer /workspace

ln -sfn /home/developer/.vscode /workspace/.vscode
ln -sfn /workspace /home/developer/workspace

ROS_DISTRO=$(ls /opt/ros/ | head -n 1)
source /opt/ros/$ROS_DISTRO/setup.bash

# 空でないSSH鍵ファイルがある場合はコピーする
if [ -e /run/secrets/host_ssh_key ]; then
  if [ -s /run/secrets/host_ssh_key ]; then
    mkdir -p /home/developer/.ssh
    sudo chmod 700 /home/developer/.ssh
    sudo chown developer:developer /home/developer/.ssh

    sudo cp /run/secrets/host_ssh_key /home/developer/.ssh/id_ed25519
    sudo chmod 600 /home/developer/.ssh/id_ed25519
    sudo chown developer:developer /home/developer/.ssh/id_ed25519
  fi
fi

cd /home/developer
exec $@
