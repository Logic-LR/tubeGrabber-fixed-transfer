#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEPLOY_DIR="/home/rm/tubeGrabber-mobile-current"
readonly AGV_DIR="$DEPLOY_DIR/tools/agv_debug_tools"
readonly CONDA="/home/rm/miniconda3/bin/conda"

cleanup() {
  cd "$AGV_DIR"
  ./grabber_rotate_relative --stop >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "[1/5] 停止冲突的机械臂控制进程"
pkill -TERM -x atom 2>/dev/null || true
pkill -TERM -f '[z]hixing_ctrl\.py' 2>/dev/null || true

for _ in $(seq 1 20); do
  if ! pgrep -x atom >/dev/null &&
     ! pgrep -f '[z]hixing_ctrl\.py' >/dev/null; then
    break
  fi
  sleep 0.25
done

if pgrep -x atom >/dev/null ||
   pgrep -f '[z]hixing_ctrl\.py' >/dev/null; then
  echo "错误：atom 或 zhixing_ctrl.py 未退出，未发送任何运动指令" >&2
  exit 20
fi

echo "[2/2] 连续执行：自动取管 -> loaded home -> 旋转/前移 -> rack2 自动选空槽插管 -> 右臂回 home -> 底盘反向回起点"
cd "$DEPLOY_DIR"
env PYTHONPATH="$DEPLOY_DIR" "$CONDA" run \
  -n tube_vision --no-capture-output --cwd "$DEPLOY_DIR" \
  python -m tube_grabber --real mobile-transfer \
    --auto-source \
    --auto-destination

echo "任务完成：已取管、旋转/前移、插入 rack2 空槽，右臂保持 home，底盘已反向回到起始位置"
