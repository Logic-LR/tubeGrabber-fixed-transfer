#!/usr/bin/env python3
"""Quickly tune the flange-to-TCP offset without touching hardware.

All values are millimetres expressed in the RealMan end/flange frame.  Every
write also closes ``motion.parameters_confirmed`` because existing flange
waypoints are invalid after a TCP change.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "app.yaml"
TCP_PATTERN = re.compile(
    r"^(?P<prefix>\s*tcp_offset_end_mm:\s*)"
    r"\[(?P<values>[^\]]+)\](?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)
MOTION_LOCK_PATTERN = re.compile(
    r"^(?P<prefix>  parameters_confirmed:\s*)"
    r"(?:true|false)(?P<suffix>\s*(?:#.*)?)$",
    re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="快速调整末端坐标系中的 TCP 偏移（mm，不连接硬件）"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--set",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        dest="set_values",
        help="绝对设置，例如 --set 3 15 220",
    )
    mode.add_argument(
        "--delta",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help="相对当前值增减，例如 --delta 0 -2 0",
    )
    mode.add_argument(
        "--move-end",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        help=(
            "希望实际夹爪下一次沿末端轴移动的距离；"
            "脚本自动按反号修改 TCP"
        ),
    )
    mode.add_argument(
        "--show",
        action="store_true",
        help="只显示当前 TCP，不修改文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算新值，不写入文件",
    )
    return parser.parse_args()


def read_config() -> tuple[str, tuple[float, float, float]]:
    text = CONFIG_PATH.read_text(encoding="utf-8")
    match = TCP_PATTERN.search(text)
    if match is None:
        raise ValueError(f"找不到 geometry.tcp_offset_end_mm：{CONFIG_PATH}")
    raw = [part.strip() for part in match.group("values").split(",")]
    if len(raw) != 3:
        raise ValueError("tcp_offset_end_mm 必须正好包含3个数")
    values = tuple(float(value) for value in raw)
    validate_tcp(values)
    return text, values


def validate_tcp(values: tuple[float, float, float]) -> None:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("TCP 必须是有限数值")
    if any(abs(value) > 500.0 for value in values):
        raise ValueError("TCP 每一轴绝对值不得超过500 mm")
    if math.sqrt(sum(value * value for value in values)) > 600.0:
        raise ValueError("TCP 总长度不得超过600 mm")


def format_tcp(values: tuple[float, float, float]) -> str:
    return "[" + ", ".join(f"{value:.3f}" for value in values) + "] mm"


def write_tcp(
    text: str,
    values: tuple[float, float, float],
) -> None:
    validate_tcp(values)
    formatted = ", ".join(f"{value:.3f}" for value in values)
    updated, tcp_count = TCP_PATTERN.subn(
        lambda match: (
            f"{match.group('prefix')}[{formatted}]{match.group('suffix')}"
        ),
        text,
        count=1,
    )
    if tcp_count != 1:
        raise ValueError("无法唯一更新 tcp_offset_end_mm")
    updated, lock_count = MOTION_LOCK_PATTERN.subn(
        lambda match: f"{match.group('prefix')}false{match.group('suffix')}",
        updated,
        count=1,
    )
    if lock_count != 1:
        raise ValueError("无法唯一关闭 motion.parameters_confirmed")
    temporary = CONFIG_PATH.with_suffix(".yaml.tcp-tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(CONFIG_PATH)


def apply_change(
    current: tuple[float, float, float],
    *,
    absolute: tuple[float, float, float] | None = None,
    delta: tuple[float, float, float] | None = None,
) -> tuple[float, float, float]:
    if (absolute is None) == (delta is None):
        raise ValueError("必须且只能提供 set 或 delta")
    if absolute is not None:
        result = absolute
    else:
        assert delta is not None
        result = tuple(current[index] + delta[index] for index in range(3))
    result = tuple(float(value) for value in result)
    validate_tcp(result)
    return result


def interactive() -> int:
    print("坐标：RealMan 末端/法兰局部坐标系；单位：mm")
    print("输入：move dx dy dz（希望实际夹爪沿末端轴移动）")
    print("或：delta dx dy dz（直接修改TCP），set x y z（绝对值）")
    print("命令：show 查看，q 退出")
    while True:
        text, current = read_config()
        print(f"\n当前 TCP：{format_tcp(current)}")
        try:
            command = input("tcp> ").strip()
        except EOFError:
            print()
            return 0
        if not command:
            continue
        if command.lower() in {"q", "quit", "exit"}:
            return 0
        if command.lower() == "show":
            continue
        parts = command.replace(",", " ").split()
        try:
            action = parts[0].lower()
            if action == "set":
                if len(parts) != 4:
                    raise ValueError("格式应为 set x y z")
                new = apply_change(
                    current,
                    absolute=tuple(float(value) for value in parts[1:4]),
                )
            elif action in {"move", "delta"}:
                if len(parts) != 4:
                    raise ValueError(f"格式应为 {action} dx dy dz")
                requested = tuple(float(value) for value in parts[1:4])
                delta = (
                    tuple(-value for value in requested)
                    if action == "move"
                    else requested
                )
                new = apply_change(current, delta=delta)
            else:
                if len(parts) != 3:
                    raise ValueError(
                        "请使用 move、delta、set、show 或 q"
                    )
                delta = tuple(float(value) for value in parts)
                new = apply_change(current, delta=delta)
            write_tcp(text, new)
            print(f"已更新：{format_tcp(current)} -> {format_tcp(new)}")
            print("运动锁已关闭；重新规划确认前不能 transfer。")
        except ValueError as error:
            print(f"输入错误：{error}")


def main() -> int:
    args = parse_args()
    try:
        text, current = read_config()
        if args.show:
            print(f"当前 TCP（end frame）：{format_tcp(current)}")
            return 0
        if (
            args.set_values is None
            and args.delta is None
            and args.move_end is None
        ):
            return interactive()
        absolute = (
            None
            if args.set_values is None
            else tuple(float(value) for value in args.set_values)
        )
        delta = (
            None
            if args.delta is None
            else tuple(float(value) for value in args.delta)
        )
        if args.move_end is not None:
            delta = tuple(-float(value) for value in args.move_end)
        new = apply_change(current, absolute=absolute, delta=delta)
        print(f"当前 TCP：{format_tcp(current)}")
        print(f"目标 TCP：{format_tcp(new)}")
        if args.dry_run:
            print("dry-run：未修改文件。")
            return 0
        write_tcp(text, new)
        print("已写入 config/app.yaml；motion.parameters_confirmed=false。")
        return 0
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
