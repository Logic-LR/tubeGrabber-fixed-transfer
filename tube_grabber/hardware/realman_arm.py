"""RealMan 右侧七轴机械臂驱动。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
import subprocess

from tube_grabber.core.errors import HardwareError
from tube_grabber.core.models import Pose6D
from tube_grabber.hardware._realman_sdk import (
    call_and_require,
    call_sdk,
    require_success,
    sdk_code,
)


class RealManArm:
    def __init__(
        self,
        ip: str,
        port: int = 8080,
        work_frame: str = "Base",
        tool_frame: str = "Arm_Tip",
        expected_dof: int = 7,
        reject_conflicting_processes: bool = True,
        conflicting_processes: Sequence[str] = ("atom", "zhixing_ctrl.py"),
    ) -> None:
        if not ip:
            raise ValueError("机械臂 IP 不能为空")
        if not 1 <= port <= 65535:
            raise ValueError("机械臂端口必须在 1..65535")
        if not work_frame:
            raise ValueError("工作坐标系不能为空")
        if not tool_frame:
            raise ValueError("工具坐标系不能为空")
        if expected_dof <= 0:
            raise ValueError("expected_dof 必须大于 0")

        self.ip = ip
        self.port = port
        self.work_frame = work_frame
        self.tool_frame = tool_frame
        self.expected_dof = int(expected_dof)
        self.reject_conflicting_processes = bool(reject_conflicting_processes)
        self.conflicting_processes = tuple(
            str(value).strip() for value in conflicting_processes if str(value).strip()
        )
        self._robot: object | None = None
        self._handle: object | None = None
        self._dof: int | None = None

    def connect(self) -> None:
        if self._robot is not None:
            return
        if self.reject_conflicting_processes:
            conflicts = _running_control_processes(self.conflicting_processes)
            if conflicts:
                raise HardwareError(
                    "检测到会覆盖 SDK 指令的后台进程："
                    + ", ".join(conflicts)
                )

        try:
            sdk = import_module("Robotic_Arm.rm_robot_interface")
        except ImportError as exc:
            raise HardwareError(
                "未安装 RealMan Python SDK，无法导入 "
                "Robotic_Arm.rm_robot_interface"
            ) from exc

        robot: object | None = None
        try:
            robot = sdk.RoboticArm(sdk.rm_thread_mode_e.RM_TRIPLE_MODE_E)
            handle = robot.rm_create_robot_arm(self.ip, self.port)
            handle_id = _handle_id(handle)
            if handle_id <= 0:
                raise HardwareError(
                    f"连接右臂 {self.ip}:{self.port} 失败，handle.id={handle_id}"
                )

            call_and_require(
                f"切换工作坐标系 {self.work_frame}",
                robot.rm_change_work_frame,
                self.work_frame,
            )
            call_and_require(
                f"切换工具坐标系 {self.tool_frame}",
                robot.rm_change_tool_frame,
                self.tool_frame,
            )
            _require_current_frame(
                robot,
                "rm_get_current_work_frame",
                self.work_frame,
                "工作坐标系",
            )
            _require_current_frame(
                robot,
                "rm_get_current_tool_frame",
                self.tool_frame,
                "工具坐标系",
            )
            dof = _read_robot_dof(robot)
            if dof != self.expected_dof:
                raise HardwareError(
                    f"连接到 {dof} 轴机械臂，但本项目要求 "
                    f"{self.expected_dof} 轴右臂"
                )
            _read_sdk_pose(robot)
        except Exception as exc:
            if robot is not None:
                try:
                    robot.rm_delete_robot_arm()
                except Exception:
                    pass
            if isinstance(exc, HardwareError):
                raise
            raise HardwareError(
                f"连接右臂 {self.ip}:{self.port} 异常: {exc}"
            ) from exc

        self._robot = robot
        self._handle = handle
        self._dof = dof

    def disconnect(self) -> None:
        robot = self._robot
        self._robot = None
        self._handle = None
        self._dof = None
        if robot is None:
            return
        call_and_require("断开机械臂", robot.rm_delete_robot_arm)

    def get_pose(self) -> Pose6D:
        return _read_sdk_pose(self.sdk_robot)

    @property
    def dof(self) -> int:
        if self._dof is None:
            raise HardwareError("机械臂尚未连接，请先调用 connect()")
        return self._dof

    def get_run_mode(self) -> int:
        """Return 0 for simulation or 1 for the physical robot."""
        return _read_integer_state(
            self.sdk_robot,
            "rm_get_arm_run_mode",
            "读取机械臂运行模式",
        )

    def get_power_state(self) -> int:
        """Return 0 for powered off or 1 for powered on."""
        return _read_integer_state(
            self.sdk_robot,
            "rm_get_arm_power_state",
            "读取机械臂上电状态",
        )

    def require_healthy(self) -> None:
        """Reject controller or joint errors before any real motion."""
        robot = self.sdk_robot
        controller_method = getattr(robot, "rm_get_controller_state", None)
        if controller_method is None:
            raise HardwareError("当前 RealMan SDK 不支持 rm_get_controller_state")
        controller = call_sdk("读取控制器状态", controller_method)
        if not isinstance(controller, Mapping):
            raise HardwareError(f"控制器状态格式错误: {controller!r}")
        require_success("读取控制器状态", controller.get("return_code", -3))
        controller_error = int(
            controller.get("sys_err", controller.get("system_error", 0)) or 0
        )
        if controller_error:
            raise HardwareError(
                f"控制器存在系统错误 0x{controller_error:04X}，禁止运动"
            )

        joint_method = getattr(robot, "rm_get_joint_err_flag", None)
        if joint_method is None:
            raise HardwareError("当前 RealMan SDK 不支持 rm_get_joint_err_flag")
        joint = call_sdk("读取关节错误", joint_method)
        if not isinstance(joint, Mapping):
            raise HardwareError(f"关节错误状态格式错误: {joint!r}")
        require_success("读取关节错误", joint.get("return_code", -3))
        flags = joint.get("err_flag", joint.get("err", ()))
        if not isinstance(flags, Sequence) or isinstance(flags, (str, bytes)):
            raise HardwareError(f"关节错误标志格式错误: {flags!r}")
        errors = [int(value) for value in flags if int(value) != 0]
        if errors:
            raise HardwareError(
                "机械臂关节存在错误，禁止运动："
                + ", ".join(f"0x{value:04X}" for value in errors)
            )

    def move_pose(
        self,
        pose: Pose6D,
        speed_percent: int,
        *,
        linear: bool = False,
    ) -> None:
        if pose.frame != "base_right":
            raise HardwareError(
                f"机械臂只接受 base_right 坐标，收到 {pose.frame!r}"
            )
        if not 1 <= int(speed_percent) <= 100:
            raise ValueError("speed_percent 必须在 1..100")

        sdk_pose = _pose_to_sdk(pose)
        method_name = "rm_movel" if linear else "rm_movej_p"
        method = getattr(self.sdk_robot, method_name, None)
        if method is None:
            raise HardwareError(f"当前 RealMan SDK 不支持 {method_name}")
        operation = f"{method_name} 阻塞运动"
        result = call_sdk(
            operation,
            method,
            sdk_pose,
            int(speed_percent),
            0,
            0,
            1,
        )
        try:
            require_success(operation, result)
        except HardwareError as exc:
            target = ", ".join(f"{value:.4f}" for value in sdk_pose)
            raise HardwareError(f"{exc}；目标 SDK(m,rad)=[{target}]") from exc

    def stop(self) -> None:
        call_and_require("机械臂减速停止", self.sdk_robot.rm_set_arm_slow_stop)

    @property
    def sdk_robot(self) -> object:
        """仅供同一硬件层的两指夹爪共用连接。"""
        if self._robot is None:
            raise HardwareError("机械臂尚未连接，请先调用 connect()")
        return self._robot


def _handle_id(handle: object) -> int:
    value = getattr(handle, "id", handle)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HardwareError(f"无法读取机械臂 handle.id: {handle!r}") from exc


def _read_sdk_pose(robot: object) -> Pose6D:
    result = call_sdk("读取机械臂状态", robot.rm_get_current_arm_state)
    code = sdk_code(result, "读取机械臂状态")
    if code != 0:
        require_success("读取机械臂状态", code)
    if not isinstance(result, tuple) or len(result) < 2:
        raise HardwareError(f"机械臂状态格式错误: {result!r}")

    state = result[1]
    if not isinstance(state, Mapping) or "pose" not in state:
        raise HardwareError(f"机械臂状态缺少 pose: {state!r}")
    return _pose_from_sdk(state["pose"])


def _read_integer_state(robot: object, method_name: str, operation: str) -> int:
    method = getattr(robot, method_name, None)
    if method is None:
        raise HardwareError(f"当前 RealMan SDK 不支持 {method_name}")
    result = call_sdk(operation, method)
    require_success(operation, result)
    if not isinstance(result, tuple) or len(result) < 2:
        raise HardwareError(f"{operation} 返回格式错误: {result!r}")
    try:
        return int(result[1])
    except (TypeError, ValueError) as exc:
        raise HardwareError(f"{operation} 返回状态无效: {result!r}") from exc


def _read_robot_dof(robot: object) -> int:
    result = call_sdk("读取机械臂型号", robot.rm_get_robot_info)
    require_success("读取机械臂型号", result)
    if not isinstance(result, tuple) or len(result) < 2:
        raise HardwareError(f"读取机械臂型号返回格式错误: {result!r}")
    info = result[1]
    if not isinstance(info, Mapping) or "arm_dof" not in info:
        raise HardwareError(f"机械臂型号信息缺少 arm_dof: {info!r}")
    try:
        return int(info["arm_dof"])
    except (TypeError, ValueError) as exc:
        raise HardwareError(f"arm_dof 无效: {info!r}") from exc


def _pose_to_sdk(pose: Pose6D) -> list[float]:
    """项目使用 mm；只在 SDK 边界转换成 m。"""
    return [
        pose.x_mm / 1000.0,
        pose.y_mm / 1000.0,
        pose.z_mm / 1000.0,
        pose.rx_rad,
        pose.ry_rad,
        pose.rz_rad,
    ]


def _pose_from_sdk(value: object) -> Pose6D:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) < 6:
            raise HardwareError(f"SDK pose 长度不足 6: {value!r}")
        try:
            x, y, z, rx, ry, rz = (float(item) for item in value[:6])
        except (TypeError, ValueError) as exc:
            raise HardwareError(f"无法解析 SDK pose: {value!r}") from exc
    elif isinstance(value, Mapping):
        position = value.get("position", value)
        euler = value.get("euler", value)
        if not isinstance(position, Mapping) or not isinstance(euler, Mapping):
            raise HardwareError(f"无法解析 SDK pose: {value!r}")
        try:
            x, y, z = (float(position[key]) for key in ("x", "y", "z"))
            rx, ry, rz = (float(euler[key]) for key in ("rx", "ry", "rz"))
        except (KeyError, TypeError, ValueError) as exc:
            raise HardwareError(f"无法解析 SDK pose: {value!r}") from exc
    else:
        raise HardwareError(f"无法解析 SDK pose: {value!r}")

    return Pose6D(
        x_mm=x * 1000.0,
        y_mm=y * 1000.0,
        z_mm=z * 1000.0,
        rx_rad=rx,
        ry_rad=ry,
        rz_rad=rz,
        frame="base_right",
    )


def _require_current_frame(
    robot: object,
    method_name: str,
    expected: str,
    label: str,
) -> None:
    method = getattr(robot, method_name, None)
    if method is None:
        raise HardwareError(f"当前 RealMan SDK 不支持 {method_name}")
    result = call_sdk(f"读取当前{label}", method)
    require_success(f"读取当前{label}", result)
    if (
        not isinstance(result, tuple)
        or len(result) < 2
        or not isinstance(result[1], Mapping)
    ):
        raise HardwareError(f"当前{label}格式错误: {result!r}")
    value = result[1]
    actual = value.get("name", value.get("frame_name", value.get("tool_name")))
    if actual is None:
        raise HardwareError(f"当前{label}缺少名称: {value!r}")
    if str(actual) != expected:
        raise HardwareError(f"当前{label}是 {actual!r}，预期 {expected!r}")


def _running_control_processes(names: Sequence[str]) -> list[str]:
    conflicts = []
    for name in names:
        command = ["pgrep", "-x", name] if name == "atom" else ["pgrep", "-f", name]
        try:
            result = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            raise HardwareError(f"无法检查控制进程 {name}: {exc}") from exc
        if result.returncode == 0:
            conflicts.append(name)
    return conflicts
