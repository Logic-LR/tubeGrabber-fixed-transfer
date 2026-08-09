"""RealMan SDK 返回值处理。此文件不导入 SDK。"""

from __future__ import annotations

from collections.abc import Callable

from tube_grabber.core.errors import HardwareError


_ERROR_TEXT = {
    1: "控制器拒绝命令，参数或机械臂状态不正确",
    -1: "数据发送失败",
    -2: "数据接收失败或控制器未返回",
    -3: "返回数据解析失败",
    -4: "等待超时",
    -5: "到位设备校验失败",
}


def sdk_code(result: object, operation: str) -> int:
    """兼容 SDK 直接返回 int 或以 code 开头的 tuple。"""
    if isinstance(result, tuple) and not result:
        raise HardwareError(f"{operation} 返回了空 tuple")
    value = result[0] if isinstance(result, tuple) else result
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HardwareError(f"{operation} 返回了无法识别的结果: {result!r}") from exc


def require_success(operation: str, result: object) -> None:
    code = sdk_code(result, operation)
    if code == 0:
        return
    detail = _ERROR_TEXT.get(code, "未知错误")
    raise HardwareError(f"{operation} 失败：SDK 错误码 {code}（{detail}）")


def call_sdk(operation: str, method: Callable[..., object], *args: object) -> object:
    """把 SDK 自己抛出的异常也统一成 HardwareError。"""
    try:
        return method(*args)
    except Exception as exc:
        raise HardwareError(f"{operation} 调用异常: {type(exc).__name__}: {exc}") from exc


def call_and_require(
    operation: str,
    method: Callable[..., object],
    *args: object,
) -> None:
    require_success(operation, call_sdk(operation, method, *args))
