"""Woosh chassis boundary used by the two-rack workflow.

The real adapter delegates motion to audited robot-side helpers.  Those
helpers own the Woosh SDK/ROS session and must close the loop against live
pose feedback; this Python layer never estimates displacement from time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess
from typing import Protocol, Sequence

from tube_grabber.core.errors import HardwareError


@dataclass(frozen=True)
class ChassisMove:
    rotation_deg: float
    translation_x_m: float
    rotation_speed_radps: float
    translation_speed_mps: float
    yaw_tolerance_deg: float
    position_tolerance_m: float
    maximum_rotation_translation_m: float
    timeout_s: float

    def __post_init__(self) -> None:
        values = (
            self.rotation_deg,
            self.translation_x_m,
            self.rotation_speed_radps,
            self.translation_speed_mps,
            self.yaw_tolerance_deg,
            self.position_tolerance_m,
            self.maximum_rotation_translation_m,
            self.timeout_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("chassis move values must be finite")
        if not 170.0 <= abs(self.rotation_deg) <= 190.0:
            raise ValueError("two-rack chassis rotation must be about 180 degrees")
        if abs(self.translation_x_m) < 0.01:
            raise ValueError("chassis translation_x_m must be measured and non-zero")
        if abs(self.translation_x_m) > 5.0:
            raise ValueError("chassis translation_x_m exceeds the 5 m safety limit")
        if not 0.02 <= self.rotation_speed_radps <= 0.10:
            raise ValueError("rotation speed must be between 0.02 and 0.10 rad/s")
        if not 0.01 <= self.translation_speed_mps <= 0.04:
            raise ValueError("translation speed must be between 0.01 and 0.04 m/s")
        if not 0.2 <= self.yaw_tolerance_deg <= 3.0:
            raise ValueError("yaw tolerance must be between 0.2 and 3 degrees")
        if not 0.005 <= self.position_tolerance_m <= 0.05:
            raise ValueError("position tolerance must be between 0.005 and 0.05 m")
        if not 0.005 <= self.maximum_rotation_translation_m <= 0.10:
            raise ValueError("rotation translation guard must be 0.005..0.10 m")
        if not 10.0 <= self.timeout_s <= 60.0:
            raise ValueError("chassis timeout must be between 10 and 60 seconds")


class ChassisPort(Protocol):
    def move_to_destination(self, move: ChassisMove) -> None: ...

    def return_to_start(self, move: ChassisMove) -> None: ...

    def stop(self) -> None: ...


class FakeChassis:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = bool(fail)
        self.moves: list[ChassisMove] = []
        self.return_moves: list[ChassisMove] = []
        self.stop_count = 0

    def move_to_destination(self, move: ChassisMove) -> None:
        self.moves.append(move)
        if self.fail:
            raise HardwareError("fake chassis move failed")

    def return_to_start(self, move: ChassisMove) -> None:
        self.return_moves.append(move)
        if self.fail:
            raise HardwareError("fake chassis return failed")

    def stop(self) -> None:
        self.stop_count += 1


class WooshHelperChassis:
    """Invoke closed-loop Woosh helpers installed on the robot computer.

    The referenced dispenser project audits a roughly 90-degree rotation
    helper.  A 180-degree turn is therefore split into two bounded rotations,
    followed by one pose-feedback translation.  Every helper must return only
    after sending repeated zero velocity and verifying its final pose.
    """

    _POSE_RE = re.compile(
        r"(?:pose|final)\s+x=(?P<x>[-+0-9.eE]+)\s+"
        r"y=(?P<y>[-+0-9.eE]+)\s+"
        r"(?:theta|yaw)=(?P<yaw>[-+0-9.eE]+)"
    )
    _MAX_TRANSLATION_SEGMENT_M = 0.25

    def __init__(
        self,
        *,
        rotate_helper_path: str,
        pose_servo_path: str,
    ) -> None:
        self.rotate_helper_path = str(rotate_helper_path)
        self.pose_servo_path = str(pose_servo_path)

    def move_to_destination(self, move: ChassisMove) -> None:
        self._require_helpers()
        half_turn_rad = math.radians(move.rotation_deg / 2.0)
        try:
            for _ in range(2):
                self._run(
                    [
                        self.rotate_helper_path,
                        "--rotate-relative-rad",
                        _number(half_turn_rad),
                        "--max-angular-radps",
                        _number(move.rotation_speed_radps),
                        "--yaw-tolerance-rad",
                        _number(math.radians(move.yaw_tolerance_deg)),
                        "--max-translation-m",
                        _number(move.maximum_rotation_translation_m),
                        "--timeout-s",
                        _number(move.timeout_s),
                    ],
                    move.timeout_s + 10.0,
                )
            for translation in _translation_segments(
                move.translation_x_m,
                self._MAX_TRANSLATION_SEGMENT_M,
            ):
                self._run(
                    [
                        self.pose_servo_path,
                        "--dx-m",
                        _number(translation),
                        "--dy-m",
                        "0",
                        "--dyaw-rad",
                        "0",
                        "--max-linear-mps",
                        _number(move.translation_speed_mps),
                        "--max-angular-radps",
                        _number(move.rotation_speed_radps),
                        "--position-tolerance-m",
                        _number(move.position_tolerance_m),
                        "--yaw-tolerance-rad",
                        _number(math.radians(move.yaw_tolerance_deg)),
                        "--timeout-s",
                        _number(move.timeout_s),
                    ],
                    move.timeout_s + 10.0,
                )
        except Exception:
            try:
                self.stop()
            except Exception:
                pass
            raise

    def return_to_start(self, move: ChassisMove) -> None:
        """Undo the chassis route while leaving the arm at its home pose.

        The forward route is rotation followed by translation in the rotated
        body frame.  Its geometric inverse is therefore translation in the
        current body frame first, followed by the two opposite half-turns.
        """

        self._require_helpers()
        half_turn_rad = -math.radians(move.rotation_deg / 2.0)
        try:
            for translation in _translation_segments(
                -move.translation_x_m,
                self._MAX_TRANSLATION_SEGMENT_M,
            ):
                self._run(
                    [
                        self.pose_servo_path,
                        "--dx-m",
                        _number(translation),
                        "--dy-m",
                        "0",
                        "--dyaw-rad",
                        "0",
                        "--max-linear-mps",
                        _number(move.translation_speed_mps),
                        "--max-angular-radps",
                        _number(move.rotation_speed_radps),
                        "--position-tolerance-m",
                        _number(move.position_tolerance_m),
                        "--yaw-tolerance-rad",
                        _number(math.radians(move.yaw_tolerance_deg)),
                        "--timeout-s",
                        _number(move.timeout_s),
                    ],
                    move.timeout_s + 10.0,
                )
            for _ in range(2):
                self._run(
                    [
                        self.rotate_helper_path,
                        "--rotate-relative-rad",
                        _number(half_turn_rad),
                        "--max-angular-radps",
                        _number(move.rotation_speed_radps),
                        "--yaw-tolerance-rad",
                        _number(math.radians(move.yaw_tolerance_deg)),
                        "--max-translation-m",
                        _number(move.maximum_rotation_translation_m),
                        "--timeout-s",
                        _number(move.timeout_s),
                    ],
                    move.timeout_s + 10.0,
                )
        except Exception:
            try:
                self.stop()
            except Exception:
                pass
            raise

    def stop(self) -> None:
        self._require_executable(self.rotate_helper_path, "Woosh rotation helper")
        self._run([self.rotate_helper_path, "--stop"], 10.0, require_pose=False)

    def _require_helpers(self) -> None:
        self._require_executable(self.rotate_helper_path, "Woosh rotation helper")
        self._require_executable(self.pose_servo_path, "Woosh pose-servo helper")

    @staticmethod
    def _require_executable(path: str, label: str) -> None:
        candidate = Path(path)
        if not candidate.is_file():
            raise HardwareError(f"{label} does not exist: {path}")

    def _run(
        self,
        command: Sequence[str],
        timeout_s: float,
        *,
        require_pose: bool = True,
    ) -> str:
        try:
            result = subprocess.run(
                list(command),
                capture_output=True,
                text=True,
                check=False,
                timeout=float(timeout_s),
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise HardwareError(f"Woosh helper failed to run: {error}") from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise HardwareError(
                f"Woosh helper returned {result.returncode}: {detail}"
            )
        if require_pose and not self._POSE_RE.search(result.stdout):
            raise HardwareError("Woosh helper output has no verified final pose")
        return result.stdout


def _number(value: float) -> str:
    return f"{float(value):.12g}"


def _translation_segments(total_m: float, maximum_m: float) -> tuple[float, ...]:
    count = max(1, math.ceil(abs(float(total_m)) / float(maximum_m)))
    segment = float(total_m) / count
    return tuple(segment for _ in range(count))
