"""Project-specific errors with clear meanings."""


class TubeGrabberError(RuntimeError):
    """Base error for expected project failures."""


class ConfigError(TubeGrabberError):
    """Configuration is missing or invalid."""


class HardwareError(TubeGrabberError):
    """A camera, arm, or gripper operation failed."""


class VisionError(TubeGrabberError):
    """The rack cannot be identified safely."""


class MotionError(TubeGrabberError):
    """A motion plan is invalid or execution failed."""


class WorkflowError(TubeGrabberError):
    """The requested manipulation cannot continue safely."""


class AgentError(TubeGrabberError):
    """A language command could not be converted into one safe task."""
