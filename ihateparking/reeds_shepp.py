"""Reeds-Shepp curves for the Hybrid A* analytic expansion.

The word families follow Reeds & Shepp (1990) in the compact formulation used
by most Hybrid A* implementations (CSC, CCC, CCCC, CCSC and CCSCC, each with
time-flip and reflection symmetries). Every candidate is integrated and kept
only if it actually reaches the requested goal, so an inapplicable formula can
never produce a path that ends somewhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

from .geometry import Pose, wrap_angle

_HALF_PI = math.pi / 2.0
_ENDPOINT_TOLERANCE = 1e-6
_ZERO_SEGMENT = 1e-10

Word = tuple[float, ...]
Formula = Callable[[float, float, float], Optional[tuple[Word, str]]]


@dataclass(frozen=True)
class ReedsSheppPath:
    """A path in world units: signed segment lengths and their L/S/R letters."""

    lengths: tuple[float, ...]
    types: str
    turning_radius: float

    @property
    def length(self) -> float:
        return sum(abs(item) for item in self.lengths)

    @property
    def gear_switches(self) -> int:
        signs = [1 if item > 0 else -1 for item in self.lengths if abs(item) > _ZERO_SEGMENT]
        return sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    @property
    def reverse_length(self) -> float:
        return sum(-item for item in self.lengths if item < 0)


@dataclass(frozen=True)
class PathSample:
    pose: Pose
    gear: int
    curvature: float


def _mod2pi(angle: float) -> float:
    value = math.fmod(angle, 2.0 * math.pi)
    if value < -math.pi:
        value += 2.0 * math.pi
    elif value > math.pi:
        value -= 2.0 * math.pi
    return value


def _polar(x: float, y: float) -> tuple[float, float]:
    return math.hypot(x, y), math.atan2(y, x)


def _lsl(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u, t = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 0.0 <= t <= math.pi:
        v = _mod2pi(phi - t)
        if 0.0 <= v <= math.pi:
            return (t, u, v), "LSL"
    return None


def _lsr(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, t1 = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u1 *= u1
    if u1 >= 4.0:
        u = math.sqrt(u1 - 4.0)
        t = _mod2pi(t1 + math.atan2(2.0, u))
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return (t, u, v), "LSR"
    return None


def _lrl_forward_flip(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + _HALF_PI)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(phi - t - u)
        return (t, -u, v), "LRL"
    return None


def _lrl_back_flip(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 <= 4.0:
        a = math.acos(0.25 * u1)
        t = _mod2pi(a + theta + _HALF_PI)
        u = _mod2pi(math.pi - 2.0 * a)
        v = _mod2pi(-phi + t + u)
        return (t, -u, -v), "LRL"
    return None


def _lrl_late_flip(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if 1e-9 < u1 <= 4.0:
        u = math.acos(max(-1.0, min(1.0, 1.0 - u1 * u1 * 0.125)))
        a = math.asin(max(-1.0, min(1.0, 2.0 * math.sin(u) / u1)))
        t = _mod2pi(-a + theta + _HALF_PI)
        v = _mod2pi(t - u - phi)
        return (t, u, -v), "LRL"
    return None


def _lrlr_inner(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 <= 2.0:
        a = math.acos((u1 + 2.0) * 0.25)
        t = _mod2pi(theta + a + _HALF_PI)
        u = _mod2pi(a)
        v = _mod2pi(phi - t + 2.0 * u)
        if t >= 0.0 and u >= 0.0 and v >= 0.0:
            return (t, u, -u, -v), "LRLR"
    return None


def _lrlr_outer(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    u2 = (20.0 - u1 * u1) / 16.0
    if 0.0 <= u2 <= 1.0 and u1 > 1e-9:
        u = math.acos(u2)
        a = math.asin(max(-1.0, min(1.0, 2.0 * math.sin(u) / u1)))
        t = _mod2pi(theta + a + _HALF_PI)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return (t, -u, -u, v), "LRLR"
    return None


def _lrsl(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 >= 2.0:
        u = math.sqrt(u1 * u1 - 4.0) - 2.0
        a = math.atan2(2.0, u + 2.0)
        t = _mod2pi(theta + a + _HALF_PI)
        v = _mod2pi(t - phi + _HALF_PI)
        if t >= 0.0 and v >= 0.0:
            return (t, -_HALF_PI, -u, -v), "LRSL"
    return None


def _lsrl(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x - math.sin(phi), y - 1.0 + math.cos(phi))
    if u1 >= 2.0:
        u = math.sqrt(u1 * u1 - 4.0) - 2.0
        a = math.atan2(u + 2.0, 2.0)
        t = _mod2pi(theta - a + _HALF_PI)
        v = _mod2pi(t - phi - _HALF_PI)
        if t >= 0.0 and v >= 0.0:
            return (t, u, _HALF_PI, -v), "LSRL"
    return None


def _lrsr(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 2.0:
        t = _mod2pi(theta + _HALF_PI)
        u = u1 - 2.0
        v = _mod2pi(phi - t - _HALF_PI)
        if t >= 0.0 and v >= 0.0:
            return (t, -_HALF_PI, -u, -v), "LRSR"
    return None


def _lslr(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 2.0:
        t = _mod2pi(theta)
        u = u1 - 2.0
        v = _mod2pi(phi - t - _HALF_PI)
        if t >= 0.0 and v >= 0.0:
            return (t, u, _HALF_PI, -v), "LSLR"
    return None


def _lrslr(x: float, y: float, phi: float) -> Optional[tuple[Word, str]]:
    u1, theta = _polar(x + math.sin(phi), y - 1.0 - math.cos(phi))
    if u1 >= 4.0:
        u = math.sqrt(u1 * u1 - 4.0) - 4.0
        a = math.atan2(2.0, u + 4.0)
        t = _mod2pi(theta + a + _HALF_PI)
        v = _mod2pi(t - phi)
        if t >= 0.0 and v >= 0.0:
            return (t, -_HALF_PI, -u, -_HALF_PI, v), "LRSLR"
    return None


_FORMULAS: tuple[Formula, ...] = (
    _lsl, _lsr, _lrl_forward_flip, _lrl_back_flip, _lrl_late_flip,
    _lrlr_inner, _lrlr_outer, _lrsl, _lsrl, _lrsr, _lslr, _lrslr,
)

_REFLECT = str.maketrans("LR", "RL")


def _normalized_words(x: float, y: float, phi: float) -> list[tuple[Word, str]]:
    """All candidate words for a unit-radius query, including symmetries."""
    words: list[tuple[Word, str]] = []
    for formula in _FORMULAS:
        plain = formula(x, y, phi)
        if plain:
            words.append(plain)
        timeflip = formula(-x, y, -phi)
        if timeflip:
            words.append((tuple(-item for item in timeflip[0]), timeflip[1]))
        reflect = formula(x, -y, -phi)
        if reflect:
            words.append((reflect[0], reflect[1].translate(_REFLECT)))
        both = formula(-x, -y, phi)
        if both:
            words.append((tuple(-item for item in both[0]), both[1].translate(_REFLECT)))
    return words


def _advance(x: float, y: float, yaw: float, length: float, kind: str) -> tuple[float, float, float]:
    """Move along one unit-radius segment; a negative length drives in reverse."""
    if kind == "S":
        return x + length * math.cos(yaw), y + length * math.sin(yaw), yaw
    if kind == "L":
        return x + math.sin(yaw + length) - math.sin(yaw), y - math.cos(yaw + length) + math.cos(yaw), yaw + length
    return x - math.sin(yaw - length) + math.sin(yaw), y + math.cos(yaw - length) - math.cos(yaw), yaw - length


def _reaches(word: Word, types: str, x: float, y: float, phi: float) -> bool:
    px = py = pyaw = 0.0
    for length, kind in zip(word, types):
        px, py, pyaw = _advance(px, py, pyaw, length, kind)
    return (
        abs(px - x) < _ENDPOINT_TOLERANCE
        and abs(py - y) < _ENDPOINT_TOLERANCE
        and abs(wrap_angle(pyaw - phi)) < _ENDPOINT_TOLERANCE
    )


def reeds_shepp_paths(start: Pose, goal: Pose, turning_radius: float) -> list[ReedsSheppPath]:
    """Return every verified Reeds-Shepp path from start to goal, shortest first."""
    if turning_radius <= 0.0:
        raise ValueError("turning_radius must be positive")
    dx, dy = goal.x - start.x, goal.y - start.y
    c, s = math.cos(start.yaw), math.sin(start.yaw)
    x = (c * dx + s * dy) / turning_radius
    y = (-s * dx + c * dy) / turning_radius
    phi = wrap_angle(goal.yaw - start.yaw)

    paths: list[ReedsSheppPath] = []
    seen: set[tuple[str, tuple[float, ...]]] = set()
    for word, types in _normalized_words(x, y, phi):
        compact = [(length, kind) for length, kind in zip(word, types) if abs(length) > _ZERO_SEGMENT]
        if not compact:
            continue
        lengths = tuple(length for length, _ in compact)
        letters = "".join(kind for _, kind in compact)
        if not _reaches(lengths, letters, x, y, phi):
            continue
        key = (letters, tuple(round(item, 9) for item in lengths))
        if key in seen:
            continue
        seen.add(key)
        paths.append(ReedsSheppPath(tuple(item * turning_radius for item in lengths), letters, turning_radius))
    paths.sort(key=lambda path: path.length)
    return paths


def sample_path(start: Pose, path: ReedsSheppPath, step: float) -> list[PathSample]:
    """Sample a path in world coordinates, always including every segment end.

    Each sample carries the gear and curvature used to arrive at it, so a
    caller can rebuild per-pose gear and steering commands.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    radius = path.turning_radius
    samples: list[PathSample] = []
    x, y, yaw = 0.0, 0.0, 0.0
    for length, kind in zip(path.lengths, path.types):
        unit_length = length / radius
        count = max(1, math.ceil(abs(length) / step))
        gear = 1 if length > 0 else -1
        curvature = 0.0 if kind == "S" else (1.0 if kind == "L" else -1.0) / radius
        for index in range(1, count + 1):
            sx, sy, syaw = _advance(x, y, yaw, unit_length * index / count, kind)
            samples.append(PathSample(_to_world(start, sx, sy, syaw, radius), gear, curvature))
        x, y, yaw = _advance(x, y, yaw, unit_length, kind)
    return samples


def _to_world(start: Pose, x: float, y: float, yaw: float, radius: float) -> Pose:
    c, s = math.cos(start.yaw), math.sin(start.yaw)
    wx, wy = x * radius, y * radius
    return Pose(start.x + c * wx - s * wy, start.y + s * wx + c * wy, wrap_angle(start.yaw + yaw))
