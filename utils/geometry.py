
import numpy as np


def toroidal_distance(p1: np.ndarray, p2: np.ndarray, w: int, h: int) -> float:
    dx = abs(p1[0] - p2[0])
    dy = abs(p1[1] - p2[1])
    dx = min(dx, w - dx)
    dy = min(dy, h - dy)
    return np.sqrt(dx * dx + dy * dy)


def toroidal_offset(pos: np.ndarray, offset: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.array([(pos[0] + offset[0]) % w, (pos[1] + offset[1]) % h])


def cone_cells(
    position: np.ndarray,
    heading: float,
    cone_width: float,
    max_range: int,
    resolution: int,
    w: int,
    h: int,
) -> list[tuple[int, int, float]]:
    cells = []
    if resolution <= 0 or max_range <= 0:
        return cells

    half_cone = cone_width / 2.0
    if resolution == 1:
        angles = [heading]
    else:
        angles = [
            heading - half_cone + i * cone_width / (resolution - 1)
            for i in range(resolution)
        ]

    for angle in angles:
        dx = np.cos(angle)
        dy = np.sin(angle)
        for dist in range(1, max_range + 1):
            cx = int(round(position[0] + dx * dist)) % w
            cy = int(round(position[1] + dy * dist)) % h
            cells.append((cx, cy, float(dist)))

    return cells


def normalize_angle(angle: float) -> float:
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle
