from pathlib import Path
import csv
import numpy as np
from PIL import Image, ImageDraw

POINT_COUNT = 256
GAMMA = 2.5
MIN_DENSITY = 0.00

WORK_WIDTH = 512
MIN_DISTANCE_DARK = 5.0
MAX_DISTANCE_LIGHT = 26.0
MAX_CANDIDATE_ATTEMPTS = 250_000
POINT_RADIUS = 4
POINT_COLOR = (0, 255, 0)

K_NEIGHBORS = 200
EPS_DISTANCE = 1e-6
GENERATED_PREFIXES = ("gray_", "points_")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = DATA_DIR / "img"
POINTS_DIR = DATA_DIR / "points"
PRECOMPUTED_DIR = DATA_DIR / "precomputed"

def convert_images_to_grayscale():
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    for path in IMG_DIR.iterdir():
        if not path.is_file():
            continue

        if path.name.startswith(GENERATED_PREFIXES):
            continue

        out_path = IMG_DIR / f"gray_{path.stem}.png"

        img = Image.open(path).convert("L")
        img.save(out_path)

        print(f"[gray] {path.name} -> {out_path.name}")


def load_gray_image(path: Path):
    img = Image.open(path).convert("L")
    original_w, original_h = img.size

    scale = 1.0

    if WORK_WIDTH is not None and original_w > WORK_WIDTH:
        scale = WORK_WIDTH / original_w
        work_h = max(1, round(original_h * scale))
        img = img.resize((WORK_WIDTH, work_h), Image.Resampling.LANCZOS)

    gray = np.asarray(img, dtype=np.uint8)
    return gray, scale


def make_density(gray: np.ndarray):
    density = 1.0 - gray.astype(np.float64) / 255.0
    density = np.power(density, GAMMA)

    if MIN_DENSITY > 0.0:
        density = MIN_DENSITY + (1.0 - MIN_DENSITY) * density

    if density.sum() <= 1e-12:
        density[:] = 1.0

    return density

def make_min_distance(density: np.ndarray):
    darkness = np.ones_like(density) if density.max() <= 1e-12 else density / density.max()
    return MAX_DISTANCE_LIGHT - darkness * (MAX_DISTANCE_LIGHT - MIN_DISTANCE_DARK)

def generate_poisson_points(
    density: np.ndarray,
    min_distance: np.ndarray,
    count: int,
    rng: np.random.Generator,
):
    h, w = density.shape
    flat = density.ravel().astype(np.float64)
    flat /= flat.sum()

    cell_size = MIN_DISTANCE_DARK / np.sqrt(2.0)
    grid_w = int(np.ceil(w / cell_size))
    grid_h = int(np.ceil(h / cell_size))
    search_cells = int(np.ceil(MAX_DISTANCE_LIGHT / cell_size))

    grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
    points = []
    radii = []
    attempts = 0

    while len(points) < count and attempts < MAX_CANDIDATE_ATTEMPTS:
        attempts += 1

        chosen = rng.choice(h * w, p=flat)
        yi, xi = divmod(chosen, w)

        x = xi + rng.random()
        y = yi + rng.random()
        radius = min_distance[yi, xi]

        gx = int(x / cell_size)
        gy = int(y / cell_size)

        accepted = True

        min_gx = max(0, gx - search_cells)
        max_gx = min(grid_w, gx + search_cells + 1)
        min_gy = max(0, gy - search_cells)
        max_gy = min(grid_h, gy + search_cells + 1)

        for ny in range(min_gy, max_gy):
            if not accepted:
                break

            for nx in range(min_gx, max_gx):
                for point_index in grid[ny][nx]:
                    px, py = points[point_index]
                    required = max(radius, radii[point_index])

                    if (x - px) ** 2 + (y - py) ** 2 < required ** 2:
                        accepted = False
                        break

                if not accepted:
                    break

        if accepted:
            grid[gy][gx].append(len(points))
            points.append((x, y))
            radii.append(radius)

    if len(points) < count:
        print(
            f"  warning: generated {len(points)}/{count} points "
            f"after {attempts} attempts"
        )

    return np.asarray(points, dtype=np.float64)


def compute_pairwise_distances(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).astype(np.float32)


def build_nearest_neighbors(
    points: np.ndarray,
    k: int = K_NEIGHBORS,
):
    n = len(points)

    if n < 2:
        raise ValueError("At least 2 points are required")

    if k >= n:
        raise ValueError(f"k={k} must be smaller than point count n={n}")

    full_dist = compute_pairwise_distances(points)

    np.fill_diagonal(full_dist, np.inf)

    unordered = np.argpartition(full_dist, kth=k - 1, axis=1)[:, :k]

    unordered_dist = np.take_along_axis(full_dist, unordered, axis=1)

    order = np.argsort(unordered_dist, axis=1)

    neighbors = np.take_along_axis(unordered, order, axis=1).astype(np.int32)
    distances = np.take_along_axis(unordered_dist, order, axis=1).astype(np.float32)

    heuristic = 1.0 / np.maximum(distances, EPS_DISTANCE)
    heuristic = heuristic.astype(np.float32)

    return neighbors, distances, heuristic


def preprocess_points_for_mmas(
    out_path: Path,
    points: np.ndarray,
    scale: float,
    source_name: str,
    k: int = K_NEIGHBORS,
):
    original_points = points.astype(np.float32, copy=True)
    original_points[:, 0] /= scale
    original_points[:, 1] /= scale

    neighbors, distances, heuristic = build_nearest_neighbors(original_points, k)

    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as f:
        f.write(np.int32(len(original_points)).tobytes())
        f.write(np.int32(neighbors.shape[1]).tobytes())
        f.write(np.float32(scale).tobytes())
        f.write(original_points.astype(np.float32).tobytes())
        f.write(neighbors.astype(np.int32).tobytes())
        f.write(distances.astype(np.float32).tobytes())
        f.write(heuristic.astype(np.float32).tobytes())

    print(f"  saved -> {out_path}")


def save_points_csv(
    out_path: Path,
    points: np.ndarray,
    gray: np.ndarray,
    density: np.ndarray,
    scale: float,
):
    h, w = gray.shape

    POINTS_DIR.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["id", "x", "y", "gray", "density"])

        for i, (x, y) in enumerate(points):
            xi = int(np.clip(round(x), 0, w - 1))
            yi = int(np.clip(round(y), 0, h - 1))

            original_x = x / scale
            original_y = y / scale

            writer.writerow([
                i,
                f"{original_x:.6f}",
                f"{original_y:.6f}",
                int(gray[yi, xi]),
                f"{density[yi, xi]:.8f}",
            ])


def save_points_image(
    out_path: Path,
    image_path: Path,
    points: np.ndarray,
    scale: float,
):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for x, y in points:
        original_x = x / scale
        original_y = y / scale

        draw.ellipse(
            (
                original_x - POINT_RADIUS,
                original_y - POINT_RADIUS,
                original_x + POINT_RADIUS,
                original_y + POINT_RADIUS,
            ),
            fill=POINT_COLOR,
        )

    img.save(out_path)

def generate_points_for_gray_images():
    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    PRECOMPUTED_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng()

    gray_images = sorted(
        path for path in IMG_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".png"
        and path.name.startswith("gray_")
    )

    for path in gray_images:
        print(f"[points] {path.name}")

        gray, scale = load_gray_image(path)
        density = make_density(gray)
        min_distance = make_min_distance(density)

        if POINT_COUNT > gray.size:
            raise ValueError(
                f"POINT_COUNT={POINT_COUNT} is larger than pixel count "
                f"of working image {path.name}: {gray.size}"
            )

        points = generate_poisson_points(
            density=density,
            min_distance=min_distance,
            count=POINT_COUNT,
            rng=rng,
        )

        image_name = path.stem.removeprefix("gray_")

        out_csv = POINTS_DIR / f"{path.stem}.csv"
        out_image = IMG_DIR / f"points_{image_name}.png"
        out_precomputed = PRECOMPUTED_DIR / f"{path.stem}.bin"

        save_points_csv(
            out_path=out_csv,
            points=points,
            gray=gray,
            density=density,
            scale=scale,
        )

        save_points_image(
            out_path=out_image,
            image_path=path,
            points=points,
            scale=scale,
        )

        preprocess_points_for_mmas(
            out_path=out_precomputed,
            points=points,
            scale=scale,
            source_name=path.name,
            k=K_NEIGHBORS,
        )

        print(f"  saved -> {out_csv}")
        print(f"  saved -> {out_image}")


def main():
    convert_images_to_grayscale()
    generate_points_for_gray_images()


if __name__ == "__main__":
    main()
