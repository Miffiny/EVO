import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROUTES_DIR = PROJECT_ROOT / "results" / "single_run_results"
DEFAULT_IMAGES_DIR = PROJECT_ROOT / "data" / "img"

ROUTE_COLOR = (255, 0, 0)
LINE_WIDTH = 5


def dataset_stem(dataset_name: str) -> str:
    return Path(dataset_name).stem


def image_stem_from_dataset(stem: str) -> str:
    return stem.removeprefix("gray_")


def read_route(route_path: Path) -> tuple[str, float, list[tuple[int, int]]]:
    with route_path.open(newline="", encoding="utf-8") as route_file:
        reader = csv.DictReader(route_file)
        rows = sorted(reader, key=lambda row: int(row["route_order"]))

    if not rows:
        raise ValueError(f"Route file is empty: {route_path}")

    dataset = rows[0]["dataset"]
    best_length = float(rows[0]["best_length"])
    points = [(round(float(row["x"])), round(float(row["y"]))) for row in rows]
    return dataset, best_length, points


def find_points_image(dataset: str, images_dir: Path) -> Path:
    stem = image_stem_from_dataset(dataset_stem(dataset))
    candidates = [
        images_dir / f"points_{stem}.png",
        images_dir / f"{dataset_stem(dataset)}.png",
        images_dir / f"gray_{stem}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No point image found for {dataset}. Checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def draw_route(
    route_path: Path,
    images_dir: Path,
    output_dir: Path,
    closed: bool,
) -> Path:
    dataset, best_length, points = read_route(route_path)
    image_path = find_points_image(dataset, images_dir)

    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if len(points) >= 2:
        draw.line(points, fill=ROUTE_COLOR, width=LINE_WIDTH)
        if closed:
            draw.line([points[-1], points[0]], fill=ROUTE_COLOR, width=LINE_WIDTH)

    result = Image.alpha_composite(image, overlay).convert("RGB")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"path_{dataset_stem(dataset)}.png"
    result.save(output_path)

    print(
        f"[path] {route_path.name} -> {output_path} "
        f"| nodes={len(points)} | best_length={best_length:.4f}"
    )
    return output_path


def iter_route_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("single_route_*.csv"))
    raise FileNotFoundError(f"Route path not found: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw single-run MMAS routes over generated point images."
    )
    parser.add_argument(
        "route_path",
        nargs="?",
        type=Path,
        default=DEFAULT_ROUTES_DIR,
        help="Route CSV file or directory with single_route_*.csv files.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DEFAULT_IMAGES_DIR,
        help="Directory with points_<dataset>.png images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ROUTES_DIR,
        help="Directory where rendered path images will be saved.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Do not connect the final route node back to the first node.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_files = iter_route_files(args.route_path)
    if not route_files:
        raise FileNotFoundError(f"No single_route_*.csv files found in {args.route_path}")

    for route_file in route_files:
        draw_route(
            route_path=route_file,
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            closed=not args.open,
        )


if __name__ == "__main__":
    main()
