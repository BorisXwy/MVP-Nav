"""Dataset path helpers for Habitat navigation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


DEFAULT_SHAREDATA_ROOT = Path("sharedata")
DEFAULT_LOCAL_DATA_ROOT = Path("data")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    dataset: str
    version: str
    relative_dir: Path
    archive_name: str
    url: str

    def data_path(self, split: str) -> Path:
        return self.relative_dir / split / f"{split}.json.gz"


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "objectnav_hm3d": DatasetSpec(
        name="objectnav_hm3d",
        task="objectnav",
        dataset="hm3d",
        version="v2",
        relative_dir=Path("datasets/objectnav/hm3d/v2"),
        archive_name="objectnav_hm3d_v2.zip",
        url="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2.zip",
    ),
    "objectnav_mp3d": DatasetSpec(
        name="objectnav_mp3d",
        task="objectnav",
        dataset="mp3d",
        version="v1",
        relative_dir=Path("datasets/objectnav/mp3d/v1"),
        archive_name="objectnav_mp3d_v1.zip",
        # Historical Habitat link uses "m3d" in the URL, but the extracted dataset
        # belongs under the standard "mp3d" directory.
        url="https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip",
    ),
    "instance_imagenav_hm3d": DatasetSpec(
        name="instance_imagenav_hm3d",
        task="instance_imagenav",
        dataset="hm3d",
        version="v3",
        relative_dir=Path("datasets/instance_imagenav/hm3d/v3"),
        archive_name="instance_imagenav_hm3d_v3.zip",
        url="https://dl.fbaipublicfiles.com/habitat/data/datasets/imagenav/hm3d/v3/instance_imagenav_hm3d_v3.zip",
    ),
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown navigation dataset '{name}'. Known: {known}") from exc


def resolve_data_root(data_root: Optional[str] = None) -> Path:
    if data_root:
        return Path(data_root).expanduser().resolve()
    share_root = DEFAULT_SHAREDATA_ROOT
    if share_root.exists():
        return share_root
    return DEFAULT_LOCAL_DATA_ROOT.resolve()


def dataset_dir(spec: DatasetSpec, data_root: Optional[str] = None) -> Path:
    return resolve_data_root(data_root) / spec.relative_dir


def dataset_json_path(spec: DatasetSpec, split: str, data_root: Optional[str] = None) -> Path:
    return resolve_data_root(data_root) / spec.data_path(split)


def verify_dataset(spec: DatasetSpec, splits: Iterable[str], data_root: Optional[str] = None) -> Dict[str, Path]:
    missing = []
    found = {}
    for split in splits:
        path = dataset_json_path(spec, split, data_root)
        if path.exists():
            found[split] = path
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing navigation dataset files:\n"
            + "\n".join(f"  - {p}" for p in missing)
            + f"\nDownload {spec.archive_name} from {spec.url}"
        )
    return found


def scene_root(dataset: str, scene_data_root: Optional[str] = None) -> Path:
    root = Path(scene_data_root).expanduser().resolve() if scene_data_root else (DEFAULT_LOCAL_DATA_ROOT / "scene_datasets").resolve()
    if dataset == "hm3d":
        hm3d_v02 = root / "hm3d_v0.2"
        return hm3d_v02 if hm3d_v02.exists() else root / "hm3d"
    if dataset == "mp3d":
        return root / "mp3d"
    return root / dataset
