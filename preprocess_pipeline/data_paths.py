import os
from pathlib import Path


def _looks_like_data_root(path: Path) -> bool:
    return all(
        (path / required_name).exists()
        for required_name in ("RawData", "parametersY", "Cleaned_Data", "Separated_Data")
    )


def _resolve_data_root() -> Path:
    env_root = os.environ.get("TFM_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser()

    default_root = Path("/mnt/temp_drive")
    if _looks_like_data_root(default_root):
        return default_root

    nested_root = default_root / "Separated_Data"
    if _looks_like_data_root(nested_root):
        return nested_root

    return default_root


DATA_ROOT = _resolve_data_root()
CLASS_FILE_TAG = os.environ.get("TFM_CLASS_FILE_TAG", "")

RAW_DATA_DIR = DATA_ROOT / "RawData"
PARAMETERS_DIR = DATA_ROOT / "parametersY"
CLEANED_DATA_DIR = DATA_ROOT / "Cleaned_Data"
CLEANED_META_DIR = CLEANED_DATA_DIR / "meta"
CLEANED_STRUCTURED_DIR = CLEANED_DATA_DIR / "structured"
SEPARATED_DATA_DIR = DATA_ROOT / "Separated_Data"
SEPARATED_CLASSES_DIR = SEPARATED_DATA_DIR / "classes"

CLASS_FILE_NAMES = {
    "PRECISION_BIMANUAL_45": "precision_bimanual_45_degrees.npy",
    "PRECISION_BIMANUAL_135": "precision_bimanual_135_degrees.npy",
    "PRECISION_BIMANUAL_45_135": "precision_bimanual_45_135_degrees.npy",
    "PRECISION_BIMANUAL_135_45": "precision_bimanual_135_45_degrees.npy",
    "PRECISION_UNIMANUAL_R_0": "precision_unimanual_right_0_degrees.npy",
    "PRECISION_UNIMANUAL_R_45": "precision_unimanual_right_45_degrees.npy",
    "PRECISION_UNIMANUAL_R_90": "precision_unimanual_right_90_degrees.npy",
    "PRECISION_UNIMANUAL_R_135": "precision_unimanual_right_135_degrees.npy",
    "PRECISION_UNIMANUAL_L_0": "precision_unimanual_left_0_degrees.npy",
    "PRECISION_UNIMANUAL_L_45": "precision_unimanual_left_45_degrees.npy",
    "PRECISION_UNIMANUAL_L_90": "precision_unimanual_left_90_degrees.npy",
    "PRECISION_UNIMANUAL_L_135": "precision_unimanual_left_135_degrees.npy",
    "POWER_UNIMANUAL_R_0": "power_unimanual_right_0_degrees.npy",
    "POWER_UNIMANUAL_R_45": "power_unimanual_right_45_degrees.npy",
    "POWER_UNIMANUAL_R_90": "power_unimanual_right_90_degrees.npy",
    "POWER_UNIMANUAL_R_135": "power_unimanual_right_135_degrees.npy",
    "POWER_UNIMANUAL_L_0": "power_unimanual_left_0_degrees.npy",
    "POWER_UNIMANUAL_L_45": "power_unimanual_left_45_degrees.npy",
    "POWER_UNIMANUAL_L_90": "power_unimanual_left_90_degrees.npy",
    "POWER_UNIMANUAL_L_135": "power_unimanual_left_135_degrees.npy",
}


def with_class_tag(file_name: str, tag: str = CLASS_FILE_TAG) -> str:
    if not tag:
        return file_name
    stem, suffix = os.path.splitext(file_name)
    return f"{stem}{tag}{suffix}"


CLASS_FILES = {
    class_name: os.fspath(SEPARATED_CLASSES_DIR / with_class_tag(file_name))
    for class_name, file_name in CLASS_FILE_NAMES.items()
}
