import json
import importlib
from pathlib import Path

import torch
from nnsight import NNsight
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_1_REPO_ID = "andyrdt/04_2026_puzzle_1a"
MODEL_2_REPO_ID = "andyrdt/04_2026_puzzle_1b"


# ── Model 1 vocab (numbers 0-9, each is its own token) ──
NUM_RANGE_1 = 10
BOS_1, SEP_1, ANS_1, EOS_1 = 10, 11, 12, 13
VOCAB_SIZE_1 = 14
TOKEN_NAMES_1 = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}


def tokenize_1(nums: list[int]) -> list[int]:
    """Tokenize a list of numbers for Model 1.
    Example: [3, 7, 2] -> [BOS, 3, SEP, 7, SEP, 2, ANS]"""
    tokens = [BOS_1]
    for i, n in enumerate(nums):
        tokens.append(n)
        if i < len(nums) - 1:
            tokens.append(SEP_1)
    tokens.append(ANS_1)
    return tokens


def token_labels_1(tokens: list[int]) -> list[str]:
    return [TOKEN_NAMES_1.get(t, str(t)) for t in tokens]


# ── Model 2 vocab (digits 0-9, two per number) ──
BOS_2, SEP_2, ANS_2, EOS_2 = 10, 11, 12, 13
VOCAB_SIZE_2 = 14
TOKEN_NAMES_2 = {10: "BOS", 11: "SEP", 12: "ANS", 13: "EOS"}


def tokenize_2(nums: list[int]) -> list[int]:
    """Tokenize a list of numbers for Model 2.
    Example: [42, 7, 85] -> [BOS, 4, 2, SEP, 0, 7, SEP, 8, 5, ANS]"""
    tokens = [BOS_2]
    for i, n in enumerate(nums):
        tokens.append(n // 10)
        tokens.append(n % 10)
        if i < len(nums) - 1:
            tokens.append(SEP_2)
    tokens.append(ANS_2)
    return tokens


def token_labels_2(tokens: list[int]) -> list[str]:
    return [TOKEN_NAMES_2.get(t, str(t)) for t in tokens]


def _load_attention_only_transformer_class(
    repo_id: str = MODEL_1_REPO_ID,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> type:
    model_py_path = hf_hub_download(
        repo_id=repo_id,
        filename="model.py",
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    module_name = f"{repo_id.replace('/', '_')}_model"
    spec = importlib.util.spec_from_file_location(module_name, model_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model definition from {model_py_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AttentionOnlyTransformer


def _download_model_1_artifacts(
    repo_id: str = MODEL_1_REPO_ID,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> dict[str, Path]:
    artifact_names = {
        "config_path": "config.json",
        "weights_path": "model.pt",
    }
    artifacts: dict[str, Path] = {}
    for key, filename in artifact_names.items():
        artifacts[key] = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
    return artifacts


def load_model_1(
    *,
    wrap_nnsight: bool = True,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    repo_id: str = MODEL_1_REPO_ID,
):
    """Load Model 1 and optionally wrap it with NNsight."""

    AttentionOnlyTransformer = _load_attention_only_transformer_class(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    artifacts = _download_model_1_artifacts(
        repo_id=repo_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )

    config = read_json(artifacts["config_path"])
    raw_model = AttentionOnlyTransformer.from_config(config["model"])

    try:
        state_dict = torch.load(
            artifacts["weights_path"],
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            artifacts["weights_path"],
            map_location=device,
        )

    raw_model.load_state_dict(state_dict)
    raw_model.eval().to(device)

    wrapped_model = None
    model = raw_model
    if wrap_nnsight:
        wrapped_model = NNsight(raw_model)
        model = wrapped_model

    return {
        "repo_id": repo_id,
        "device": device,
        "config": config,
        "artifacts": artifacts,
        "raw_model": raw_model,
        "wrapped_model": wrapped_model,
        "model": model,
    }


# DATA

DATA_DIR_MODEL_1 = Path("data")
DATA_DIR_MODEL_2 = Path("data2")

# Backward-compatible default used by the existing model1 notebook/helpers.
DATA_DIR = DATA_DIR_MODEL_1

DATA_DIR_ALIASES = {
    "data": DATA_DIR_MODEL_1,
    "model1": DATA_DIR_MODEL_1,
    "puzzle1": DATA_DIR_MODEL_1,
    "data2": DATA_DIR_MODEL_2,
    "model2": DATA_DIR_MODEL_2,
    "puzzle2": DATA_DIR_MODEL_2,
}


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        return DATA_DIR
    if isinstance(data_dir, Path):
        return data_dir.resolve()

    normalized = data_dir.strip()
    alias_path = DATA_DIR_ALIASES.get(normalized.lower())
    if alias_path is not None:
        return alias_path.resolve()

    return Path(normalized).resolve()


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: str | Path):
    return list(iter_jsonl(path))


def load_data_manifest(data_dir: str | Path | None = None):
    data_path = _resolve_data_dir(data_dir)
    return read_json(data_path / "manifest.json")


def get_final_dataset_path(data_dir: str | Path | None = None) -> Path:
    data_path = _resolve_data_dir(data_dir)
    manifest = load_data_manifest(data_path)
    return data_path / manifest["files"]["final_dataset"]


def get_canonical_probe_dataset_path(data_dir: str | Path | None = None) -> Path:
    data_path = _resolve_data_dir(data_dir)
    manifest = load_data_manifest(data_path)
    return data_path / manifest["files"]["canonical_probe_dataset"]


def get_families_dir(data_dir: str | Path | None = None) -> Path:
    data_path = _resolve_data_dir(data_dir)
    manifest = load_data_manifest(data_path)
    return data_path / manifest["files"]["families_dir"]


def get_pairs_dir(data_dir: str | Path | None = None) -> Path:
    data_path = _resolve_data_dir(data_dir)
    manifest = load_data_manifest(data_path)
    pairs_dir = manifest["files"].get("pairs_dir")
    if pairs_dir is None:
        raise FileNotFoundError(f"No pairs_dir configured in {data_path / 'manifest.json'}")
    return data_path / pairs_dir


def iter_final_dataset(data_dir: str | Path | None = None):
    return iter_jsonl(get_final_dataset_path(data_dir))


def load_final_dataset(data_dir: str | Path | None = None):
    return list(iter_final_dataset(data_dir))


def iter_canonical_probe_dataset(data_dir: str | Path | None = None):
    return iter_jsonl(get_canonical_probe_dataset_path(data_dir))


def load_canonical_probe_dataset(data_dir: str | Path | None = None):
    return list(iter_canonical_probe_dataset(data_dir))


def get_family_path(
    family_name: str,
    data_dir: str | Path | None = None,
) -> Path:
    families_dir = get_families_dir(data_dir)
    family_filename = family_name if family_name.endswith(".json") else f"{family_name}.json"
    family_path = families_dir / family_filename
    if not family_path.exists():
        raise FileNotFoundError(f"Unknown family dataset: {family_name}")
    return family_path


def load_family_index(
    family_name: str,
    data_dir: str | Path | None = None,
):
    return read_json(get_family_path(family_name, data_dir))


def load_family_row_ids(
    family_name: str,
    data_dir: str | Path | None = None,
):
    family_index = load_family_index(family_name, data_dir)
    return family_index["row_ids"]


def _sorted_unique_row_ids(row_ids):
    row_ids = list(row_ids)
    unique_ids = sorted(set(row_ids))
    if len(unique_ids) != len(row_ids):
        raise ValueError("row_ids must not contain duplicates")
    return unique_ids


def iter_rows_by_id(row_ids, data_dir: str | Path | None = None):
    target_ids = _sorted_unique_row_ids(row_ids)
    if not target_ids:
        return

    target_iter = iter(target_ids)
    current_target = next(target_iter, None)

    for row in iter_final_dataset(data_dir):
        row_id = row["row_id"]
        while current_target is not None and row_id > current_target:
            raise ValueError(f"row_id {current_target} not found in final dataset")
        if current_target is None:
            break
        if row_id == current_target:
            yield row
            current_target = next(target_iter, None)
            if current_target is None:
                break

    if current_target is not None:
        raise ValueError(f"row_id {current_target} not found in final dataset")


def load_rows_by_id(row_ids, data_dir: str | Path | None = None):
    return list(iter_rows_by_id(row_ids, data_dir))


def iter_family_dataset(
    family_name: str,
    data_dir: str | Path | None = None,
):
    row_ids = load_family_row_ids(family_name, data_dir)
    return iter_rows_by_id(row_ids, data_dir)


def load_family_dataset(
    family_name: str,
    data_dir: str | Path | None = None,
):
    return list(iter_family_dataset(family_name, data_dir))


def get_pair_dataset_path(
    pair_dataset_name: str,
    data_dir: str | Path | None = None,
) -> Path:
    pairs_dir = get_pairs_dir(data_dir)
    pair_filename = (
        pair_dataset_name
        if pair_dataset_name.endswith(".jsonl")
        else f"{pair_dataset_name}.jsonl"
    )
    pair_path = pairs_dir / pair_filename
    if not pair_path.exists():
        raise FileNotFoundError(f"Unknown pair dataset: {pair_dataset_name}")
    return pair_path


def iter_pair_dataset(
    pair_dataset_name: str,
    data_dir: str | Path | None = None,
):
    return iter_jsonl(get_pair_dataset_path(pair_dataset_name, data_dir))


def load_pair_dataset(
    pair_dataset_name: str,
    data_dir: str | Path | None = None,
):
    return list(iter_pair_dataset(pair_dataset_name, data_dir))
