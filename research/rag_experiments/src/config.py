from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

HYDRA_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
BASE_DIR = Path(__file__).resolve().parents[3]
RAG_EXP_DIR = Path(__file__).resolve().parents[1]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_cfg(overrides: list[str] | None = None) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(HYDRA_CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides or []))


def get_collection_name(cfg: DictConfig) -> str:
    return str(OmegaConf.select(cfg, "qdrant.collection_name", default="boardgame_rules_chunk512"))


def paths_from_cfg(cfg: DictConfig) -> dict[str, Path]:
    manifest = OmegaConf.select(cfg, "data.manifest_path", default="manifests/index_manifest.csv")
    data_dir = OmegaConf.select(cfg, "data.data_dir", default="data/rules_texts_cleaned_good")
    eval_dir = OmegaConf.select(cfg, "data.eval_datasets_dir", default="data/eval")
    return {
        "base_dir": BASE_DIR,
        "rag_dir": RAG_EXP_DIR,
        "manifest_path": BASE_DIR / str(manifest),
        "data_dir": BASE_DIR / str(data_dir),
        "eval_datasets_dir": RAG_EXP_DIR / str(eval_dir),
    }
