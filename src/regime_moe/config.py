"""Experiment configuration.

Defaults live in ``src/configs/defaults/<section>.toml`` (file name = section name).
An experiment is one TOML file that overrides any keys of any section.
The dataclasses below define structure and types only; all values come from the
TOML files. Unknown or missing keys raise an error.
"""

import dataclasses
import tomllib
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "src" / "configs"
RESULTS_DIR = ROOT / "results"
DEFAULTS_DIR = CONFIG_DIR / "defaults"


@dataclass(frozen=True)
class DataConfig:
    source: str
    assets: list[str]
    cash: str
    warmup_start: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str


@dataclass(frozen=True)
class SyntheticConfig:
    n_days: int
    seeds: list[int]
    train_fraction: float
    noise_feature: bool
    transition: list[list[float]]
    means: list[list[float]]
    vols: list[float]


@dataclass(frozen=True)
class FeaturesConfig:
    window: int


@dataclass(frozen=True)
class HMMConfig:
    n_states: int
    covariance_type: str
    n_iter: int
    seed: int


@dataclass(frozen=True)
class LearnedGateConfig:
    hidden: int


@dataclass(frozen=True)
class RouterConfig:
    types: list[str]
    hmm: HMMConfig
    learned: LearnedGateConfig


@dataclass(frozen=True)
class MLPConfig:
    hidden: list[int]


@dataclass(frozen=True)
class ModelConfig:
    n_experts: int
    expert: str
    mlp: MLPConfig


@dataclass(frozen=True)
class TrainConfig:
    objective: str
    risk_aversion: float
    epochs: int
    batch_size: int
    learning_rate: float
    seeds: list[int]


@dataclass(frozen=True)
class EvalConfig:
    split: str
    periods_per_year: int


@dataclass(frozen=True)
class RunConfig:
    mode: str


@dataclass(frozen=True)
class Config:
    name: str
    run: RunConfig
    data: DataConfig
    synthetic: SyntheticConfig
    features: FeaturesConfig
    router: RouterConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


SECTIONS = [f.name for f in dataclasses.fields(Config) if f.name != "name"]


def load_config(experiment: str | Path | None = None) -> Config:
    """Load all defaults and apply one experiment file on top (None = defaults only)."""
    raw = _load_defaults()
    name = "defaults"
    if experiment is not None:
        path = Path(experiment)
        with path.open("rb") as f:
            raw = _merge(raw, tomllib.load(f), where=path.name)
        name = path.stem
    return _build(Config, {"name": name, **raw}, "config")


def _load_defaults() -> dict[str, Any]:
    files = {p.stem: p for p in DEFAULTS_DIR.glob("*.toml")}
    if unknown := sorted(set(files) - set(SECTIONS)):
        raise ValueError(f"defaults files without a config section: {unknown}")
    if missing := sorted(set(SECTIONS) - set(files)):
        raise ValueError(f"missing defaults files: {[m + '.toml' for m in missing]}")
    raw = {}
    for section, path in files.items():
        with path.open("rb") as f:
            raw[section] = tomllib.load(f)
    return raw


def _merge(base: dict, override: dict, where: str) -> dict:
    out = dict(base)
    for key, value in override.items():
        if key not in base:
            raise KeyError(f"{where}: unknown key '{key}'")
        if isinstance(base[key], dict) and isinstance(value, dict):
            out[key] = _merge(base[key], value, f"{where}.{key}")
        else:
            out[key] = value
    return out


def _build(cls: type, values: dict, where: str):
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    if unknown := sorted(set(values) - names):
        raise KeyError(f"{where}: unknown keys {unknown}")
    if missing := sorted(names - set(values)):
        raise KeyError(f"{where}: missing keys {missing}")
    kwargs = {}
    for key, hint in hints.items():
        value = values[key]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise TypeError(f"{where}.{key}: expected a table")
            kwargs[key] = _build(hint, value, f"{where}.{key}")
        else:
            kwargs[key] = _check_type(value, hint, f"{where}.{key}")
    return cls(**kwargs)


def _check_type(value: Any, hint: Any, where: str) -> Any:
    origin = typing.get_origin(hint) or hint
    if hint is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, origin) or (origin is int and isinstance(value, bool)):
        raise TypeError(f"{where}: expected {hint}, got {type(value).__name__}")
    return value
