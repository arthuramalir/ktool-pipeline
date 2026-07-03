from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "data" / "processed" / "173" / "test"
DST_DIR = ROOT / "data" / "processed" / "173_synthetic" / "test"


def hash_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)


def deterministic_budget(global_id: str, initiative_type: str) -> int:
    rnd = hash_int(global_id)
    if initiative_type == "project":
        low, high = 180_000, 4_800_000
    elif initiative_type == "pilot":
        low, high = 70_000, 1_600_000
    else:
        low, high = 25_000, 650_000
    return low + (rnd % (high - low + 1))


def deterministic_investment(global_id: str) -> tuple[str, int]:
    rnd = hash_int(global_id)
    bands = [
        ("low", 15_000, 90_000),
        ("medium", 90_000, 350_000),
        ("high", 350_000, 1_200_000),
        ("very_high", 1_200_000, 4_000_000),
    ]
    band = bands[rnd % len(bands)]
    label = band[0]
    amount = band[1] + (rnd % (band[2] - band[1] + 1))
    return label, amount


def deterministic_impact_level(global_id: str, initiative_type: str, budget: int) -> str:
    rnd = hash_int(f"{global_id}:{initiative_type}:{budget}")
    if initiative_type == "project":
        bands = ["community", "small_medium_scale", "big_scale", "small_medium_scale"]
    elif initiative_type == "pilot":
        bands = ["community", "public_service", "small_medium_scale", "tertiary"]
    else:
        bands = ["regulation", "tertiary", "small_medium_scale", "big_scale"]
    return bands[rnd % len(bands)]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def update_initiative_table(path: Path, initiative_type: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if "initiative_global_id" not in df.columns:
        return

    budgets = []
    impact_levels = []
    for _, row in df.iterrows():
        gid = str(row.get("initiative_global_id", ""))
        inferred_type = str(row.get("initiative_type", initiative_type)).strip().lower() or initiative_type
        budget = deterministic_budget(gid, inferred_type)
        budgets.append(budget)
        current_impact = str(row.get("impact_level", "")).strip()
        impact_levels.append(current_impact if current_impact else deterministic_impact_level(gid, inferred_type, budget))

    df["associated_budget"] = budgets
    df["impact_level"] = impact_levels
    if "investment_level" not in df.columns:
        df["investment_level"] = pd.Series(
            ["high" if b > 800_000 else "medium" if b > 250_000 else "low" for b in budgets]
        )
    write_csv(df, path)


def update_agents_table(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    id_col = "id" if "id" in df.columns else None
    if id_col is None:
        return

    levels = []
    amounts = []
    for _, row in df.iterrows():
        gid = f"agent_{row.get(id_col)}"
        level, amount = deterministic_investment(gid)
        levels.append(level)
        amounts.append(amount)

    df["investment"] = levels
    df["investment_eur_estimate"] = amounts
    write_csv(df, path)


def update_unified_table(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if "initiative_global_id" not in df.columns:
        return

    budgets = []
    impact_levels = []
    levels = []
    for _, row in df.iterrows():
        gid = str(row.get("initiative_global_id", ""))
        initiative_type = str(row.get("initiative_type", "project")).strip().lower() or "project"
        budget = deterministic_budget(gid, initiative_type)
        budgets.append(budget)
        current_impact = str(row.get("impact_level", "")).strip()
        impact_levels.append(current_impact if current_impact else deterministic_impact_level(gid, initiative_type, budget))
        levels.append("high" if budget > 800_000 else "medium" if budget > 250_000 else "low")

    df["associated_budget"] = budgets
    df["impact_level"] = impact_levels
    df["investment_level"] = levels
    write_csv(df, path)


def update_nodes_table(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    df = pd.read_csv(path)
    if not {"global_id", "node_type"}.issubset(df.columns):
        return

    impact_levels = []
    for _, row in df.iterrows():
        node_type = str(row.get("node_type", "")).strip().lower()
        current_impact = str(row.get("impact_level", "")).strip()
        if current_impact:
            impact_levels.append(current_impact)
            continue
        if node_type in {"project", "pilot", "prototype"}:
            budget = int(float(row.get("associated_budget", 0) or 0))
            impact_levels.append(deterministic_impact_level(str(row.get("global_id", "")), node_type, budget))
        else:
            impact_levels.append("")

    df["impact_level"] = impact_levels
    write_csv(df, path)


def copy_source_to_synthetic() -> None:
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Missing source dataset: {SRC_DIR}")

    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    shutil.copytree(SRC_DIR, DST_DIR)


def generate_synthetic_dataset() -> None:
    copy_source_to_synthetic()

    for table_name, initiative_type in [
        ("projects.csv", "project"),
        ("pilots.csv", "pilot"),
        ("prototypes.csv", "prototype"),
    ]:
        update_initiative_table(DST_DIR / table_name, initiative_type)
        update_initiative_table(DST_DIR / "entities" / table_name, initiative_type)

    update_unified_table(DST_DIR / "initiatives_unified.csv")
    update_unified_table(DST_DIR / "analytics" / "initiatives_unified.csv")

    update_nodes_table(DST_DIR / "nodes.csv")
    update_nodes_table(DST_DIR / "analytics" / "nodes.csv")

    update_agents_table(DST_DIR / "agents.csv")
    update_agents_table(DST_DIR / "entities" / "agents.csv")

    print("Synthetic dataset generated successfully.")
    print(f"Source: {SRC_DIR}")
    print(f"Target: {DST_DIR}")


if __name__ == "__main__":
    generate_synthetic_dataset()
