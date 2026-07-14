from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
YEAR_RANGE = list(range(2019, 2026))
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR

THEMATIC_TO_VALUE_DIMENSION: dict[str, str] = {
    "youth": "social_justice",
    "youth support": "social_justice",
    "youth engagement": "social_justice",
    "education": "social_justice",
    "education & youth empowerment": "social_justice",
    "health & wellbeing": "evidence_based",
    "mental health": "evidence_based",
    "social innovation": "innovation_drive",
    "innovation": "innovation_drive",
    "philanthropy": "collaboration",
    "philanthropy & funding": "collaboration",
    "funding": "collaboration",
    "collaboration": "collaboration",
    "community & inclusion": "social_justice",
    "inclusion": "social_justice",
    "community": "community_autonomy",
    "challenges": "innovation_drive",
    "culture": "cultural_identity",
    "arts": "cultural_identity",
    "heritage": "cultural_identity",
    "environment": "innovation_drive",
    "climate": "innovation_drive",
    "rural": "community_autonomy",
    "marine": "community_autonomy",
    "sport": "collaboration",
    "migration": "social_justice",
    "traveller": "social_justice",
    "disability": "social_justice",
    "family": "social_justice",
    "poverty": "social_justice",
    "wellbeing": "evidence_based",
    "housing": "social_justice",
    "technology": "innovation_drive",
    "economy": "innovation_drive",
}

LOCATION_TO_COUNTY: dict[str, str] = {
    "Athlone": "Westmeath",
    "Ballinrobe, Co. Mayo": "Mayo",
    "Ballyfermot": "Dublin",
    "Ballyshannon": "Donegal",
    "Bandon": "Cork",
    "Belfast": "Antrim",
    "Birr": "Offaly",
    "Blanchardstown": "Dublin",
    "Boyle": "Roscommon",
    "Bray": "Wicklow",
    "Buncrana": "Donegal",
    "Bundoran": "Donegal",
    "Carlow": "Carlow",
    "Carrick-on-Suir": "Tipperary",
    "Castlebar": "Mayo",
    "Cavan": "Cavan",
    "Celbridge": "Kildare",
    "Clondalkin": "Dublin",
    "Cobh": "Cork",
    "Coleraine": "Derry",
    "Cork": "Cork",
    "Cork, Co. Cork": "Cork",
    "Crumlin": "Dublin",
    "Derry": "Derry",
    "Donegal": "Donegal",
    "Donegal Town": "Donegal",
    "Drogheda": "Louth",
    "Dublin": "Dublin",
    "Dublin 1": "Dublin",
    "Dublin 12": "Dublin",
    "Dublin 2": "Dublin",
    "Dublin 7": "Dublin",
    "Dublin 8": "Dublin",
    "Dublin 9": "Dublin",
    "Dublin, Co. Dublin": "Dublin",
    "Dun Laoghaire": "Dublin",
    "Dundalk": "Louth",
    "Dundrum": "Down",
    "Dungarvan": "Waterford",
    "Edenderry": "Offaly",
    "Ennis": "Clare",
    "Enniscorthy": "Wexford",
    "Galway": "Galway",
    "Galway City": "Galway",
    "Galway, Co. Galway": "Galway",
    "Gorey": "Wexford",
    "Kildare": "Kildare",
    "Kilkenny": "Kilkenny",
    "Kilkenny, Co. Kilkenny": "Kilkenny",
    "Killarney": "Kerry",
    "Kinsale": "Cork",
    "Letterkenny": "Donegal",
    "Limerick": "Limerick",
    "Limerick City": "Limerick",
    "Limerick, Co. Limerick": "Limerick",
    "Listowel": "Kerry",
    "Longford": "Longford",
    "Lucan": "Dublin",
    "Malahide": "Dublin",
    "Maynooth": "Kildare",
    "Monaghan": "Monaghan",
    "Monaghan, Co. Monaghan": "Monaghan",
    "Navan": "Meath",
    "Nenagh": "Tipperary",
    "Newbridge": "Kildare",
    "Portlaoise": "Laois",
    "Rathfarnham": "Dublin",
    "Roscommon": "Roscommon",
    "Shannon": "Clare",
    "Skibbereen": "Cork",
    "Sligo": "Sligo",
    "Sligo, Co. Sligo": "Sligo",
    "Swords": "Dublin",
    "Tallaght": "Dublin",
    "Tallaght, Dublin 24": "Dublin",
    "Tralee": "Kerry",
    "Tralee, Co. Kerry": "Kerry",
    "Tullamore": "Offaly",
    "Waterford": "Waterford",
    "Waterford City": "Waterford",
    "Waterford, Co. Waterford": "Waterford",
    "Westport": "Mayo",
    "Wexford": "Wexford",
    "Wicklow": "Wicklow",
}


def thematic_to_value_dimension(thematic_areas: str) -> str | None:
    if pd.isna(thematic_areas) or not str(thematic_areas).strip():
        return None
    text = str(thematic_areas).lower().strip()
    parts = [p.strip() for p in re.split(r"[|,/]", text)]
    for part in parts:
        if part in THEMATIC_TO_VALUE_DIMENSION:
            return THEMATIC_TO_VALUE_DIMENSION[part]
        for key, vd in THEMATIC_TO_VALUE_DIMENSION.items():
            if key in part:
                return vd
    return None


def location_to_county(location: str) -> str | None:
    if pd.isna(location) or not str(location).strip():
        return None
    loc = str(location).strip()
    if loc in LOCATION_TO_COUNTY:
        return LOCATION_TO_COUNTY[loc]
    for loc_key, county in LOCATION_TO_COUNTY.items():
        if loc_key in loc or loc in loc_key:
            return county
    return None


def hash_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)


def pick_sim_year(global_id: str) -> int:
    h = hash_int(global_id + "_simyear")
    return YEAR_RANGE[h % len(YEAR_RANGE)]


def enrich_nodes(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)

    vd_backfilled = 0
    county_backfilled = 0
    sim_year_assigned = 0

    if "value_dimension" not in df.columns:
        df["value_dimension"] = None

    if "county" not in df.columns:
        df["county"] = None

    if "sim_year" not in df.columns:
        df["sim_year"] = None

    for idx, row in df.iterrows():
        gid = str(row.get("global_id", ""))
        ta = row.get("thematic_areas")
        loc = row.get("location")

        if pd.isna(row.get("value_dimension")) or not str(row.get("value_dimension", "")).strip():
            vd = thematic_to_value_dimension(ta)
            if vd:
                df.at[idx, "value_dimension"] = vd
                vd_backfilled += 1

        if pd.isna(row.get("county")) or not str(row.get("county", "")).strip():
            county = location_to_county(loc)
            if county:
                df.at[idx, "county"] = county
                county_backfilled += 1

        if pd.isna(row.get("sim_year")) or not str(row.get("sim_year", "")).strip():
            if gid:
                df.at[idx, "sim_year"] = pick_sim_year(gid)
                sim_year_assigned += 1

    df.to_csv(path, index=False)
    print(f"  {path.name}: {vd_backfilled} vd, {county_backfilled} counties, {sim_year_assigned} sim_years")


def main() -> None:
    print("=== Enriching Value Dimensions & Counties ===")
    print(f"Platform: {PLATFORM_ID}")
    for path in [
        BASE_DIR / "nodes.csv",
        BASE_DIR / "analytics" / "nodes.csv",
    ]:
        enrich_nodes(path)
    print("Value dimension & county enrichment complete.")


if __name__ == "__main__":
    main()
