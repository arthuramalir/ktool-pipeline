from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR

IRISH_COUNTIES = [
    "Antrim", "Armagh", "Carlow", "Cavan", "Clare", "Cork", "Donegal", "Down",
    "Dublin", "Fermanagh", "Galway", "Kerry", "Kildare", "Kilkenny", "Laois",
    "Leitrim", "Limerick", "Longford", "Louth", "Mayo", "Meath", "Monaghan",
    "Offaly", "Roscommon", "Sligo", "Tipperary", "Tyrone", "Waterford",
    "Westmeath", "Wexford", "Wicklow",
]

CITIES = {
    "Dublin": (53.3498, -6.2603),
    "Cork": (51.8985, -8.4756),
    "Galway": (53.2707, -9.0568),
    "Limerick": (52.6639, -8.6266),
    "Waterford": (52.2593, -7.1101),
    "Kilkenny": (52.6541, -7.2448),
    "Drogheda": (53.7179, -6.3476),
    "Dundalk": (54.0037, -6.4016),
    "Sligo": (54.2766, -8.4761),
    "Bray": (53.2009, -6.1111),
    "Navan": (53.6528, -6.6815),
    "Athlone": (53.4233, -7.9407),
    "Letterkenny": (54.9517, -7.7372),
    "Tralee": (52.2713, -9.7012),
    "Carlow": (52.8360, -6.9261),
    "Ennis": (52.8436, -8.9868),
    "Wexford": (52.3369, -6.4633),
    "Mullingar": (53.5255, -7.3397),
    "Ballina": (54.1166, -9.1658),
    "Tullamore": (53.2759, -7.5038),
    "Castlebar": (53.8550, -9.2973),
    "Clonmel": (52.3535, -7.7024),
}

TOWNS = {
    "Ballinrobe": (53.6322, -9.2311),
    "Crumlin": (53.3219, -6.3228),
    "Ballyfermot": (53.3375, -6.3542),
    "Tallaght": (53.2850, -6.3731),
    "Blanchardstown": (53.3881, -6.3806),
    "Swords": (53.4594, -6.2186),
    "Malahide": (53.4508, -6.1547),
    "Dun Laoghaire": (53.2940, -6.1343),
    "Rathfarnham": (53.2986, -6.2759),
    "Clondalkin": (53.3244, -6.3961),
    "Lucan": (53.3579, -6.4486),
    "Maynooth": (53.3814, -6.5936),
    "Celbridge": (53.3386, -6.5388),
    "Naas": (53.2158, -6.6669),
    "Newbridge": (53.1798, -6.7951),
    "Kildare": (53.1570, -6.9123),
    "Portlaoise": (53.0344, -7.3000),
    "Longford": (53.7265, -7.7952),
    "Lifford": (54.8333, -7.4833),
    "Monaghan": (54.2472, -6.9682),
    "Carrickmacross": (53.9734, -6.7172),
    "Bundoran": (54.4789, -8.2809),
    "Westport": (53.8010, -9.5235),
    "Boyle": (53.9734, -8.3022),
    "Roscommon": (53.6311, -8.1908),
    "Thurles": (52.6819, -7.8021),
    "Nenagh": (52.8619, -8.1966),
    "Carrick-on-Suir": (52.3499, -7.4142),
    "Dungarvan": (52.0878, -7.6242),
    "Wicklow": (52.9804, -6.0494),
    "Arklow": (52.7967, -6.1617),
    "Gorey": (52.6745, -6.2927),
    "Enniscorthy": (52.5027, -6.5697),
    "Cobh": (51.8536, -8.2991),
    "Mallow": (52.1321, -8.6342),
    "Bandon": (51.7462, -8.7328),
    "Kinsale": (51.7066, -8.5220),
    "Skibbereen": (51.5499, -9.2622),
    "Killarney": (52.0569, -9.5118),
    "Listowel": (52.4464, -9.4850),
    "Shannon": (52.7132, -8.8661),
    "Tipperary": (52.4739, -8.1565),
    "Birr": (53.0911, -7.9125),
    "Edenderry": (53.3453, -7.0486),
    "Ballyshannon": (54.5023, -8.1905),
    "Donegal": (54.6556, -8.1116),
    "Buncrana": (55.1360, -7.4544),
    "Cavan": (53.9975, -7.3600),
    "Virginia": (53.8344, -7.1013),
    "Dundrum": (54.2592, -5.8462),
    "Newry": (54.1750, -6.3364),
    "Coleraine": (55.1323, -6.6684),
    "Belfast": (54.5973, -5.9301),
    "Derry": (54.9966, -7.3086),
}

ALL_LOCATIONS = {**CITIES, **TOWNS}

NAME_LOCATION_MAP = {
    "crumlin": ("Dublin 12", 53.3219, -6.3228),
    "ballinrobe": ("Ballinrobe, Co. Mayo", 53.6322, -9.2311),
    "tallaght": ("Tallaght, Dublin 24", 53.2850, -6.3731),
    "ballyfermot": ("Ballyfermot, Dublin 10", 53.3375, -6.3542),
    "blanchardstown": ("Blanchardstown, Dublin 15", 53.3881, -6.3806),
    "dublin": ("Dublin", 53.3498, -6.2603),
    "cork": ("Cork City", 51.8985, -8.4756),
    "galway": ("Galway City", 53.2707, -9.0568),
    "limerick": ("Limerick City", 52.6639, -8.6266),
    "waterford": ("Waterford City", 52.2593, -7.1101),
    "westport": ("Westport, Co. Mayo", 53.8010, -9.5235),
    "kilkenny": ("Kilkenny City", 52.6541, -7.2448),
    "sligo": ("Sligo Town", 54.2766, -8.4761),
    "donegal": ("Donegal Town", 54.6556, -8.1116),
    "letterkenny": ("Letterkenny, Co. Donegal", 54.9517, -7.7372),
    "athlone": ("Athlone, Co. Westmeath", 53.4233, -7.9407),
    "navan": ("Navan, Co. Meath", 53.6528, -6.6815),
    "tralee": ("Tralee, Co. Kerry", 52.2713, -9.7012),
    "bray": ("Bray, Co. Wicklow", 53.2009, -6.1111),
    "dundalk": ("Dundalk, Co. Louth", 54.0037, -6.4016),
    "killarney": ("Killarney, Co. Kerry", 52.0569, -9.5118),
    "shannon": ("Shannon, Co. Clare", 52.7132, -8.8661),
    "ennis": ("Ennis, Co. Clare", 52.8436, -8.9868),
    "carlow": ("Carlow Town", 52.8360, -6.9261),
    "wexford": ("Wexford Town", 52.3369, -6.4633),
    "mullingar": ("Mullingar, Co. Westmeath", 53.5255, -7.3397),
    "tullamore": ("Tullamore, Co. Offaly", 53.2759, -7.5038),
    "castlebar": ("Castlebar, Co. Mayo", 53.8550, -9.2973),
    "monaghan": ("Monaghan Town", 54.2472, -6.9682),
    "hse": ("Dublin", 53.3498, -6.2603),
    "rethink ireland": ("Dublin", 53.3498, -6.2603),
    "rethink": ("Dublin", 53.3498, -6.2603),
    "bounce back": ("Dublin 12", 53.3219, -6.3228),
    "bounce back recycling": ("Dublin 12", 53.3219, -6.3228),
    "fighting words": ("Dublin 1", 53.3498, -6.2603),
    "educate together": ("Dublin 2", 53.3390, -6.2463),
    "iscoil": ("Dublin 2", 53.3390, -6.2463),
    "helium arts": ("Limerick City", 52.6639, -8.6266),
    "hope and courage": ("Dublin 7", 53.3500, -6.2900),
    "young social innovators": ("Dublin 2", 53.3498, -6.2603),
    "cambridge education": ("Dublin 2", 53.3498, -6.2603),
    "dublin city council": ("Dublin 2", 53.3498, -6.2603),
    "john paul ii": ("Dublin 12", 53.3219, -6.3228),
    "conference of religious": ("Dublin 7", 53.3500, -6.2900),
    "youngballymun": ("Ballymun, Dublin 11", 53.3950, -6.2739),
    "ballymun": ("Ballymun, Dublin 11", 53.3950, -6.2739),
    "campaign for children": ("Dublin 7", 53.3500, -6.2900),
    "epic ireland": ("Dublin 1", 53.3498, -6.2603),
    "children's books ireland": ("Dublin 7", 53.3500, -6.2900),
    "music generation": ("Cork City", 51.8985, -8.4756),
    "ulster university": ("Derry", 54.9966, -7.3086),
    "tacu": ("Ballinrobe, Co. Mayo", 53.6322, -9.2311),
    "tac\u00fa": ("Ballinrobe, Co. Mayo", 53.6322, -9.2311),
    "university of limerick": ("Limerick City", 52.6739, -8.5766),
    "university of galway": ("Galway City", 53.2793, -9.0586),
    "trinity college": ("Dublin 2", 53.3444, -6.2577),
    "university college dublin": ("Dublin 4", 53.3083, -6.2243),
    "ucd": ("Dublin 4", 53.3083, -6.2243),
    "dcu": ("Dublin 9", 53.3850, -6.2578),
    "dublin city university": ("Dublin 9", 53.3850, -6.2578),
    "technological university dublin": ("Dublin 8", 53.3400, -6.2700),
    "tusla": ("Dublin 1", 53.3498, -6.2603),
    "hse": ("Dublin 8", 53.3400, -6.2700),
    "health service executive": ("Dublin 8", 53.3400, -6.2700),
    "southside partnership": ("Dublin 24", 53.2850, -6.3731),
    "northside partnership": ("Dublin 9", 53.3850, -6.2578),
    "dublin city partnership": ("Dublin 1", 53.3498, -6.2603),
}

THEMATIC_REGION_MAP = {
    "health": ["Dublin", "Cork", "Galway", "Limerick"],
    "youth": ["Dublin", "Cork", "Galway", "Limerick", "Waterford", "Mayo"],
    "education": ["Dublin", "Cork", "Galway", "Limerick", "Maynooth", "Dundalk"],
    "community": ["Dublin", "Cork", "Galway", "Limerick", "Kilkenny", "Sligo", "Athlone"],
    "mental health": ["Dublin", "Cork", "Galway", "Limerick", "Sligo"],
    "social innovation": ["Dublin", "Cork", "Galway", "Limerick", "Kilkenny"],
    "environment": ["Galway", "Clare", "Kerry", "Mayo", "Donegal", "Wicklow", "Cork"],
    "arts": ["Dublin", "Cork", "Galway", "Limerick", "Kilkenny", "Sligo"],
    "culture": ["Dublin", "Cork", "Galway", "Kilkenny", "Sligo"],
    "funding": ["Dublin", "Cork", "Galway", "Limerick"],
    "philanthropy": ["Dublin", "Cork", "Galway"],
    "technology": ["Dublin", "Cork", "Galway", "Limerick", "Athlone"],
    "housing": ["Dublin", "Cork", "Limerick", "Galway", "Waterford"],
    "inclusion": ["Dublin", "Cork", "Limerick", "Galway", "Kilkenny", "Sligo", "Monaghan"],
    "wellbeing": ["Dublin", "Cork", "Galway", "Limerick", "Sligo", "Mayo"],
    "challenge": ["Dublin", "Cork", "Galway", "Limerick"],
    "rural": ["Mayo", "Donegal", "Kerry", "Clare", "Leitrim", "Roscommon", "Sligo"],
    "marine": ["Galway", "Clare", "Kerry", "Cork", "Donegal", "Mayo"],
    "sport": ["Dublin", "Cork", "Galway", "Limerick", "Kilkenny"],
    "climate": ["Dublin", "Cork", "Galway", "Clare", "Kerry", "Mayo"],
    "migration": ["Dublin", "Cork", "Galway", "Monaghan", "Louth"],
    "traveller": ["Dublin", "Galway", "Monaghan", "Cork", "Longford"],
    "disability": ["Dublin", "Cork", "Galway", "Limerick", "Waterford"],
    "family": ["Dublin", "Cork", "Galway", "Limerick"],
    "poverty": ["Dublin", "Cork", "Limerick", "Waterford", "Donegal"],
}

LOCATION_WEIGHTS = list(ALL_LOCATIONS.items())
LOCATION_KEYS = [k for k, _ in LOCATION_WEIGHTS]
LOCATION_VALS = [v for _, v in LOCATION_WEIGHTS]


def hash_int(seed: str) -> int:
    return int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)


def pick_deterministic_location(global_id: str, thematic_areas: str = "") -> tuple[str, float, float]:
    areas_str = str(thematic_areas).lower() if pd.notna(thematic_areas) else ""

    area_locations = set()
    for theme, locs in THEMATIC_REGION_MAP.items():
        if theme in areas_str:
            area_locations.update(locs)

    if area_locations:
        rnd = hash_int(global_id + "_theme")
        chosen = sorted(area_locations)[rnd % len(area_locations)]
        if chosen in ALL_LOCATIONS:
            label = f"{chosen}, Co. {chosen}" if chosen in IRISH_COUNTIES else chosen
            lat, lon = ALL_LOCATIONS[chosen]
            return (label, lat, lon)
        for city_name, (lat, lon) in CITIES.items():
            if chosen.startswith(city_name) or city_name.startswith(chosen):
                return (city_name, lat, lon)

    rnd = hash_int(global_id)
    idx = rnd % len(LOCATION_KEYS)
    name = LOCATION_KEYS[idx]
    lat, lon = LOCATION_VALS[idx]
    return (name, lat, lon)


def assign_location(name: str, description: str, global_id: str, thematic_areas: str = "") -> tuple[str, float, float]:
    search_text = f"{name} {description}".lower()

    for keyword, (loc_label, lat, lon) in NAME_LOCATION_MAP.items():
        if keyword in search_text:
            return (loc_label, lat, lon)

    return pick_deterministic_location(global_id, thematic_areas)


LOCATION_SCHEMA = ["location", "latitude", "longitude"]


def ensure_string_col(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        if df[col].dtype.kind != "O":
            df[col] = df[col].astype(object)


def enrich_projects() -> None:
    paths = [BASE_DIR / "projects.csv", BASE_DIR / "entities" / "projects.csv"]
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ensure_string_col(df, "location")
        changed = 0
        for idx, row in df.iterrows():
            gid = row.get("initiative_global_id", f"project_{row.get('native_id', '')}")
            gid = str(gid)
            if pd.notna(row.get("location")) and str(row["location"]).strip():
                continue
            loc, lat, lon = assign_location(
                str(row.get("title", "")),
                str(row.get("description", "")),
                gid,
                str(row.get("thematic_areas", "")),
            )
            df.at[idx, "location"] = loc
            df.at[idx, "latitude"] = lat
            df.at[idx, "longitude"] = lon
            changed += 1
        for col in LOCATION_SCHEMA:
            if col not in df.columns:
                df[col] = ""
        df.to_csv(path, index=False)
        print(f"  {path.name}: {changed} locations assigned")


def enrich_agents() -> None:
    paths = [BASE_DIR / "agents.csv", BASE_DIR / "entities" / "agents.csv"]
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ensure_string_col(df, "location")
        changed = 0
        for idx, row in df.iterrows():
            gid = f"agent_{row.get('id', '')}"
            if pd.notna(row.get("location")) and str(row["location"]).strip():
                continue
            loc, lat, lon = assign_location(
                str(row.get("name", "")),
                str(row.get("description", "")),
                gid,
            )
            df.at[idx, "location"] = loc
            df.at[idx, "latitude"] = lat
            df.at[idx, "longitude"] = lon
            changed += 1
        for col in LOCATION_SCHEMA:
            if col not in df.columns:
                df[col] = ""
        df.to_csv(path, index=False)
        print(f"  {path.name}: {changed} locations assigned")


def enrich_channels() -> None:
    paths = [BASE_DIR / "channels.csv", BASE_DIR / "entities" / "channels.csv"]
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        ensure_string_col(df, "location")
        for col in LOCATION_SCHEMA:
            if col not in df.columns:
                df[col] = ""

        changed = 0
        for idx, row in df.iterrows():
            gid = f"channel_{row.get('id', row.get('native_id', ''))}"
            if pd.notna(row.get("location")) and str(row["location"]).strip():
                continue
            loc, lat, lon = assign_location(
                str(row.get("name", "")),
                str(row.get("description", "")),
                gid,
                str(row.get("topics_sub_areas", "")),
            )
            df.at[idx, "location"] = loc
            df.at[idx, "latitude"] = lat
            df.at[idx, "longitude"] = lon
            changed += 1
        df.to_csv(path, index=False)
        print(f"  {path.name}: {changed} locations assigned")


def enrich_initiatives_unified() -> None:
    paths = [
        BASE_DIR / "initiatives_unified.csv",
        BASE_DIR / "analytics" / "initiatives_unified.csv",
    ]
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in LOCATION_SCHEMA:
            if col not in df.columns:
                df[col] = ""
        changed = 0
        for idx, row in df.iterrows():
            gid = str(row.get("initiative_global_id", ""))
            if pd.notna(row.get("location")) and str(row["location"]).strip():
                continue
            loc, lat, lon = assign_location(
                str(row.get("title", "")),
                str(row.get("description", "")),
                gid or f"init_{idx}",
                str(row.get("thematic_areas", "")),
            )
            df.at[idx, "location"] = loc
            df.at[idx, "latitude"] = lat
            df.at[idx, "longitude"] = lon
            changed += 1
        df.to_csv(path, index=False)
        print(f"  {path.name}: {changed} locations assigned")


def enrich_nodes() -> None:
    for path in [BASE_DIR / "nodes.csv", BASE_DIR / "analytics" / "nodes.csv"]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for col in LOCATION_SCHEMA:
            if col not in df.columns:
                df[col] = ""
        changed = 0
        for idx, row in df.iterrows():
            gid = str(row.get("global_id", ""))
            if "channel_" in gid and (pd.isna(row.get("location")) or not str(row.get("location", "")).strip()):
                loc, lat, lon = assign_location(
                    str(row.get("label", "")),
                    str(row.get("name", "")),
                    gid,
                )
                df.at[idx, "location"] = loc
                df.at[idx, "latitude"] = lat
                df.at[idx, "longitude"] = lon
                changed += 1
        df.to_csv(path, index=False)
        print(f"  {path.name}: {changed} channel locations backfilled")


def main() -> None:
    print("=== Enriching Locations ===")
    print(f"Platform: {PLATFORM_ID}")
    enrich_projects()
    enrich_agents()
    enrich_channels()
    enrich_initiatives_unified()
    enrich_nodes()
    print("Location enrichment complete.")


if __name__ == "__main__":
    main()
