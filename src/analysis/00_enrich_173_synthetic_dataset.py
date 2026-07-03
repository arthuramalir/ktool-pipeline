from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "173_synthetic")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR


SECTOR_CHOICES = [
    "community",
    "academy_and_education",
    "public_administration",
    "health",
    "environment",
    "arts_and_culture",
    "housing",
    "technology",
]
PROFESSION_CHOICES = [
    "programme_manager",
    "community_builder",
    "researcher",
    "project_lead",
    "facilitator",
    "policy_maker",
]
GENDER_CHOICES = ["female", "male", "nonbinary", "unspecified"]
AGE_CHOICES = ["18-24", "25-34", "35-44", "45-54", "55-64"]
ACCESS_CHOICES = ["open", "restricted"]
POWER_CHOICES = ["low", "medium", "high"]
PRIORITY_CHOICES = ["low", "medium", "high"]
PARTNER_CHOICES = [
    "community",
    "academy_and_education",
    "public_administration",
    "health",
    "environment",
    "arts_and_culture",
    "housing",
    "technology",
]


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


def is_blank(value) -> bool:
    return pd.isna(value) or not str(value).strip() or str(value).strip().lower() == "nan"


def deterministic_date(seed: str) -> str:
    year = 2025
    month = (hash_int(seed) % 12) + 1
    day = (hash_int(f"{seed}:day") % 27) + 1
    return f"{year:04d}-{month:02d}-{day:02d}"


def deterministic_pick(seed: str, pool: list[str], count: int) -> list[str]:
    cleaned = [str(item).strip() for item in pool if str(item).strip()]
    if not cleaned or count <= 0:
        return []
    ordered = sorted(cleaned, key=lambda item: hash_int(f"{seed}:{item}"))
    return ordered[: min(count, len(ordered))]


def parse_list(value) -> list[str]:
    if is_blank(value):
        return []
    value_str = str(value).strip()
    if value_str.startswith("[") and value_str.endswith("]"):
        try:
            parsed = json.loads(value_str)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception:
            pass
    parts = []
    for chunk in value_str.replace("|", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return parts


def dump_list(values: list[str]) -> str:
    return json.dumps([str(value) for value in values if str(value).strip()], ensure_ascii=False)


def join_values(values: list[str]) -> str:
    return " | ".join([str(value) for value in values if str(value).strip()])


def load_df(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_pools(nodes_df: pd.DataFrame) -> dict[str, list[str] | dict[str, str]]:
    node_types = nodes_df.get("node_type", pd.Series(index=nodes_df.index, dtype=str)).fillna("").astype(str).str.lower()
    ids = nodes_df.get("global_id", pd.Series(index=nodes_df.index, dtype=str)).fillna("").astype(str)
    labels = nodes_df.get("label", pd.Series(index=nodes_df.index, dtype=str)).fillna("").astype(str)

    def collect(kind: str) -> tuple[list[str], dict[str, str]]:
        mask = node_types.eq(kind)
        kind_ids = ids[mask].tolist()
        kind_labels = {gid: label for gid, label in zip(ids[mask], labels[mask], strict=False)}
        return kind_ids, kind_labels

    agent_ids, agent_labels = collect("agent")
    project_ids, project_labels = collect("project")
    pilot_ids, pilot_labels = collect("pilot")
    prototype_ids, prototype_labels = collect("prototype")
    perception_ids, perception_labels = collect("perception")
    theme_ids, theme_labels = collect("theme")
    channel_ids, channel_labels = collect("channel")
    value_ids, value_labels = collect("value")
    challenge_ids, challenge_labels = collect("challenge")
    information_ids, information_labels = collect("information")

    return {
        "agent_ids": agent_ids,
        "agent_labels": agent_labels,
        "initiative_ids": project_ids + pilot_ids + prototype_ids,
        "initiative_labels": {**project_labels, **pilot_labels, **prototype_labels},
        "perception_ids": perception_ids,
        "perception_labels": perception_labels,
        "theme_ids": theme_ids,
        "theme_labels": theme_labels,
        "channel_ids": channel_ids,
        "channel_labels": channel_labels,
        "value_ids": value_ids,
        "value_labels": value_labels,
        "challenge_ids": challenge_ids,
        "challenge_labels": challenge_labels,
        "information_ids": information_ids,
        "information_labels": information_labels,
    }


def deterministic_sector(seed: str) -> str:
    return SECTOR_CHOICES[hash_int(seed) % len(SECTOR_CHOICES)]


def deterministic_contact(seed: str, label: str) -> str:
    slug = "".join(ch if ch.isalnum() else "-" for ch in f"{label or seed}".lower()).strip("-")
    return f"https://synthetic.local/{slug or seed}"


def deterministic_description(title: str, sector: str, initiative_type: str) -> str:
    base = title or f"Synthetic {initiative_type} initiative"
    return f"{base} operates in the {sector} sector as part of the synthetic experiment set for prototype-project analysis."


def deterministic_quote(title: str, sector: str, theme_names: list[str], value_names: list[str]) -> str:
    theme_part = " and ".join(theme_names[:2]) if theme_names else sector
    value_part = ", ".join(value_names[:2]) if value_names else "community impact"
    base = title or "This initiative"
    return f"{base} speaks about {theme_part} with emphasis on {value_part}."


def backfill_initiative_table(df: pd.DataFrame, initiative_type: str, pools: dict[str, list[str] | dict[str, str]]) -> pd.DataFrame:
    if df.empty or "initiative_global_id" not in df.columns:
        return df

    agent_ids = pools["agent_ids"]
    agent_labels = pools["agent_labels"]
    initiative_ids = pools["initiative_ids"]
    initiative_labels = pools["initiative_labels"]
    perception_ids = pools["perception_ids"]
    perception_labels = pools["perception_labels"]
    theme_ids = pools["theme_ids"]
    theme_labels = pools["theme_labels"]
    value_ids = pools["value_ids"]
    value_labels = pools["value_labels"]

    rows = []
    for _, row in df.iterrows():
        row = row.copy()
        gid = str(row.get("initiative_global_id", "")).strip()
        title = str(row.get("title", "")).strip()
        native_id = str(row.get("native_id", "")).strip()
        inferred_type = str(row.get("initiative_type", initiative_type)).strip().lower() or initiative_type
        budget = deterministic_budget(gid or native_id, inferred_type)

        lead_ids = parse_list(row.get("lead_agent_ids"))
        agent_list = parse_list(row.get("agent_ids"))
        perception_list = parse_list(row.get("perception_ids"))
        topic_list = parse_list(row.get("topic_ids"))
        indicator_list = parse_list(row.get("indicator_ids"))
        interconnections = parse_list(row.get("interconnections_json"))
        partner_blob = str(row.get("partners_json", "")).strip()

        if not lead_ids:
            lead_ids = deterministic_pick(f"{gid}:lead", agent_ids, 1)
        if not agent_list:
            agent_list = deterministic_pick(f"{gid}:agents", agent_ids, 3)
        if not perception_list:
            perception_list = deterministic_pick(f"{gid}:perceptions", perception_ids, 2)
        if not topic_list:
            topic_list = deterministic_pick(f"{gid}:topics", theme_ids, 2)
        if not indicator_list:
            indicator_list = deterministic_pick(f"{gid}:values", value_ids, 2)
        if not interconnections:
            interconnections = [target for target in deterministic_pick(f"{gid}:links", initiative_ids, 2) if target != gid]
        if is_blank(partner_blob):
            partner_blob = dump_list(deterministic_pick(f"{gid}:partners", PARTNER_CHOICES, 2))

        sector = str(row.get("sector", "")).strip() or deterministic_sector(gid)
        description = str(row.get("description", "")).strip() or deterministic_description(title, sector, inferred_type)
        date = str(row.get("date", "")).strip() or deterministic_date(gid)

        topics_named = [theme_labels.get(topic_id, topic_id) for topic_id in topic_list]
        agents_named = [agent_labels.get(agent_id, agent_id) for agent_id in agent_list]
        lead_named = [agent_labels.get(agent_id, agent_id) for agent_id in lead_ids]
        perceptions_named = [perception_labels.get(perception_id, perception_id) for perception_id in perception_list]
        indicators_named = [value_labels.get(indicator_id, indicator_id) for indicator_id in indicator_list]

        row["associated_budget"] = budget
        row["impact_level"] = str(row.get("impact_level", "")).strip() or deterministic_impact_level(gid or native_id, inferred_type, budget)
        row["investment_level"] = str(row.get("investment_level", "")).strip() or ("high" if budget > 800_000 else "medium" if budget > 250_000 else "low")
        row["sector"] = sector
        row["description"] = description
        row["date"] = date
        row["lead_agent_ids"] = dump_list(lead_ids)
        row["agent_ids"] = dump_list(agent_list)
        row["perception_ids"] = dump_list(perception_list)
        row["topic_ids"] = dump_list(topic_list)
        row["indicator_ids"] = dump_list(indicator_list)
        row["partners_json"] = partner_blob
        row["interconnections_json"] = dump_list(interconnections)
        row["lead_agents"] = join_values(lead_named)
        row["agents"] = join_values(agents_named)
        row["perceptions"] = join_values(perceptions_named)
        row["thematic_areas"] = join_values(topics_named)
        row["topics_thematic_areas"] = join_values(topics_named)
        row["indicators"] = join_values(indicators_named)
        row["status"] = str(row.get("status", "")).strip() or "Unknown"
        rows.append(row)

    return pd.DataFrame(rows)


def backfill_agents_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "id" not in df.columns:
        return df

    rows = []
    for _, row in df.iterrows():
        row = row.copy()
        gid = f"agent_{row.get('id')}"
        label = str(row.get("name", row.get("title", gid))).strip()
        row["investment"] = str(row.get("investment", "")).strip() or deterministic_investment(gid)[0]
        row["investment_eur_estimate"] = int(row.get("investment_eur_estimate", 0) or deterministic_investment(gid)[1])
        row["contact"] = str(row.get("contact", "")).strip() or deterministic_contact(gid, label)
        row["type_other"] = str(row.get("type_other", "")).strip() or ["NGO", "institution", "civil_society", "social_enterprise", "public_body"][hash_int(gid) % 5]
        row["profession"] = str(row.get("profession", "")).strip() or PROFESSION_CHOICES[hash_int(f"{gid}:profession") % len(PROFESSION_CHOICES)]
        row["gender"] = str(row.get("gender", "")).strip() or GENDER_CHOICES[hash_int(f"{gid}:gender") % len(GENDER_CHOICES)]
        row["age"] = str(row.get("age", "")).strip() or AGE_CHOICES[hash_int(f"{gid}:age") % len(AGE_CHOICES)]
        row["sector"] = str(row.get("sector", "")).strip() or deterministic_sector(f"{gid}:sector")
        row["people_involved"] = str(row.get("people_involved", "")).strip() or ("from_1_to_10" if hash_int(gid) % 2 == 0 else "from_10_to_50")
        row["source"] = str(row.get("source", "")).strip() or "synthetic_agents"
        row["priority"] = str(row.get("priority", "")).strip() or PRIORITY_CHOICES[hash_int(f"{gid}:priority") % len(PRIORITY_CHOICES)]
        rows.append(row)

    return pd.DataFrame(rows)


def backfill_information_table(df: pd.DataFrame, pools: dict[str, list[str] | dict[str, str]]) -> pd.DataFrame:
    if df.empty or not ("global_id" in df.columns or "native_id" in df.columns):
        return df

    channel_ids = pools["channel_ids"]
    channel_labels = pools["channel_labels"]
    theme_ids = pools["theme_ids"]
    theme_labels = pools["theme_labels"]
    value_ids = pools["value_ids"]
    perception_ids = pools["perception_ids"]

    rows = []
    for _, row in df.iterrows():
        row = row.copy()
        gid = str(row.get("global_id", row.get("native_id", ""))).strip()
        title = str(row.get("title", row.get("name", ""))).strip()
        quote = str(row.get("quote", "")).strip()
        description = str(row.get("description", "")).strip()
        channel_id = str(row.get("channel_id", "")).strip()
        channel_code = str(row.get("channel_code", "")).strip()
        topics_sub = parse_list(row.get("topics_sub_areas"))
        topics_theme = parse_list(row.get("topics_thematic_areas"))
        values = parse_list(row.get("values"))
        patterns = parse_list(row.get("information_pattern_connections_json"))

        if is_blank(channel_id) and channel_ids:
            channel_id = deterministic_pick(f"{gid}:channel", channel_ids, 1)[0]
        if is_blank(channel_code) and channel_id:
            channel_code = channel_labels.get(channel_id, f"Channel {channel_id}")
        if is_blank(quote):
            theme_names = [theme_labels.get(topic_id, topic_id) for topic_id in deterministic_pick(f"{gid}:theme_quote", theme_ids, 2)]
            value_names = [str(v) for v in deterministic_pick(f"{gid}:values", value_ids, 2)]
            quote = deterministic_quote(title, str(row.get("sector", "")).strip() or "ecosystem", theme_names, value_names)
        if is_blank(description):
            description = quote
        if not topics_sub:
            topics_sub = deterministic_pick(f"{gid}:topics_sub", theme_ids, 2)
        if not topics_theme:
            topics_theme = [theme_labels.get(topic_id, topic_id) for topic_id in topics_sub]
        if not values:
            values = deterministic_pick(f"{gid}:values_fallback", value_ids, 2)
        if not patterns:
            patterns = [f"pattern_{(hash_int(gid) % 97) + 1}"]

        row["channel_id"] = channel_id
        row["channel_code"] = channel_code
        row["channel_name"] = channel_code or "synthetic_channel"
        row["quote"] = quote
        row["description"] = description
        row["information_volume"] = str(row.get("information_volume", "")).strip() or ("high" if len(quote) > 160 else "medium" if len(quote) > 90 else "low")
        row["listening_channel"] = str(row.get("listening_channel", "")).strip() or (channel_code or "synthetic_channel")
        row["channel_accesibility"] = str(row.get("channel_accesibility", "")).strip() or ACCESS_CHOICES[hash_int(f"{gid}:access") % len(ACCESS_CHOICES)]
        row["power_dynamic"] = str(row.get("power_dynamic", "")).strip() or POWER_CHOICES[hash_int(f"{gid}:power") % len(POWER_CHOICES)]
        row["derived_from_information_channel"] = str(row.get("derived_from_information_channel", "")).strip() or (channel_code or "synthetic_channel")
        row["values"] = join_values(values)
        row["topics_sub_areas"] = join_values(topics_sub)
        row["topics_thematic_areas"] = join_values(topics_theme)
        row["information_pattern_connections_json"] = dump_list(patterns)
        row["source"] = str(row.get("source", "")).strip() or "synthetic_listening"
        row["code"] = str(row.get("code", "")).strip() or f"INFO-{gid}"
        row["priority"] = str(row.get("priority", "")).strip() or PRIORITY_CHOICES[hash_int(f"{gid}:priority") % len(PRIORITY_CHOICES)]
        rows.append(row)

    return pd.DataFrame(rows)


def backfill_nodes_table(df: pd.DataFrame, pools: dict[str, list[str] | dict[str, str]]) -> pd.DataFrame:
    if df.empty or "global_id" not in df.columns:
        return df

    rows = []
    for _, row in df.iterrows():
        row = row.copy()
        node_type = str(row.get("node_type", "")).strip().lower()
        gid = str(row.get("global_id", "")).strip()
        label = str(row.get("label", row.get("name", gid))).strip()

        if node_type in {"project", "pilot", "prototype"}:
            budget = deterministic_budget(gid, node_type)
            row["associated_budget"] = int(float(row.get("associated_budget", 0) or budget))
            row["impact_level"] = str(row.get("impact_level", "")).strip() or deterministic_impact_level(gid, node_type, int(row["associated_budget"]))
            row["investment_level"] = str(row.get("investment_level", "")).strip() or ("high" if int(row["associated_budget"]) > 800_000 else "medium" if int(row["associated_budget"]) > 250_000 else "low")
            row["sector"] = str(row.get("sector", "")).strip() or deterministic_sector(gid)
            row["description"] = str(row.get("description", "")).strip() or deterministic_description(label, row["sector"], node_type)
            row["date"] = str(row.get("date", "")).strip() or deterministic_date(gid)
            row["lead_agent_ids"] = str(row.get("lead_agent_ids", "")).strip() or dump_list(deterministic_pick(f"{gid}:lead", pools["agent_ids"], 1))
            row["agent_ids"] = str(row.get("agent_ids", "")).strip() or dump_list(deterministic_pick(f"{gid}:agents", pools["agent_ids"], 3))
            row["perception_ids"] = str(row.get("perception_ids", "")).strip() or dump_list(deterministic_pick(f"{gid}:perceptions", pools["perception_ids"], 2))
            row["topic_ids"] = str(row.get("topic_ids", "")).strip() or dump_list(deterministic_pick(f"{gid}:topics", pools["theme_ids"], 2))
            row["partners_json"] = str(row.get("partners_json", "")).strip() or dump_list(deterministic_pick(f"{gid}:partners", PARTNER_CHOICES, 2))
            row["interconnections_json"] = str(row.get("interconnections_json", "")).strip() or dump_list([x for x in deterministic_pick(f"{gid}:links", pools["initiative_ids"], 2) if x != gid])
            row["agents"] = str(row.get("agents", "")).strip() or join_values([pools["agent_labels"].get(x, x) for x in parse_list(row["agent_ids"] )])
            row["lead_agents"] = str(row.get("lead_agents", "")).strip() or join_values([pools["agent_labels"].get(x, x) for x in parse_list(row["lead_agent_ids"] )])
            row["perceptions"] = str(row.get("perceptions", "")).strip() or join_values([pools["perception_labels"].get(x, x) for x in parse_list(row["perception_ids"] )])
            row["topics_thematic_areas"] = str(row.get("topics_thematic_areas", "")).strip() or join_values([pools["theme_labels"].get(x, x) for x in parse_list(row["topic_ids"] )])
            row["thematic_areas"] = str(row.get("thematic_areas", "")).strip() or row["topics_thematic_areas"]
        elif node_type == "agent":
            row["investment"] = str(row.get("investment", "")).strip() or deterministic_investment(gid)[0]
            row["contact"] = str(row.get("contact", "")).strip() or deterministic_contact(gid, label)
            row["type_other"] = str(row.get("type_other", "")).strip() or ["NGO", "institution", "civil_society", "social_enterprise", "public_body"][hash_int(gid) % 5]
            row["profession"] = str(row.get("profession", "")).strip() or PROFESSION_CHOICES[hash_int(f"{gid}:profession") % len(PROFESSION_CHOICES)]
            row["gender"] = str(row.get("gender", "")).strip() or GENDER_CHOICES[hash_int(f"{gid}:gender") % len(GENDER_CHOICES)]
            row["age"] = str(row.get("age", "")).strip() or AGE_CHOICES[hash_int(f"{gid}:age") % len(AGE_CHOICES)]
            row["sector"] = str(row.get("sector", "")).strip() or deterministic_sector(f"{gid}:sector")
            row["people_involved"] = str(row.get("people_involved", "")).strip() or ("from_1_to_10" if hash_int(gid) % 2 == 0 else "from_10_to_50")
            row["source"] = str(row.get("source", "")).strip() or "synthetic_agents"
            row["priority"] = str(row.get("priority", "")).strip() or PRIORITY_CHOICES[hash_int(f"{gid}:priority") % len(PRIORITY_CHOICES)]
        elif node_type == "information":
            quote = str(row.get("quote", "")).strip()
            title = str(row.get("title", row.get("name", gid))).strip()
            row["channel_id"] = str(row.get("channel_id", "")).strip() or (deterministic_pick(f"{gid}:channel", pools["channel_ids"], 1)[0] if pools["channel_ids"] else "")
            if is_blank(row.get("channel_code")) and row.get("channel_id"):
                row["channel_code"] = pools["channel_labels"].get(str(row["channel_id"]), f"Channel {row['channel_id']}")
            row["quote"] = quote or deterministic_quote(title, "ecosystem", [pools["theme_labels"].get(x, x) for x in deterministic_pick(f"{gid}:theme", pools["theme_ids"], 2)], [str(x) for x in deterministic_pick(f"{gid}:values", pools["value_ids"], 2)])
            row["description"] = str(row.get("description", "")).strip() or row["quote"]
            row["information_volume"] = str(row.get("information_volume", "")).strip() or ("high" if len(row["quote"]) > 160 else "medium" if len(row["quote"]) > 90 else "low")
            row["listening_channel"] = str(row.get("listening_channel", "")).strip() or row.get("channel_code", "synthetic_channel")
            row["channel_accesibility"] = str(row.get("channel_accesibility", "")).strip() or ACCESS_CHOICES[hash_int(f"{gid}:access") % len(ACCESS_CHOICES)]
            row["power_dynamic"] = str(row.get("power_dynamic", "")).strip() or POWER_CHOICES[hash_int(f"{gid}:power") % len(POWER_CHOICES)]
            row["derived_from_information_channel"] = str(row.get("derived_from_information_channel", "")).strip() or row.get("channel_code", "synthetic_channel")
            row["source"] = str(row.get("source", "")).strip() or "synthetic_listening"
            row["code"] = str(row.get("code", "")).strip() or f"INFO-{gid}"
            row["priority"] = str(row.get("priority", "")).strip() or PRIORITY_CHOICES[hash_int(f"{gid}:priority") % len(PRIORITY_CHOICES)]
            if is_blank(row.get("topics_sub_areas")):
                row["topics_sub_areas"] = join_values(deterministic_pick(f"{gid}:topics_sub", pools["theme_ids"], 2))
            if is_blank(row.get("topics_thematic_areas")):
                row["topics_thematic_areas"] = row["topics_sub_areas"]
            if is_blank(row.get("values")):
                row["values"] = join_values([str(x) for x in deterministic_pick(f"{gid}:values", pools["value_ids"], 2)])
            if is_blank(row.get("information_pattern_connections_json")):
                row["information_pattern_connections_json"] = dump_list([f"pattern_{(hash_int(f'{gid}:pattern') % 97) + 1}"])
        rows.append(row)

    return pd.DataFrame(rows)


def backfill_edges_table(df: pd.DataFrame, node_lookup: dict[str, pd.Series]) -> pd.DataFrame:
    if df.empty or not {"source_global_id", "target_global_id"}.issubset(df.columns):
        return df

    rows = []
    for _, row in df.iterrows():
        row = row.copy()
        source_id = str(row.get("source_global_id", "")).strip()
        target_id = str(row.get("target_global_id", "")).strip()
        source_node = node_lookup.get(source_id)
        target_node = node_lookup.get(target_id)
        source_type = str(source_node.get("node_type") if source_node is not None else "").strip().lower()
        target_type = str(target_node.get("node_type") if target_node is not None else "").strip().lower()
        edge_type = str(row.get("edge_type", "")).strip()
        edge_family = str(row.get("edge_family", "")).strip()

        if is_blank(row.get("target_kind")):
            row["target_kind"] = target_type or (target_id.split("_", 1)[0] if "_" in target_id else "unknown")
        if is_blank(row.get("connection_type")):
            if edge_type in {"declared_interconnection", "initiative_has_agent", "initiative_has_lead_agent"}:
                row["connection_type"] = "mapping"
            elif edge_family in {"listening", "interpretive", "qualitative_narrative"}:
                row["connection_type"] = "semantic"
            else:
                row["connection_type"] = edge_type or edge_family or "unknown"
        if is_blank(row.get("weight")) or str(row.get("weight", "")).strip().lower() == "unknown":
            row["weight"] = 2.0 if row["connection_type"] == "semantic" else 1.0 if row["connection_type"] == "mapping" else 0.75
        if is_blank(row.get("source_agent_id")) and source_type == "agent":
            row["source_agent_id"] = source_id.split("_", 1)[1] if "_" in source_id else source_id
        if is_blank(row.get("source_initiative_global_id")) and source_type in {"project", "pilot", "prototype"}:
            row["source_initiative_global_id"] = source_id
        if is_blank(row.get("information_id")) and source_type == "information":
            row["information_id"] = source_id.split("_", 1)[1] if "_" in source_id else source_id
        if is_blank(row.get("value_id")) and target_type == "value":
            row["value_id"] = target_id.split("_", 1)[1] if "_" in target_id else target_id
        if is_blank(row.get("perception_id")) and target_type == "perception":
            row["perception_id"] = target_id.split("_", 1)[1] if "_" in target_id else target_id
        if is_blank(row.get("challenge_id")) and target_type == "challenge":
            row["challenge_id"] = target_id.split("_", 1)[1] if "_" in target_id else target_id
        if is_blank(row.get("challenge_name")) and not is_blank(row.get("challenge_id")):
            row["challenge_name"] = f"challenge_{row['challenge_id']}"
        if is_blank(row.get("parameter")):
            row["parameter"] = edge_type or row.get("connection_type", "relation")
        if is_blank(row.get("narrative_coalition")):
            row["narrative_coalition"] = "Yes" if row.get("connection_type") in {"semantic", "listening"} else "Potential"
        if is_blank(row.get("description")):
            source_label = str(source_node.get("label") if source_node is not None else source_id).strip() or source_id
            target_label = str(target_node.get("label") if target_node is not None else target_id).strip() or target_id
            row["description"] = f"Synthetic {edge_type or row.get('connection_type', 'relation')} link from {source_label} to {target_label}."
        if is_blank(row.get("channel_name")) and source_type == "information":
            row["channel_name"] = str(source_node.get("channel_code") if source_node is not None else "").strip() or "synthetic_channel"
        if is_blank(row.get("generated_by")):
            row["generated_by"] = "00_enrich_173_synthetic_dataset.py"
        if is_blank(row.get("inference_method")):
            row["inference_method"] = "deterministic_table_backfill"
        if is_blank(row.get("alc_semantic_parameter")) and row.get("connection_type") in {"semantic", "listening"}:
            row["alc_semantic_parameter"] = row.get("parameter", edge_type or row.get("connection_type", "relation"))
        rows.append(row)

    return pd.DataFrame(rows)


def update_file(path: Path, transform) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    df = load_df(path)
    if df.empty:
        return
    updated = transform(df)
    if not updated.empty:
        save_df(updated, path)


def main() -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Missing synthetic dataset directory: {BASE_DIR}")

    nodes_path = BASE_DIR / "nodes.csv"
    nodes_df = load_df(nodes_path)
    pools = build_pools(nodes_df)
    node_lookup = {str(row["global_id"]): row for _, row in nodes_df.iterrows()} if not nodes_df.empty and "global_id" in nodes_df.columns else {}

    initiative_files = [
        BASE_DIR / "projects.csv",
        BASE_DIR / "pilots.csv",
        BASE_DIR / "prototypes.csv",
        BASE_DIR / "initiatives_unified.csv",
        BASE_DIR / "analytics" / "initiatives_unified.csv",
        BASE_DIR / "entities" / "projects.csv",
        BASE_DIR / "entities" / "pilots.csv",
        BASE_DIR / "entities" / "prototypes.csv",
    ]
    agent_files = [BASE_DIR / "agents.csv", BASE_DIR / "entities" / "agents.csv"]
    information_files = [BASE_DIR / "informations.csv", BASE_DIR / "entities" / "informations.csv"]
    node_files = [BASE_DIR / "nodes.csv", BASE_DIR / "analytics" / "nodes.csv"]
    edge_files = [BASE_DIR / "edges.csv", BASE_DIR / "analytics" / "edges.csv"]

    for path in initiative_files:
        update_file(path, lambda df, path=path: backfill_initiative_table(df, path.stem.rstrip("s"), pools))
    for path in agent_files:
        update_file(path, backfill_agents_table)
    for path in information_files:
        update_file(path, lambda df: backfill_information_table(df, pools))

    # Refresh nodes after the source tables have been enriched.
    nodes_df = load_df(nodes_path)
    node_lookup = {str(row["global_id"]): row for _, row in nodes_df.iterrows()} if not nodes_df.empty and "global_id" in nodes_df.columns else {}
    for path in node_files:
        update_file(path, lambda df: backfill_nodes_table(df, pools))

    nodes_df = load_df(nodes_path)
    node_lookup = {str(row["global_id"]): row for _, row in nodes_df.iterrows()} if not nodes_df.empty and "global_id" in nodes_df.columns else {}
    for path in edge_files:
        update_file(path, lambda df, lookup=node_lookup: backfill_edges_table(df, lookup))

    print("Synthetic enrichment complete.")
    print(f"Dataset: {BASE_DIR}")


if __name__ == "__main__":
    main()
