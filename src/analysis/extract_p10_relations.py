"""Re-extract missing relational data for Platform 10 (Chile COPOLAD).
The main pipeline uses platform 173's field names which don't match p10's schema.

Problems fixed:
  - Initiative populate: thematic_areas → topics_thematic_areas
  - Initiative populate: lead_agent (singular) → lead_agents (plural)
  - New: extract information→challenge links from challenge_opportunities relation
  - New: extract project interconnections (project→prototype links)

Output: missing link tables in relationships/ directory.
"""

import os, json, sys
from pathlib import Path
import pandas as pd
import requests

AUTH_TOKEN = os.environ.get("KTOOL_AUTH_TOKEN")
if not AUTH_TOKEN:
    raise ValueError("KTOOL_AUTH_TOKEN required")

PLATFORM_ID = os.environ.get("KTOOL_PLATFORM_ID", "10")
OUTPUT_SUBDIR = os.environ.get("KTOOL_OUTPUT_SUBDIR", "test")
BASE_URL = "https://ktool.agirrecenter.eus/api"
HEADERS = {"Authorization": f"Bearer {AUTH_TOKEN}", "Accept": "application/json"}

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / PLATFORM_ID / OUTPUT_SUBDIR
ENTITIES_DIR = DATA_DIR / "entities"
RELS_DIR = DATA_DIR / "relationships"
ANALYTICS_DIR = DATA_DIR / "analytics"

# Platform 10 correct deep-populate params
# Using the actual schema field names (not platform 173's names)
# Simpler populate — no nested interconnections populate (was breaking the query)
FULL_INIT_POPULATE_P10 = (
    "&populate[0]=perceptions"
    "&populate[1]=topics_thematic_areas"
    "&populate[2]=partners"
    "&populate[3]=lead_agents"
    "&populate[4]=agents"
    "&populate[5]=interconnections"
    "&populate[6]=indicators"
    "&populate[7]=attachments"
)

# Information populate to get challenge_opportunities and pattern links
FULL_INFO_POPULATE_P10 = (
    "&populate[0]=challenge_opportunities"
    "&populate[1]=information_pattern_connections"
    "&populate[2]=values"
    "&populate[3]=tags"
    "&populate[4]=topics_thematic_areas"
    "&populate[5]=topics_sub_areas"
)


def fetch_all(endpoint, populate="", page_size=100):
    records = []
    page = 1
    sep = "&" if "?" in endpoint else "?"
    while True:
        url = f"{BASE_URL}{endpoint}{sep}pagination[page]={page}&pagination[pageSize]={page_size}{populate}"
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} on page {page}")
            break
        data = r.json().get("data", [])
        if not data:
            break
        records.extend(data)
        page += 1
    return records


def get_relation_ids(record, field):
    """Extract IDs from a populated Strapi relation field."""
    val = record.get("attributes", {}).get(field)
    if not val:
        return []
    data = val.get("data", [])
    if not data:
        return []
    if isinstance(data, list):
        return [item.get("id") for item in data if item]
    if isinstance(data, dict):
        return [data.get("id")] if data.get("id") else []
    return []


def get_single_relation_id(record, field):
    """Extract a single relation ID."""
    ids = get_relation_ids(record, field)
    return ids[0] if ids else None


# ============================================================
# 1. RE-EXTRACT INITIATIVES WITH CORRECT POPULATE
# ============================================================
def extract_initiative_relations():
    print("\n=== 1. Initiative relations (correct field names) ===")

    for init_type in ["projects", "pilots", "protots"]:
        print(f"\n  Fetching {init_type}...")
        records = fetch_all(f"/{init_type}?filters[parent_platform][id]={PLATFORM_ID}", FULL_INIT_POPULATE_P10)
        print(f"  Got {len(records)} records")

        lead_agent_links = []
        agent_links = []
        theme_links = []
        interconnection_links = []
        perception_links = []

        for rec in records:
            rid = rec.get("id")
            if not rid:
                continue
            init_gid = f"{init_type.rstrip('s')}_{rid}"

            # Lead agents
            for aid in get_relation_ids(rec, "lead_agents"):
                lead_agent_links.append({"initiative_global_id": init_gid, "agent_id": aid, "initiative_type": init_type.rstrip('s')})

            # Agents
            for aid in get_relation_ids(rec, "agents"):
                agent_links.append({"initiative_global_id": init_gid, "agent_id": aid})

            # Thematic areas
            for tid in get_relation_ids(rec, "topics_thematic_areas"):
                theme_links.append({"initiative_global_id": init_gid, "thematic_area_id": tid})

            # Perceptions
            for pid in get_relation_ids(rec, "perceptions"):
                perception_links.append({"initiative_global_id": init_gid, "perception_id": pid})

            # Interconnections — extract linked entity IDs
            interconns = rec.get("attributes", {}).get("interconnections", {}).get("data", [])
            if isinstance(interconns, list):
                for ic in interconns:
                    ic_attrs = ic.get("attributes", {})
                    ic_id = ic.get("id")
                    # Linked agents
                    for a in get_relation_ids({"attributes": ic_attrs}, "agents"):
                        interconnection_links.append({
                            "initiative_global_id": init_gid,
                            "interconnection_id": ic_id,
                            "linked_agent_id": a,
                            "initiative_type": init_type.rstrip('s'),
                        })
                    # Linked projects
                    for p in get_relation_ids({"attributes": ic_attrs}, "projects"):
                        interconnection_links.append({
                            "initiative_global_id": init_gid,
                            "interconnection_id": ic_id,
                            "linked_initiative_id": p,
                            "linked_initiative_type": "project",
                        })
                    # Linked pilots
                    for p in get_relation_ids({"attributes": ic_attrs}, "pilots"):
                        interconnection_links.append({
                            "initiative_global_id": init_gid,
                            "interconnection_id": ic_id,
                            "linked_initiative_id": p,
                            "linked_initiative_type": "pilot",
                        })
                    # Linked protots
                    for p in get_relation_ids({"attributes": ic_attrs}, "protots"):
                        interconnection_links.append({
                            "initiative_global_id": init_gid,
                            "interconnection_id": ic_id,
                            "linked_initiative_id": p,
                            "linked_initiative_type": "prototype",
                        })

        # Write CSV files
        base_name = init_type.rstrip('s')
        if lead_agent_links:
            pd.DataFrame(lead_agent_links).to_csv(RELS_DIR / f"p10_{base_name}_lead_agent_links.csv", index=False)
            print(f"  lead_agent_links: {len(lead_agent_links)} rows")
        if agent_links:
            pd.DataFrame(agent_links).to_csv(RELS_DIR / f"p10_{base_name}_agent_links.csv", index=False)
            print(f"  agent_links: {len(agent_links)} rows")
        if theme_links:
            pd.DataFrame(theme_links).to_csv(RELS_DIR / f"p10_{base_name}_theme_links.csv", index=False)
            print(f"  theme_links: {len(theme_links)} rows")
        if perception_links:
            pd.DataFrame(perception_links).to_csv(RELS_DIR / f"p10_{base_name}_perception_links.csv", index=False)
            print(f"  perception_links: {len(perception_links)} rows")
        if interconnection_links:
            pd.DataFrame(interconnection_links).to_csv(RELS_DIR / f"p10_{base_name}_interconnection_links.csv", index=False)
            print(f"  interconnection_links: {len(interconnection_links)} rows")


# ============================================================
# 2. RE-EXTRACT AGENTS WITH CORRECT POPULATE
# ============================================================
def extract_agent_interconnections():
    print("\n=== 2. Agent interconnections ===")

    records = fetch_all(f"/agents?filters[parent_platform][id]={PLATFORM_ID}",
                        "&populate[0]=interconnections&populate[interconnections][populate][0]=agents&populate[interconnections][populate][1]=projects&populate[interconnections][populate][2]=pilots&populate[interconnections][populate][3]=protots")
    print(f"  Got {len(records)} agents")

    agent_interconn_links = []
    for rec in records:
        aid = rec.get("id")
        interconns = rec.get("attributes", {}).get("interconnections", {}).get("data", [])
        if isinstance(interconns, list):
            for ic in interconns:
                ic_attrs = ic.get("attributes", {})
                for linked_aid in get_relation_ids({"attributes": ic_attrs}, "agents"):
                    agent_interconn_links.append({
                        "agent_id": aid,
                        "interconnection_id": ic.get("id"),
                        "linked_agent_id": linked_aid,
                    })

    if agent_interconn_links:
        pd.DataFrame(agent_interconn_links).to_csv(RELS_DIR / "p10_agent_interconnection_links.csv", index=False)
        print(f"  agent_interconnection_links: {len(agent_interconn_links)} rows")
    else:
        print("  No agent interconnection links found")


# ============================================================
# 3. RE-EXTRACT INFORMATION → CHALLENGE LINKS
# ============================================================
def extract_info_challenge_links():
    print("\n=== 3. Information → Challenge links ===")

    records = fetch_all(f"/informations?filters[parent_platform][id]={PLATFORM_ID}",
                        "&populate[0]=challenge_opportunities&populate[1]=values&populate[2]=tags&populate[3]=patterns&populate[4]=information_pattern_connections&populate[5]=topics_thematic_areas")
    print(f"  Got {len(records)} informations")

    challenge_links = []
    value_links = []
    pattern_links = []
    tag_links = []
    theme_links = []
    ipc_links = []

    for rec in records:
        info_id = rec.get("id")

        # Challenge-opportunities
        for cid in get_relation_ids(rec, "challenge_opportunities"):
            challenge_links.append({"information_id": info_id, "challenge_id": cid})

        # Values
        for vid in get_relation_ids(rec, "values"):
            value_links.append({"information_id": info_id, "value_id": vid})

        # Tags
        for tid in get_relation_ids(rec, "tags"):
            tag_links.append({"information_id": info_id, "tag_id": tid})

        # Themes
        for tid in get_relation_ids(rec, "topics_thematic_areas"):
            theme_links.append({"information_id": info_id, "thematic_area_id": tid})

        # Information-pattern-connections
        for ipc_id in get_relation_ids(rec, "information_pattern_connections"):
            ipc_links.append({"information_id": info_id, "ipc_id": ipc_id})

        # Patterns (component, not relation)
        patterns = rec.get("attributes", {}).get("patterns", {})
        if isinstance(patterns, dict):
            pat_id = patterns.get("id")
            if pat_id:
                pattern_links.append({"information_id": info_id, "pattern_id": pat_id})

    if challenge_links:
        pd.DataFrame(challenge_links).to_csv(RELS_DIR / "p10_information_challenge_links.csv", index=False)
        print(f"  information_challenge_links: {len(challenge_links)} rows")
    if value_links:
        pd.DataFrame(value_links).to_csv(RELS_DIR / "p10_information_value_links.csv", index=False)
        print(f"  information_value_links: {len(value_links)} rows")
    if tag_links:
        pd.DataFrame(tag_links).to_csv(RELS_DIR / "p10_information_tag_links.csv", index=False)
        print(f"  information_tag_links: {len(tag_links)} rows")
    if pattern_links:
        pd.DataFrame(pattern_links).to_csv(RELS_DIR / "p10_information_pattern_links.csv", index=False)
        print(f"  information_pattern_links: {len(pattern_links)} rows")
    if theme_links:
        pd.DataFrame(theme_links).to_csv(RELS_DIR / "p10_information_theme_links.csv", index=False)
        print(f"  information_theme_links: {len(theme_links)} rows")
    if ipc_links:
        pd.DataFrame(ipc_links).to_csv(RELS_DIR / "p10_information_ipc_links.csv", index=False)
        print(f"  information_ipc_links: {len(ipc_links)} rows")


# ============================================================
# 4. RE-EXTRACT CHALLENGES WITH CHANNEL LINKS
# ============================================================
def extract_challenge_channel_links():
    print("\n=== 4. Challenge → Channel links ===")

    records = fetch_all(f"/challenge-opportunities?filters[parent_platform][id]={PLATFORM_ID}",
                        "&populate[0]=channels&populate[1]=supporting_quotes")
    print(f"  Got {len(records)} challenges")

    channel_links = []
    quote_links = []
    for rec in records:
        cid = rec.get("id")
        for chid in get_relation_ids(rec, "channels"):
            channel_links.append({"challenge_id": cid, "channel_id": chid})
        for qid in get_relation_ids(rec, "supporting_quotes"):
            quote_links.append({"challenge_id": cid, "information_id": qid})

    if channel_links:
        pd.DataFrame(channel_links).to_csv(RELS_DIR / "p10_challenge_channel_links.csv", index=False)
        print(f"  challenge_channel_links: {len(channel_links)} rows")
    if quote_links:
        pd.DataFrame(quote_links).to_csv(RELS_DIR / "p10_challenge_quote_links.csv", index=False)
        print(f"  challenge_quote_links: {len(quote_links)} rows")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print(f"Platform 10 (Chile COPOLAD) — Relation Re-extraction")
    print("=" * 60)
    print(f"Token: {AUTH_TOKEN[:20]}...")
    print(f"Output: {RELS_DIR}")

    extract_initiative_relations()
    extract_agent_interconnections()
    extract_info_challenge_links()
    extract_challenge_channel_links()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"New link tables in: {RELS_DIR}")


if __name__ == "__main__":
    main()
