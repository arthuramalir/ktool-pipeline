from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

# Note: Schemas are fully dynamic; extra attributes are ingested automatically
from graph_utils import (
    ANALYSIS_DIR,
    ANALYTICS_DIR,
    DATA_DIR,
    PLATFORM_ID,
    ensure_output_dirs,
    load_table,
    write_json,
)


def text_value(row: pd.Series, *columns: str, default: str = "") -> str:
    for col in columns:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return default


EDGE_FAMILY_DETAILS = {
    "declared_relational": {
        "label": "Declared relational",
        "description": "Direct source-record relationships recovered from the exported database.",
    },
    "interpretive": {
        "label": "Interpretive (AI-inferred)",
        "description": "Relationships inferred from semantic interpretation rather than explicitly stored links.",
    },
    "listening": {
        "label": "Listening layer",
        "description": "Evidence links connecting channels, information, values, and narrative traces.",
    },
    "qualitative_narrative": {
        "label": "Qualitative narrative",
        "description": "Narrative relationships based on perceptions, challenges, and thematic meaning.",
    },
}


def edge_family_metadata(family: str) -> dict[str, str]:
    key = str(family).strip()
    return EDGE_FAMILY_DETAILS.get(
        key,
        {
            "label": key.replace("_", " ").title(),
            "description": "Derived graph relationship family.",
        },
    )


def parse_ids(value) -> list[str]:
    """Safely extracts lists of identifiers from both comma-separated strings

    and JSON arrays embedded inside dataframes.
    """
    if pd.isna(value) or not str(value).strip():
        return []
    val_str = str(value).strip()

    # Attempt to process as a JSON array
    if val_str.startswith("[") and val_str.endswith("]"):
        try:
            parsed = json.loads(val_str)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if x]
        except json.JSONDecodeError:
            pass

    # Fallback to standard comma-separated extraction
    return [x.strip() for x in val_str.split(",") if x.strip()]


def add_node(
    rows: list[dict],
    global_id: str,
    native_id,
    node_type: str,
    label: str,
    description: str,
    phase: str,
    source_table: str,
    extra_attrs: dict = None,
) -> None:
    if not global_id or str(global_id) == "nan":
        return

    node = {
        "global_id": str(global_id),
        "native_id": "" if pd.isna(native_id) else native_id,
        "node_type": node_type,
        "label": label or "Unnamed",
        "description": description or "",
        "methodological_phase": phase,
        "platform_id": PLATFORM_ID,
        "source_table": source_table,
    }

    # Dynamically ingest all other columns from the source table
    if extra_attrs:
        for key, value in extra_attrs.items():
            if key not in node and not pd.isna(value):
                node[key] = value

    rows.append(node)


def add_edge(
    rows: list[dict],
    source: str,
    target: str,
    edge_type: str,
    family: str,
    phase: str,
    directed: bool,
    evidence: str,
    extra_attrs: dict = None,
    ai_generated: bool = False,
    edge_origin: str = "source_data",
    generated_by: str = "",
    inference_method: str = "",
) -> None:
    if not source or not target or str(source) == "nan" or str(target) == "nan":
        return

    edge = {
        "source_global_id": str(source),
        "target_global_id": str(target),
        "edge_type": edge_type,
        "edge_family": family,
        "edge_family_label": edge_family_metadata(family)["label"],
        "edge_family_description": edge_family_metadata(family)["description"],
        "methodological_phase": phase,
        "directed": directed,
        "evidence_source": evidence,
        "platform_id": PLATFORM_ID,
        "is_ai_generated": bool(ai_generated),
        "edge_origin": edge_origin,
        "generated_by": generated_by,
        "inference_method": inference_method,
    }

    # Dynamically inject extra relationship metrics (weights, timestamps, metadata)
    if extra_attrs:
        for key, value in extra_attrs.items():
            if key not in edge and not pd.isna(value):
                edge[key] = value

    rows.append(edge)


def file_hash(path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def build_nodes() -> pd.DataFrame:
    rows: list[dict] = []

    # 1. Processing Agents
    agents = load_table("agents.csv", "entities")
    for _, row in agents.iterrows():
        extra = row.to_dict()
        add_node(
            rows,
            f"agent_{row.get('id')}",
            row.get("id"),
            "agent",
            text_value(row, "name", default="Unnamed Agent"),
            text_value(row, "description"),
            "mapping",
            "agents.csv",
            extra_attrs=extra,
        )

    # 2. Processing Initiatives (Projects, Pilots, Prototypes)
    for node_type, filename in [
        ("project", "projects.csv"),
        ("pilot", "pilots.csv"),
        ("prototype", "prototypes.csv"),
    ]:
        frame = load_table(filename, "entities")
        for _, row in frame.iterrows():
            gid = text_value(
                row,
                "initiative_global_id",
                default=f"{node_type}_{row.get('native_id')}",
            )
            extra = row.to_dict()
            add_node(
                rows,
                gid,
                row.get("native_id"),
                node_type,
                text_value(row, "title", "name", default="Unnamed Initiative"),
                text_value(row, "description"),
                "mapping/co-creation",
                filename,
                extra_attrs=extra,
            )

    # 3. Processing Entity Specs (Listening Layer Components)
    entity_specs = [
        ("perception", "perceptions.csv", "listening", "perception"),
        ("challenge", "challenges.csv", "listening", "challenge"),
        ("channel", "channels.csv", "listening", "channel"),
        ("information", "informations.csv", "listening", "information"),
        ("value", "values.csv", "listening", "value"),
        ("theme", "thematic_areas.csv", "mapping", "theme"),
        ("session", "sessions.csv", "sensemaking", "session"),
    ]
    for node_type, filename, phase, prefix in entity_specs:
        frame = load_table(filename, "entities")
        for _, row in frame.iterrows():
            # Check for explicit global_id (e.g. from the new information schema) before auto-generating
            gid = text_value(
                row,
                "global_id",
                default=f"{prefix}_{row.get('id') or row.get('native_id')}",
            )
            label = text_value(
                row,
                "name",
                "title",
                "reference_number",
                default=f"{node_type.title()} {row.get('id') or row.get('native_id')}",
            )
            description = text_value(
                row, "description", "quote", "information_text"
            )
            extra = row.to_dict()
            add_node(
                rows,
                gid,
                row.get("native_id") or row.get("id"),
                node_type,
                label,
                description,
                phase,
                filename,
                extra_attrs=extra,
            )

    # Some KTool exports expose channels through the enriched informations table
    # before the same channel records appear in channels.csv.
    existing_ids = {str(row["global_id"]) for row in rows if row.get("global_id")}
    infos = load_table("informations.csv", "entities")
    for _, row in infos.iterrows():
        channel_id = text_value(row, "channel_id")
        if not channel_id:
            continue
        channel_gid = f"channel_{channel_id}"
        if channel_gid in existing_ids:
            continue
        add_node(
            rows,
            channel_gid,
            channel_id,
            "channel",
            text_value(row, "channel_code", default=f"Channel {channel_id}"),
            "",
            "listening",
            "informations.csv",
            extra_attrs={"derived_from_information_channel": True},
        )
        existing_ids.add(channel_gid)

    # Optionally include claim nodes from narrative extraction
    claim_path = ANALYSIS_DIR / "narrative_layers" / "claim_nodes.csv"
    if claim_path.exists():
        claim_df = pd.read_csv(claim_path)
        required_claim_cols = ["global_id", "node_type", "label"]
        if all(c in claim_df.columns for c in required_claim_cols):
            for _, cr in claim_df.iterrows():
                def _v(k, d=""):
                    v = cr.get(k, d)
                    return d if pd.isna(v) else str(v)
                add_node(
                    rows,
                    _v("global_id"),
                    _v("global_id"),
                    "claim",
                    _v("label", "Unnamed Claim"),
                    _v("description"),
                    "narrative_extraction",
                    "narrative_layers/claim_nodes.csv",
                    extra_attrs={
                        "narrative_level": _v("narrative_level"),
                        "verb": _v("verb"),
                        "subject_raw": _v("subject_raw"),
                        "object_raw": _v("object_raw"),
                        "source_node_id": _v("source_node_id"),
                        "value_dimension": _v("value_dimension"),
                        "belief_level": _v("belief_level"),
                        "is_ai_generated": True,
                        "generated_by": "21_extract_narrative_layers.py",
                    },
                )
            print(f"  + {len(claim_df)} claim nodes from narrative extraction")

    nodes = pd.DataFrame(rows).fillna("")
    if not nodes.empty:
        nodes = nodes.drop_duplicates(subset=["global_id"], keep="first")
    return nodes


def build_edges(valid_nodes: set[str]) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    expected_empty = [
        "agent_initiative_links.csv",
        "initiative_lead_agent_links.csv",
        "initiative_perception_links.csv",
        "initiative_thematic_area_links.csv",
        "information_pattern_links.csv",
        "pattern_perception_links.csv",
        "channel_information_links.csv",
        "value_perception_links.csv",
    ]

    # --- 1. HISTORICAL / SEPARATE RELATIONSHIP TABLES ---
    EDGE_SPECS = [
        (
            "agent_initiative_links.csv",
            "agent",
            "agent_id",
            None,
            "initiative_global_id",
            "initiative_has_agent",
            "mapping",
            "mapping",
            False,
        ),
        (
            "initiative_lead_agent_links.csv",
            None,
            "initiative_global_id",
            "agent",
            "agent_id",
            "initiative_has_lead_agent",
            "governance",
            "mapping",
            True,
        ),
        (
            "initiative_perception_links.csv",
            None,
            "initiative_global_id",
            "perception",
            "perception_id",
            "initiative_mentions_perception",
            "listening",
            "listening",
            False,
        ),
        (
            "initiative_thematic_area_links.csv",
            None,
            "initiative_global_id",
            "theme",
            "thematic_area_id",
            "initiative_addresses_theme",
            "semantic",
            "mapping",
            False,
        ),
        (
            "link_agent_interconnections.csv",
            "agent",
            "source_agent_id",
            None,
            "target_global_id",
            "declared_interconnection",
            "declared_relational",
            "mapping",
            False,
        ),
        (
            "initiative_initiative_links.csv",
            None,
            "source_initiative_global_id",
            None,
            "target_global_id",
            "declared_interconnection",
            "declared_relational",
            "mapping",
            False,
        ),
        (
            "channel_information_links.csv",
            "information",
            "information_id",
            "channel",
            "channel_id",
            "information_originates_from_channel",
            "listening",
            "listening",
            True,
        ),
        (
            "information_pattern_links.csv",
            "information",
            "information_id",
            "pattern",
            "pattern_id",
            "information_forms_pattern",
            "interpretive",
            "listening",
            True,
        ),
        (
            "information_value_links.csv",
            "information",
            "information_id",
            "value",
            "value_id",
            "information_expresses_value",
            "interpretive",
            "listening",
            True,
        ),
        (
            "pattern_perception_links.csv",
            "pattern",
            "pattern_id",
            "perception",
            "perception_id",
            "pattern_feeds_perception",
            "interpretive",
            "listening",
            True,
        ),
        (
            "perception_challenge_links.csv",
            "perception",
            "perception_id",
            "challenge",
            "challenge_id",
            "perception_reveals_challenge",
            "interpretive",
            "listening",
            True,
        ),
        (
            "value_perception_links.csv",
            "value",
            "value_id",
            "perception",
            "perception_id",
            "value_frames_perception",
            "interpretive",
            "listening",
            True,
        ),
        (
            "information_pattern_connections.csv",
            "information",
            "information_id",
            "pattern",
            "pattern_id",
            "information_pattern_connection",
            "interpretive",
            "listening",
            True,
        ),
    ]

    for (
        filename,
        src_prefix,
        src_col,
        tgt_prefix,
        tgt_col,
        edge_type,
        family,
        phase,
        directed,
    ) in EDGE_SPECS:
        frame = load_table(filename, "relationships")
        for _, row in frame.iterrows():
            raw_src = text_value(row, src_col)
            raw_tgt = text_value(row, tgt_col)

            if not raw_src or not raw_tgt:
                continue

            source_id = f"{src_prefix}_{raw_src}" if src_prefix else raw_src
            target_id = f"{tgt_prefix}_{raw_tgt}" if tgt_prefix else raw_tgt

            extra = row.to_dict()
            add_edge(
                rows,
                source_id,
                target_id,
                edge_type,
                family,
                phase,
                directed,
                filename,
                extra_attrs=extra,
            )

    # --- 2. INLINE RELATIONSHIP EXTRACTION FROM NEW MAPPING SCHEMAS (projects.csv) ---
    projects_df = load_table("projects.csv", "entities")
    for _, row in projects_df.iterrows():
        src_id = text_value(
            row,
            "initiative_global_id",
            default=f"project_{row.get('native_id')}",
        )
        if not src_id or src_id == "nan":
            continue

        extra_metadata = {"extracted_from": "projects_inline"}

        # Lead Agents
        for la_id in parse_ids(row.get("lead_agent_ids")):
            add_edge(
                rows,
                src_id,
                f"agent_{la_id}",
                "initiative_has_lead_agent",
                "governance",
                "mapping",
                True,
                "projects.csv",
                extra_attrs=extra_metadata,
            )

        # Regular Agents
        for a_id in parse_ids(row.get("agent_ids")):
            add_edge(
                rows,
                src_id,
                f"agent_{a_id}",
                "initiative_has_agent",
                "mapping",
                "mapping",
                False,
                "projects.csv",
                extra_attrs=extra_metadata,
            )

        # Perceptions
        for p_id in parse_ids(row.get("perception_ids")):
            add_edge(
                rows,
                src_id,
                f"perception_{p_id}",
                "initiative_mentions_perception",
                "listening",
                "listening",
                False,
                "projects.csv",
                extra_attrs=extra_metadata,
            )

        # Topics / Thematic Areas
        for t_id in parse_ids(row.get("topic_ids")):
            add_edge(
                rows,
                src_id,
                f"theme_{t_id}",
                "initiative_addresses_theme",
                "semantic",
                "mapping",
                False,
                "projects.csv",
                extra_attrs=extra_metadata,
            )

        # Interconnections Array (JSON)
        for target_gid in parse_ids(row.get("interconnections_json")):
            add_edge(
                rows,
                src_id,
                target_gid,
                "declared_interconnection",
                "declared_relational",
                "mapping",
                False,
                "projects.csv",
                extra_attrs=extra_metadata,
            )

    # --- 3. INLINE RELATIONSHIP EXTRACTION FROM NEW LISTENING SCHEMAS (informations.csv) ---
    channels_df = load_table("channels.csv", "entities")
    channel_lookup = {}
    for _, row in channels_df.iterrows():
        channel_id = text_value(row, "id", "native_id")
        channel_display = text_value(row, "code", "name", "title", default=f"Channel {channel_id}")
        if channel_id:
            channel_lookup[str(channel_id)] = channel_display

    infos_df = load_table("informations.csv", "entities")
    for _, row in infos_df.iterrows():
        src_id = text_value(
            row,
            "global_id",
            default=f"information_{row.get('id') or row.get('native_id')}",
        )
        if not src_id or src_id == "nan":
            continue

        extra_metadata = {"extracted_from": "informations_inline"}

        # Direct channel connection
        ch_id = text_value(row, "channel_id")
        ch_name = text_value(row, "channel_name", "channel_code", default="")
        if not ch_name and ch_id:
            ch_name = channel_lookup.get(str(ch_id), f"Channel {ch_id}")
        if not ch_name:
            ch_name = "No channel defined"

        if ch_id:
            add_edge(
                rows,
                src_id,
                f"channel_{ch_id}",
                "information_originates_from_channel",
                "listening",
                "listening",
                True,
                "informations.csv",
                extra_attrs={**extra_metadata, "channel_name": ch_name},
            )

        # Information-Pattern Connections (JSON array)
        for pat_id in parse_ids(row.get("information_pattern_connections_json")):
            target_pat = (
                pat_id if str(pat_id).startswith("pattern_") else f"pattern_{pat_id}"
            )
            add_edge(
                rows,
                src_id,
                target_pat,
                "information_forms_pattern",
                "interpretive",
                "listening",
                True,
                "informations.csv",
                extra_attrs=extra_metadata,
            )

    # --- 4. AI-INFERRED SEMANTIC QUOTE EDGES ---
    quote_semantic_edges = load_table("quote_semantic_edges.csv", "relationships")
    semantic_edge_types = {"sequence", "causality", "contradiction", "frequency", "similarity"}
    for _, row in quote_semantic_edges.iterrows():
        source_id = text_value(row, "source_global_id", "source")
        target_id = text_value(row, "target_global_id", "target")
        edge_type = text_value(row, "edge_type", "connection_type", default="semantic_similarity")
        if not source_id or not target_id:
            continue

        extra = row.to_dict()
        if edge_type in semantic_edge_types:
            extra["alc_semantic_parameter"] = text_value(row, "parameter", default=edge_type)

        add_edge(
            rows,
            source_id,
            target_id,
            edge_type,
            text_value(row, "edge_family", default="ai_semantic"),
            text_value(row, "methodological_phase", default="listening"),
            text_value(row, "directed", default="False").strip().lower() in {"true", "1", "yes"},
            text_value(row, "evidence_source", default="quote_semantic_edges.csv"),
            extra_attrs=extra,
            ai_generated=True,
            edge_origin="ai_inferred",
            generated_by=text_value(row, "generated_by", default="14_alc_advanced_semantic_edges.py"),
            inference_method=text_value(row, "inference_method", default="sentence_transformer_cross_encoder"),
        )

    # --- 5. OPTIONAL NARRATIVE CLAIM EDGES ---
    claim_edges_path = ANALYSIS_DIR / "narrative_layers" / "claim_edges.csv"
    if claim_edges_path.exists():
        claim_edge_df = pd.read_csv(claim_edges_path)
        required_edge_cols = ["source_global_id", "target_global_id", "edge_type"]
        if all(c in claim_edge_df.columns for c in required_edge_cols):
            for _, er in claim_edge_df.iterrows():
                def _v(k, d=""):
                    v = er.get(k, d)
                    return d if pd.isna(v) else str(v)
                add_edge(
                    rows,
                    _v("source_global_id"),
                    _v("target_global_id"),
                    _v("edge_type", "claim_relation"),
                    _v("edge_family", "narrative_claim"),
                    _v("methodological_phase", "narrative_extraction"),
                    str(_v("directed", "True")).strip().lower() in {"true", "1", "yes"},
                    _v("evidence_source", "narrative_layers/claim_edges.csv"),
                    extra_attrs={
                        "weight": er.get("weight", 1.0),
                    },
                    ai_generated=True,
                    edge_origin="ai_inferred",
                    generated_by="21_extract_narrative_layers.py",
                    inference_method="hyperbase_alphabeta_parser",
                )
            print(f"  + {len(claim_edge_df)} claim edges from narrative extraction")

    # --- VALIDATION, CLEANUP, & DEDUPLICATION ---
    edges = pd.DataFrame(rows).fillna("")
    before_dedupe = len(edges)
    if edges.empty:
        return edges, {
            "orphan_edges_removed": 0,
            "duplicate_edges_removed": 0,
            "expected_empty_relations": expected_empty,
        }

    edges = edges.drop_duplicates(
        subset=["source_global_id", "target_global_id", "edge_type", "evidence_source"],
        keep="first",
    )
    duplicate_edges_removed = before_dedupe - len(edges)

    orphan_mask = ~edges["source_global_id"].isin(valid_nodes) | ~edges[
        "target_global_id"
    ].isin(valid_nodes)
    orphan_edges_removed = int(orphan_mask.sum())
    edges = edges.loc[~orphan_mask].copy()

    if "edge_id" in edges.columns:
        edges = edges.drop(columns=["edge_id"])
    edges.insert(0, "edge_id", [f"edge_{idx + 1}" for idx in range(len(edges))])
    return edges, {
        "orphan_edges_removed": orphan_edges_removed,
        "duplicate_edges_removed": duplicate_edges_removed,
        "expected_empty_relations": expected_empty,
    }


def main() -> None:
    ensure_output_dirs()
    nodes = build_nodes()
    valid_nodes = set(nodes["global_id"].astype(str))
    edges, diagnostics = build_edges(valid_nodes)

    for out_dir in [DATA_DIR, ANALYTICS_DIR]:
        nodes.to_csv(out_dir / "nodes.csv", index=False)
        edges.to_csv(out_dir / "edges.csv", index=False)

    input_files = sorted(
        path
        for path in DATA_DIR.glob("*.csv")
        if path.name not in {"nodes.csv", "edges.csv"}
    )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(DATA_DIR),
        "node_count": int(len(nodes)),
        "edge_count": int(len(edges)),
        "node_types": (
            nodes["node_type"].value_counts().to_dict() if not nodes.empty else {}
        ),
        "edge_families": (
            edges["edge_family"].value_counts().to_dict() if not edges.empty else {}
        ),
        "diagnostics": diagnostics,
        "input_files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256_16": file_hash(path),
            }
            for path in input_files
        ],
    }
    write_json(ANALYSIS_DIR / "baseline_manifest.json", manifest)
    print(f"Prepared {len(nodes)} nodes and {len(edges)} edges.")
    print(f"Wrote normalized tables to {ANALYTICS_DIR}")


if __name__ == "__main__":
    main()
