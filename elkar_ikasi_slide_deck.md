# Elkar Ikasi — Slide Deck
**15 min · Dashboard-led · Each slide maps to a tab**

I will switch to the live dashboard for every example. Slides are the guide; the dashboard is the star.

---

## SLIDE 1 — Title

**What to say:** "From Data to Diagnosis"

Arthur Amalir — Internship — Elkar Ikasi 2026
ALC logo

**Ask the room:** "How many of you have used KTool to export data and then found yourself manually cross-referencing tables in Excel?"

---

## SLIDE 2 — The Problem: KTool Has a Blind Spot

**What to say:**

KTool is a rich repository built on a sound methodology. It captures agents, projects, perceptions, channels, quotes, budgets — all linked by a shared ontology. That is rare and valuable.

BUT: its architecture stores relationships **in rows, not as a model**.

**01 / VISIBILITY — Invisible Structure**
To understand who is connected to whom, you must mentally join tables. This is a human impossibility at scale. With 6+ years of data across multiple platforms, the relational complexity exceeds what any person can hold in working memory.

**02 / STRUCTURE — Static Schema**
Questions like "Who is most central?" or "What happens if this actor leaves?" are not queries you can run against raw relational tables. KTool has no graph engine — it cannot compute structural metrics.

**03 / ANALYSIS — Isolated Domains**
Data exists across domains (who, what, what they said, how much it costs). But cross-domain analysis is manual. You cannot run a query that says "show me which narratives correlate with high-budget initiatives."

**04 / FORECAST — No Simulation**
The tool is static. You cannot model "what if" scenarios — shift funding, merge clusters, remove a key actor.

**Before / After example:**
- **Before:** Finding "which agents connect to which projects through which channels" requires manually joining 4 tables. If someone asks "is this network fragile?", you guess.
- **After:** Same multi-table relationship becomes one graph query across typed nodes and edges. Fragility is computed as a robustness decay curve with a precise diagnostic metric.

**Transition:** "The dashboard walks through how we fill this gap, tab by tab."

---

## SLIDE 3 — TAB 0: Overview (The Graph at a Glance)

**What to say:**

This is the entry point. The pipeline transforms KTool exports into a **heterogeneous graph** — every row becomes a node, every relationship becomes an edge.

**Show on dashboard:**
- Node-type bar chart: "375 items — agents, projects, perceptions, information nodes, channels. Each type has a colour."
- Network map: "Each dot is an item, each line is a relationship. Colour = type. Size = connectedness. The map is force-directed — connected nodes cluster together."
- Mouse over a node: "This agent has X connections. Hover to see type and degree."
- Giant component share: "Only 19% of nodes are in the largest cluster. That tells us immediately: this ecosystem is fragmented."

**Why this matters:**
"The graph is the foundation. Every subsequent tab is a different lens on this same structure."

---

## SLIDE 4 — TAB 1: Network Layers (Anatomy of the Graph)

**What to say:**

Not all connections are the same. KTool captures multiple kinds of relationships, and the pipeline preserves their semantic differences.

**Four families of links:**

| Family | What it records | Example |
|---|---|---|
| **Declared** | Agent↔project, project↔perception, initiative interconnections | "Rethink Ireland funds Project X" — from source data |
| **Listening** | Channel→information→value evidence chain | "Channel Y produced Quote Z about value W" |
| **Narrative** | Perception↔challenge, pattern↔perception | "Perception P frames Challenge C as a funding problem" |
| **AI-inferred** | Quote-to-quote semantic links (similar, contradictory, causal, sequential) | "Quote A and Quote B both argue for community health — AI marks them as similar" |

**Show on dashboard:**
- The link-type bar chart: "Declared links dominate in count. But narrative and listening links are structurally more important — they connect otherwise separate parts of the graph."
- The filter: "You can toggle link types on and off. Watch what happens when you hide declared links — the network splits into isolated components. That tells you declared links are the backbone."
- The cluster count changing: "Each link type connects a different set of components. Listening links merge X clusters. Narrative links merge Y."

**Why this matters:**
"Understanding which link type does the structural work tells you where to invest. If listening links are the ones merging components, invest in listening. If narrative links are, invest in narrative analysis."

---

## SLIDE 5 — TAB 2: Health Check (Structural Diagnostics)

**What to say:**

Once we know what the graph is made of, we ask: **is it healthy?**

**Stress test — targeted vs random removal:**
- We simulate removing the 10% most central nodes (targeted attack)
- Then remove 10% at random
- Measure how fast the giant component shrinks

**Show on dashboard:**
- The two metric cards: "Targeted drop: X%. Random drop: Y%. The gap tells us how much the system depends on a few irreplaceable hubs."
- If gap > 20%: "⚠️ Too many eggs in one basket. If [top fragile node] leaves, 3 components disconnect."
- The decay curve chart: "The blue line (targeted) drops faster than the orange line (random). That is the signature of a hub-dependent network."

**Weak spots table:**
"Below the stress test: a ranked list of the most vulnerable nodes — by name, with risk scores."
- "Example: [Node Name] is an articulation point. Remove it and the graph fragments."
- "These are not theoretical. Each node is a real agent or project with a name and a type."

**What to do:**
"Add backup bridges for fragile items. Aim for at least 3 links each. If [Node] is the only connection between Components A and B, build a parallel pathway."

---

## SLIDE 6 — TAB 3: Listening (What People Are Saying)

**What to say:**

The structure tells us who is connected. The listening data tells us **what they are saying**.

KTool captures listening exercises: workshops, consultations, interviews. Each produces quotes tagged by channel, topic, and value.

**Show on dashboard:**
- Quotes by channel bar chart: "Which channels produce the most discourse? The top channel is [name] with [N] quotes."
- Most influential voices: "Diffusion bias measures whether a voice gets more attention than expected. These are the actors whose words travel furthest through the network."
- Browse quotes: "Every quote is here, filterable by channel. This is the raw material for the next two tabs."

**Why this matters:**
"Listening data is the richest qualitative layer in KTool. But in raw form it is unreadable at scale — hundreds of quotes, no structure. The next tabs add that structure."

---

## SLIDE 7 — TAB 4: AI-Generated Links (Semantic Edges)

**What to say:**

The listening tab gives us raw quotes. But relationships between quotes exist too — similarity, contradiction, causality, sequence.

We find these using semantic NLP. Every quote is converted to a mathematical vector. Pairs that are close in vector space get an AI-inferred edge. This sits right after raw quotes because it shows you the latent connections between what people said.

**Provenance is explicit:**
Every AI-generated edge carries a label: `is_ai_generated=True`, `edge_origin="ai_inferred"`, `generated_by="[script name]"`. Source edges carry no such label. You can always tell them apart.

**Show on dashboard:**
- "498 AI-inferred edges vs 156 source edges. The AI more than doubles the link count."
- Semantic type bar chart: "Most AI edges are 'similarity'. Some are 'contradiction' — those are the interesting ones."
- Filter by type: "Show only contradictions. These are places where the ecosystem disagrees with itself."
- Browse the table: "Each row shows source, target, semantic type, inference method, and the AI's explanation."

**Why this matters:**
"This tab makes the AI's work visible. You can inspect every edge, check its reasoning, and decide whether you trust it. Nothing is hidden behind a black box."

---

## SLIDE 8 — TAB 5: Story Clusters (Narrative Profiles)

**What to say:**

With AI links in place, we can now group semantically related quotes into **story clusters** — sets of quotes that share thematic content.

**Two kinds of profiles:**
1. **Auto-detected:** Algorithm finds clusters in the quote text. [N] clusters found from [M] quotes.
2. **Manual:** Expert-curated three-layer profiles (surface / implicit / metanarrative) written by ALC analysts.

**Show on dashboard:**
- Bubble chart: "Each bubble is a topic in a story cluster. Bigger = more quotes. Clusters overlap when they share topics."
- The three-layer guide: "Every story can be read at three levels — surface (what they say), hidden (what they assume), big picture (the deeper pattern). This is the ALC Urdaibai methodology, operationalised."
- Expand a manual profile: "Here is a real profile with representative quotes, tone, associated values, and contradictions."

**Concrete example:**
"Cluster [X] is about short-term funding. Surface: 'short-term funding makes long-term planning impossible.' Hidden: 'funders don't trust communities.' Big picture: 'short-term funding is accountability theatre — it cuts core support while appearing responsible.'"

**Why this matters:**
"Raw quotes become structured narratives. You can now ask: which clusters dominate? Which are under-represented? Which contradict each other?"

---

## SLIDE 9 — TAB 6: Perceptions (How People Think)

**What to say:**

With story clusters identifying narrative themes, the perceptions tab asks: **how solid are these perceptions?**

KTool captures structured perceptions — how specific actors frame specific challenges. The perception diagnostics tab measures robustness.

**Four metrics:**

| Metric | What it measures | Threshold |
|---|---|---|
| **Agreement** | How similar quotes are inside a perception | >0.6 = strong |
| **Focus** | Whether quotes stay in their own perception or drift into others | >0.3 = clean |
| **Source diversity** | How many channels feed this perception | >0.5 = diverse |
| **Contradiction** | Fraction of internal links marked as disagreement | <30% = coherent |

**Show on dashboard:**
- Scatter plot (Agreement vs Focus): "Each dot is a perception. Top-right = solid. Bottom-left = weak."
- The threshold lines: "Anything below orange lines needs attention."
- Colour-coded health table: "Green = robust. Yellow = weak. Red = contradictory."

**Concrete example:**
"Perception [X]: agreement 0.82, focus 0.45, source diversity 0.3. The quotes agree strongly but come from only one channel — one group's view, not a shared perception."
"Perception [Y]: agreement 0.21, contradiction 40%. The quotes contradict each other — this is a contested perception, important to show but not as a single unified view."

**Why this matters:**
"Not all perceptions are equally valid. This tab separates robust, multi-source perceptions from weak, single-source, or contradictory ones. It tells you where to trust and where to investigate."

---

## SLIDE 10 — TAB 7: Claims (Structured Narrative Extraction)

**What to say:**

This is the newest and richest layer. We extract **structured claims** from quote text — not just clusters, but Subject→Verb→Object triples with named entities, value dimensions, and narrative levels.

**Three levels:**

| Level | What it captures | Example |
|---|---|---|
| **Surface** | Explicit statements | "Public institutions seek measurable impact" |
| **Implicit** | Inferred assumptions (negation, conditionals, emergency framing, emotion) | "If we had the resources" → unstated: resources are insufficient |
| **Metanarrative** | Value dimension classification | Cultural identity, social justice, collaboration, innovation, evidence-based, community autonomy, austerity/scarcity |

**Entity linking:**
Claims are linked back to graph nodes. If a claim mentions "HSE", the pipeline finds the matching agent node. Current linking rate: 78%.

**Show on dashboard:**
- Value dimension bar chart: "7 value dimensions detected. `Cultural identity` dominates with 14 claims. `Community autonomy` has 1."
- Claim table: "Every claim has a source node ID, a value dimension, and a narrative level. Filter by level."
- Extraction summary: "122 total claims. 79 surface, 43 implicit. 78% entity linking rate."

**Concrete example — the funding gap:**
"We cross-tabulated claims with financial data (next tab). Result: `Cultural identity` claims link to €17.6M in funding. `Community autonomy` claims link to €0. That is not random — it is a structural misalignment between narrative prevalence and resource allocation."

**Why this matters:**
"This is the layer that turns unstructured text into testable hypotheses. Each claim is traceable back to the original source text. Nothing is invented — it is extracted, classified, and linked."

---

## SLIDE 11 — TAB 8: What-If Simulator (GNN Predictions)

**What to say:**

All previous tabs describe what **exists** in the data. This tab asks: **what could exist?**

The Graph Neural Network learns patterns from existing connections — "nodes of type X tend to connect to nodes of type Y when they share topics Z" — and then predicts missing links.

**Show on dashboard:**
- Top suggested links: "The model recommends [N] new connections. The top one: connect [Node A] to [Node B] — they share thematic overlap but have no recorded link."
- Merge components: "[M] of these links would merge separate clusters. Those get highest priority."
- Link impact sensitivity: "For each proposed link, we simulate the effect on the network. Sensitivity score = how much the conversation flow changes."

**Concrete example:**
"Platform 173_synthetic: 3 high-confidence predictions (probability > 0.99). The model says 'Surf Club' should be linked to 'Mental Health & Wellbeing'. They share topics but have no recorded relationship."

**Honest about limits:**
"GNNs need data. With 260-650 edges per platform, the model is data-constrained. These are hypotheses, not findings. But the question — 'what if we added this link?' — is itself valuable regardless of prediction quality."

---

## SLIDE 12 — TAB 9: Structural Change (Is Change Possible?)

**What to say:**

This is the capstone. All previous layers describe the network. This layer asks: **given this structure, is change feasible?**

I built a framework that bridges network metrics to political science theory:

**Three frameworks, three questions:**

| Theory | Question | Method | Finding (platform 173) |
|---|---|---|---|
| **ACF** (Sabatier & Jenkins-Smith) | Who holds power? | k-core finds the entrenched coalition | Core exists but is small (12% of nodes) — stable minority |
| **PET** (Baumgartner & Jones) | Is the system brittle? | Robustness gap (targeted vs random) | Gap = −0.11 — moderate dependence on core |
| **NPG** (Provan & Kenis) | What governance mode fits? | Intervention maturity matrix | Fragmented → recommend Core Consolidation (NAO) |

**Composite readiness score:**
- Leverage: 0.31 — few existing bridges
- Plasticity: 0.33 — saturated, hard to add links
- Blockage: 1.0 — 33 fragile connectors, 263 isolated nodes
- Lock-in: 0.41 — moderate
- **Overall: 0.28 — LOW readiness**

**Show on dashboard:**
- The five score cards across the top
- The hypothesis expanders — grouped by type with confidence indicators
- Specific hypothesis examples:
  - "🟢 Rethink Ireland is an articulation point. If removed, 3 components disconnect. Action: engineer redundant connections."
  - "🟢 Perception 'Siobhán' has no path to any information node. Action: deploy a channel targeting her sector."
  - "🔒 The k-core contains [nodes]. Small dominant coalition. Action: fund venue-shopping, not direct challenge."
- Financial-Perception Bridge: "€17.6M for `cultural identity` vs €0 for `community autonomy`. That gap IS a structural change question."

**Bottom line:**
"The pipeline generates ~48 specific, testable hypotheses per platform. Each one names a real node, gives a confidence level, and recommends an action. These are not abstract metrics — they are concrete starting points for strategic discussion."

---

## SLIDE 13 — Summary

**What to say:**

**What exists now:**
- A pipeline that turns any KTool export into a multi-layer ecosystem model
- 10 dashboard tabs, each a different analytical lens on the same graph
- Structural analysis, narrative extraction, financial-perception bridge, GNN predictions, political-science-grounded diagnosis
- Every AI output labelled and traceable

**What this enables:**
- Questions that were previously unanswerable become computable
- Cross-domain analysis that was previously manual becomes automated
- Intuition is replaced by testable hypotheses

**What is next (if time):**
- Validate claim extraction against human coders
- Temporal analysis: older KTool exports → narrative change detection
- Cross-platform comparison

**Where I fit in this programme:**
- Session 1: I can live-demo the entire pipeline on a platform of your choice in 5 minutes
- AI Protocol: the provenance labelling system is directly relevant
- Strategic Mapping: if ALC's data is in KTool, I can bring a draft organisational map

---

## Backup Slides (if asked)

**How provenance works:** Every node and edge carries `is_ai_generated`, `generated_by`, `edge_origin`. Source data: these fields are empty. AI data: explicitly set. You can filter by provenance at any time.

**Architecture:** KTool export → Python pipeline (15 scripts) → NetworkX + CSVs → GNN (PyTorch) → Dashboard (Streamlit + Plotly). Each script independently runnable.

**Known limitations:** Small graph limits GNN. Claim extraction unvalidated against human coders (planned). Synthetic budget data is realistic but not real.
