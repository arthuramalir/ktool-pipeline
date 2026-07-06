# Elkar Ikasi — Slide Plan

I will switch to the live dashboard for examples. Platform 173 is one case study — KTool is much richer than any single export.

**~12-15 slides. 15 min.**

Structure: **walk through the dashboard tabs in order.** Each slide covers one tab. I will switch to the live dashboard for every example.
This way the slides are a guide, and the dashboard is the star — the audience sees real data with every point.

---

## SLIDE 1 — Title

"From Data to Diagnosis"
Arthur Amalir — Internship — Elkar Ikasi 2026
ALC logo

---

## SLIDE 2 — Topic 1: What KTool Actually Captures

**What to say:**

KTool is not a survey tool. It is a **relational ecosystem model** designed around a specific methodological framework — the ALC Urdaibai methodology.

What it captures, structurally:

- **Actors** — agents, their roles, sectors, demographics, power dynamics
- **Initiatives** — projects, pilots, prototypes, with budgets, timelines, impact levels
- **Listening infrastructure** — channels, their accessibility, their power dynamics, who runs them
- **Discourse** — information items (quotes, workshop outputs), tagged by topic, thematic area, and value
- **Perceptions** — structured perception mappings linked back to information channels
- **Relationships** — who collaborates with whom, which channel feeds which information, which perception links to which challenge

**Most platforms** use only a subset of this. A consultation exercise might use agents + channels + information. A strategic planning process might use projects + perceptions + challenges. A budget analysis adds the financial fields.

**The richness is in the schema, not any single dataset.**
KTool's data model is designed so that these pieces **fit together** even when collected separately — because they share the same ontology of actors, channels, information, perceptions, and relationships.

**Concrete example:**
Platform 173 is one case study — 375 nodes, 6 methodological phases, workshops + consultations + listening exercises. It uses about 60% of KTool's full schema. Other platforms use different subsets. The pipeline handles any subset because it reads the schema, not the specific fields.

---

## SLIDE 3 — Topic 2: The Problem

**What to say:**

KTool stores more relational data than almost any comparable tool. But it has a structural limitation: **data is stored in rows. A row is not a model.**

What that means in practice, regardless of which platform or which subset of the schema you are using:

- **You cannot see the system whole.** To understand how agents connect to projects through channels to perceptions, you need to mentally join tables. At 100+ nodes, that is difficult. At 500+, it is impossible.
- **Structural questions are unanswerable.** "Who is most central?" "Which actor is the single point of failure?" "Is this ecosystem hierarchical or distributed?" These are not SQL queries you can write against the current export schema.
- **Cross-domain questions require manual work.** "Which narratives are associated with which budgets?" "Do highly-funded projects correlate with certain perception profiles?" The data exists. The analysis does not.
- **Simulation is impossible.** "What if we connect these two clusters?" "What if funding shifts?" The tool is static.

**This is not a criticism of KTool.** KTool was built to **collect** richly structured data. No tool that collects data also models it — they are different functions. The gap between collection and diagnosis is what the pipeline fills.

**Concrete example:**
- In the raw export, finding "which agents connect to which projects through which channels" requires joining 4+ tables manually
- In the pipeline, it is one query against a graph with typed edges
- Before: "is this network fragile?" = guess. After: robustness decay curve with a specific number

---

## SLIDE 4 — Topic 3: Extraction & Mapping (How the Graph Is Built)

**What to say:**

The first thing the pipeline does is transform the KTool export into a **network model** — a graph. This works for any platform, any subset of the schema.

**Everything becomes a node with a type:**
- agent, project, pilot, prototype (operational entities)
- perception, information, channel (discourse entities)
- challenge, value, theme (analytical entities)
- claim (narrative extraction output — more on this later)

**Every relationship becomes an edge with a family:**
- `declared_relational`: agent↔project, project↔perception, initiative interconnections
- `listening`: channel→information→value evidence chain
- `interpretive`: AI-inferred semantic links between quotes
- `qualitative_narrative`: perception↔challenge, pattern↔perception
- `narrative_claim`: claim→entity links

**The result is a single, unified graph — regardless of which platform or which data subset.**

For platform 173 (one case study): 375 nodes, 260 edges from source data.
For 173_synthetic: 497 nodes, 652 edges including claims.
For other platforms: it depends on what data exists. The pipeline adapts.

**Why this matters:**
- **Lossless** — every column in the export becomes a node attribute. No information is discarded.
- **Type-safe** — an agent→project edge means something different from a perception→challenge edge. The pipeline never collapses them into "connection."
- **Extensible** — new platforms, new methodological phases, new data types all merge into the same graph structure.

**Dashboard reference:**
The Overview tab shows the graph for whatever platform is selected. The node-type bar chart, the network map, the metrics row — all generated from the graph.

---

## SLIDE 5 — Topic 4: Analytical Layers (In Plain English)

**What to say:**

Once we have the graph, we run a series of analyses. Each one answers a specific governance question.

**Layer 1 — Centrality: who matters?**
- **Betweenness centrality**: which nodes are the bridges between otherwise disconnected groups?
- **PageRank**: which nodes are considered authoritative because they connect to other authoritative nodes?
- **For platform 173:** the top bridges are mostly information nodes, not people — meaning shared knowledge artefacts, not individual actors, hold the network together.

**Layer 2 — Fragility: what breaks?**
- **Articulation points**: nodes whose removal fragments the network into isolated pieces.
- **Robustness decay**: simulate removing the most central nodes vs random nodes. Measure how fast the core shrinks.
- **For platform 173:** 33 fragile connectors. The network degrades 11% faster under targeted attack than random failure.

**Layer 3 — Community detection: who clusters together?**
- **K-core decomposition**: find the densest, most entrenched group in the network.
- **Component analysis**: count disconnected clusters.
- **For platform 173:** 263 clusters. Only 19% of nodes in the giant component. This is a highly fragmented ecosystem.

**Layer 4 — Perception diagnostics: what do people see?**
- For each perception node, measure whether it has a path to any information node.
- Perceptions with no path = discursive silos — opinions formed without access to the broader conversation.
- **For platform 173:** 5 blocked perceptions. These actors hold opinions that cannot be informed or challenged by the ecosystem's communication channels.

**Dashboard reference:**
- Health Check tab: stress test results, fragile connectors table
- Perceptions tab: diagnostic table with blocked perceptions highlighted
- Structural Change tab: all four layers feed into the composite readiness score

---

## SLIDE 6 — Topic 5: NLP for Semantic Edges & Narrative Extraction

**What to say:**

The source data encodes structure — who is linked to whom. It does not encode **meaning** — what people are actually saying, how their statements relate, what values they express.

To add meaning, we analyse the quote text.

**Semantic edges between quotes:**
- Every quote is converted to a mathematical vector using sentence embeddings.
- Pairs of quotes that are semantically similar get an "interpretive" edge.
- The relationship is classified: similarity, contradiction, causality, or sequence.
- **Example:** "we need more funding for community health" and "community health is under-resourced" — recognised as a **reinforcement** pair.

**Three-level narrative extraction:**
We extract structured **claims** from each quote:

- **Surface** — explicit statements. Parsed as Subject→Verb→Object triples.
  - Example: "public institutions seek measurable impact"

- **Implicit** — unstated assumptions. Detected through negation ("cannot afford to fail" → failure is possible), conditionals ("if we had resources" → resources are insufficient), emergency framing, emotion markers.

- **Value dimensions** — each claim classified into one of seven:
  - cultural_identity, social_justice, collaboration, innovation_drive, evidence_based, community_autonomy, austerity_scarcity

**Entity linking:**
Every claim is linked back to graph nodes. If a claim mentions "HSE", the pipeline finds the matching agent node and creates an edge.
- For platform 173: 78% linking rate (62 of 79 surface claims)
- Every claim carries a provenance label — `is_ai_generated=True`, `generated_by="21_extract_narrative_layers.py"`

**Dashboard reference:**
- Claims tab (Tab 9): value dimension bar chart, claim table with level filter, extraction summary
- I can filter by surface/implicit, sort by value dimension, click through to source entity

---

## SLIDE 7 — Topic 6: Financial Simulations & Synthetic Data

**What to say:**

Real platforms often have incomplete financial data. Budget figures exist for some nodes but not all. Categories are inconsistent.

To test financial analysis methods — and to show what becomes possible with complete data — I built a synthetic augmentation layer.

**How it works:**
- Takes the real graph (nodes, types, existing relationships)
- Assigns realistic investment amounts based on sector averages and network centrality
- Computes `financial_bias`: does this node receive more or less funding than its centrality predicts?
- Runs a **budget reallocation simulation**: shuffle budgets 1,000 times, compare real allocation to random

**Key finding (for the synthetic platform):**
The real budget allocation is at the **1st percentile** — money sits on less central nodes than random chance would produce. This is almost certainly not maximising impact.

**Stranded assets:**
Nodes with high alignment to goals but zero connections to the core. They are valuable but structurally invisible.
- One example: a project with €600K budget, zero connections to the main network

**Budget × story clusters:**
Which value dimensions are funded vs which are not?
- `cultural_identity`: 13 claims, €17.6M linked funding
- `community_autonomy`: 1 claim, €0
- `austerity_scarcity`: 1 claim, €0
- This is a structural misalignment between narrative prevalence and resource allocation

**Important caveat:**
The numbers are synthetic. The method is real. When real budget data is available, the same pipeline applies immediately.

**Dashboard reference:**
- Budget & Finance tab (only visible for synthetic platform): reallocation test, stranded assets table, budget-by-story-cluster chart
- Structural Change tab: Financial-Perception Bridge section shows the gap directly

---

## SLIDE 8 — Topic 7: Graph Neural Networks (The "What If" Engine)

**What to say:**

The graph tells us what exists. A Graph Neural Network tells us what **could** exist.

**The problem:**
No ecosystem is fully observed. There are always latent connections — collaborations that happened but were not documented, relationships that exist in practice but not in KTool.

The GNN predicts these missing links.

**How it works (plain English):**
- The model learns connection patterns from existing edges
- "Nodes of type X tend to connect to nodes of type Y when they share topics Z"
- It scores every possible missing link by fit to the learned pattern
- Output: a ranked list of recommended new connections

**What limits this:**
GNNs need data. With ~260-650 edges per platform, the model is data-constrained. The predictions are **hypotheses, not findings** — but the structure of the question ("what if we added this link?") is valuable regardless.

**What we do with the predictions:**
- **Component-merging links** get highest priority — they reduce fragmentation
- **Impact simulation**: for each recommended link, recompute PageRank for affected nodes
- **Sensitivity score**: how much does the network change if we add this link?

**For platform 173_synthetic (where we have more data):**
- AUC 0.83-0.86 (strong predictive power for the data available)
- 3 high-confidence recommendations (probability > 0.99)
- Example: "Surf Club ↔ Mental Health & Wellbeing project" — shared thematic overlap, no recorded connection

**Dashboard reference:**
- What-If Simulator tab (Tab 6): recommended links with sensitivity scores
- "If we add this link, the network response is X. If this other one, Y."
- AI-Generated Links tab (Tab 7): all AI-inferred edges with provenance labels

---

## SLIDE 9 — Topic 8: Structural Change (Political Science + Network Metrics)

**What to say:**

All previous layers describe the network. This layer asks: **is change possible?**

I built a framework that bridges network metrics to political science theory. Three frameworks, three questions.

**1. Advocacy Coalition Framework (Sabatier & Jenkins-Smith, 1993)**
- Who holds power, and how stable is their coalition?
- Method: k-core identifies the entrenched core. Perception coherence measures belief alignment.
- Finding: the core exists but is small (12% of nodes). It is a stable minority, not an overwhelming monopoly.

**2. Punctuated Equilibrium (Baumgartner & Jones, 1993)**
- Is the system brittle or resilient?
- Method: robustness gap (targeted vs random removal). High gap = dependence on few irreplaceable nodes.
- Finding: gap is moderate (−0.11). The system depends on its core but not catastrophically.

**3. New Public Governance (Provan & Kenis, 2007)**
- What governance mode fits this topology?
- Method: map network structure to intervention strategy.
- Finding: the ecosystem is fragmented (263 components). Recommendation: **Core Consolidation** — fund a Network Administrative Organisation as a central hub.

**Composite — Change Readiness (platform 173):**
- Leverage: 0.31 — few existing bridges to amplify change
- Plasticity: 0.33 — network is saturated, hard to add links
- Blockage: 1.0 — maximum blockage (33 fragile connectors, 263 isolated nodes, 5 blocked perceptions)
- Lock-in: 0.41 — moderate, core is not overwhelmingly dominant
- **Overall: 0.28 — low readiness**

**The output is not just scores.**
The pipeline generates **~48 specific hypotheses** per platform. Each one names a real node, gives a confidence level, and recommends an action.

Examples from platform 173:
- "Rethink Ireland is an articulation point. If removed, 3 components disconnect. Action: engineer redundant connections."
- "Perception 'Siobhán' has no path to any information node. Action: deploy a channel targeting her sector."
- "The k-core contains Change Clothes Crumlin, Educate Together, Fighting Words, iScoil. This is a small dominant coalition. Action: fund venue-shopping, not direct challenge."

**Dashboard reference:**
- Structural Change tab (Tab 8): five score cards, expandable hypothesis groups, priority actions table
- Financial-Perception Bridge section: the €17.6M → €0 gap lives here
- "You can read every hypothesis, check its confidence, and trace it back to source data."

---

## SLIDE 10 — Summary

**What to say:**

**What exists:**
- A pipeline that turns any KTool export into a multi-layer ecosystem model
- Works for any platform, any subset of the schema
- Structural analysis, narrative extraction, financial-perception bridge, GNN predictions, structural change diagnosis
- A dashboard that puts everything in one place, built for non-technical users

**What this enables:**
- Structural questions that were previously unanswerable
- Cross-domain analysis that was previously manual
- Testable hypotheses instead of intuition
- Traceable provenance — every AI output is labelled as such

**What is next:**
- Validate claim extraction against human coders
- Temporal analysis: if older KTool exports exist, we can detect narrative change over time
- Cross-platform comparison: run on a third platform to test generalisability
- Embed the hypothesis generator as an interactive decision tool inside the dashboard

**Where I fit in this programme:**
- Session 1: I can demo the entire pipeline on a platform of your choice in 5 minutes
- AI Protocol: we need a shared framework for labelling AI outputs — I have one working
- Strategic Mapping: if ALC's stakeholder data is in KTool, I can bring a draft organisational map

**Close:**
"KTool collects the data. The pipeline models the system. The dashboard enables the decision."
