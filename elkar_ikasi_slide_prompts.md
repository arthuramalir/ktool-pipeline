# Slide Prompts — Elkar Ikasi 2026

Audience: Social scientists, non-technical, familiar with ALC's work and KTool.
Tone: Conceptual, not technical. Explain what each layer does and why it matters, not how it works.
No code. No model architectures. No dataframe screenshots. Every slide should answer "so what?".

---

## SLIDE 1 — Title

**Prompt:**

A single, clean title slide. White background, minimal.

Main title: "From Data to Diagnosis: What the KTool Pipeline Actually Does"

Subtitle: "Arthur Amalir — Internship Presentation — Elkar Ikasi 2026"

Bottom: small ALC logo or text "ALC · Urdaibai"

Design: Sans-serif font (Lato or Roboto). Title in dark navy (#1B2A4A). Subtitle in medium grey (#5A6A7A).

---

## SLIDE 2 — The Problem: KTool Today

**Prompt:**

Split layout: 60% text left, 40% icon/graphic right.

**Headline (top left, large):** "KTool stores data. It does not model the system."

**Four bullet points below, each with a small icon:**

🔲 "Data is **disconnected** — agents, projects, perceptions, channels, quotes sit in separate tables. You cannot see the system as a whole."

🔲 "Analysis is **manual** — every new question needs a SQL query or a spreadsheet. Structural questions (who is central? who is fragile?) are not answerable at all."

🔲 "Insight is **invisible** — there is no network map. No "what if" engine. No way to cross-reference narratives with budgets or perceptions with information flow."

🔲 "Governance questions are **untestable** — 'where would a small intervention have the biggest effect?' and 'what blocks change?' are answered by intuition, not data."

**Right column:** A simple illustration — a set of disconnected dots/labels (AGENT, PROJECT, PERCEPTION, QUOTE, BUDGET) floating apart, with no connecting lines. Maybe a faint "X" or broken-link icon between them.

**Footer line, smaller:** "The tool stores relationships in rows. But a row is not a model."

---

## SLIDE 3 — What a Social Scientist Actually Wants to Know

**Prompt:**

Full-width layout, three columns.

**Headline (centered, top):** "You do not want a database. You want a model."

**Three columns, each with:**

**Column 1 — Icon: network nodes**
**Headline:** "Structure"
**Body:** "Who is connected to whom? Which actors are central, which are peripheral? Is this ecosystem distributed or hierarchical? Where are the bottlenecks?"

**Column 2 — Icon: speech bubble with thought**
**Headline:** "Meaning"
**Body:** "What are people actually saying? Which stories dominate? Which narratives exist but go unheard? How do different claims relate — agree? contradict? cause and effect?"

**Column 3 — Icon: branching arrow / fork**
**Headline:** "Possibility"
**Body:** "Given this structure and these meanings — is change possible? Where would a small intervention have the biggest effect? What blocks propagation? Is the system locked in?"

**Footer (centered, italic):** "These are social science questions. They need a social science answer — not a SQL answer."

---

## SLIDE 4 — What the Pipeline Adds: The Missing Layer

**Prompt:**

Flow diagram, horizontal, 4 boxes connected by arrows.

**Title (top):** "Raw Data → Relational Model → Structural Diagnosis → Actionable Hypothesis"

**Box 1 — light blue:** "1. **Extract & Link**" — "Agents, projects, perceptions, channels, quotes, budgets: everything becomes a node with typed connections. 500+ nodes, 650+ edges per platform."

**Box 2 — light green:** "2. **Analyse Structure**" — "Centrality, k-core decomposition, robustness stress tests, articulation points. Who is essential? Who is fragile? What happens if key actors leave?"

**Box 3 — light yellow:** "3. **Extract Narratives**" — "Three-level story framework: surface claims, implicit assumptions, metanarratives. Value dimensions (cultural identity, social justice, collaboration, etc.). Entity-linked to the network."

**Box 4 — light coral/salmon:** "4. **Diagnose Change Feasibility**" — "Political science frameworks (ACF, PET, NPG) applied to network metrics. ~48 specific hypotheses per platform. Financial-perception bridge: who gets funded vs who is heard."

**Below the flow, a single line:** "Each layer is human-verifiable. Every AI inference carries a provenance label: `source_data` or `ai_inferred`. Nothing is hidden."

---

## SLIDE 5 — Three Levels of Narrative (The ALC Method, Operationalised)

**Prompt:**

Three stacked horizontal bands, like geological layers.

**Title (top):** "From Listening Data to Structured Claims"

**Layer 1 — top, lightest shade, "Surface":**
- Icon: microphone
- "What people **say** — explicit statements"
- "Extracted as Subject → Verb → Object triples"
- "Example: *public institutions seek measurable impact*"
- "79 claims extracted for this platform"

**Layer 2 — middle, medium shade, "Implicit":**
- Icon: thought bubble / subtext
- "What people **assume** — unstated premises"
- "Detected through linguistic markers: negation, conditionals, emergency framing, emotion"
- "Example: *innovation is risky* (stated indirectly through caution framing)"
- "43 implicit claims"

**Layer 3 — bottom, darkest shade, "Metanarrative":**
- Icon: magnifying glass over map
- "The **big picture** — underlying worldviews across multiple claims"
- "Aggregated from patterns: which value dimensions recur? Which coalitions share beliefs?"
- "Example: a metanarrative of *scarcity vs solidarity* running across multiple sessions"

**Right side of each layer:** a small "N" icon with a count: 79 / 43 / aggregated.

**Footer (below all three):** "Every claim is linked to the entity that made it and the value dimension it expresses. No claim floats — everything is traceable back to source text."

---

## SLIDE 6 — Structural Change: Political Science in the Pipeline

**Prompt:**

Three vertical columns, each representing a theory applied.

**Title (top):** "Network Metrics + Political Theory = Diagnosing Whether Change Is Possible"

**Column 1 — "Advocacy Coalition Framework" — Sabatier & Jenkins-Smith (1993)**
- Icon: two overlapping groups
- "Core idea: policy subsystems contain competing coalitions bound by shared beliefs. Change requires external shocks or cross-coalition learning."
- "In the pipeline: k-core decomposition identifies the dominant coalition. Perception coherence measures belief-system tightness."

**Column 2 — "Punctuated Equilibrium" — Baumgartner & Jones (1993)**
- Icon: flat line with sudden spike
- "Core idea: long stability interrupted by brief bursts of non-linear change. Policy monopolies maintain control via a dominant narrative + closed venue."
- "In the pipeline: robustness gap (targeted vs random removal) reveals system brittleness. Semantic betweenness measures whether alternative framings exist."

**Column 3 — "New Public Governance" — Provan & Kenis (2007)**
- Icon: network of dots with one central hub
- "Core idea: public value is co-produced through networks, not hierarchy. Systems converge toward stable governance modes."
- "In the pipeline: articulation point analysis reveals lead-organization bottlenecks. Intervention maturity matrix maps topology to appropriate governance mode."

**Footer:** "The output is not just a score. It is ~48 specific, actionable hypotheses — with node names, types, confidence levels, and recommended actions."

---

## SLIDE 7 — The Dashboard: Designed for Decision-Makers

**Prompt:**

Full-width screenshot of the dashboard (the Overview tab showing the network map, ideally). If no screenshot available, describe:

**Title (top):** "One Dashboard. Ten Lenses."

**Below the title, a 2×5 grid of small cards, each with an icon and short label:**

Row 1: Overview | Health Check | Listening | Story Clusters | Perceptions
Row 2: What-If Simulator | Network Layers | AI-Generated Links | Structural Change | Claims

(Optional: highlight the Structural Change tile in a different color — that's the newest addition.)

**Right side or bottom, callout box:** "Built for non-technical users. Every chart is annotated. Every section has a 'so what?' explanation. Expand any data table to verify — but you should not need to."

---

## SLIDE 8 — What This Enables: Governance Questions Answered

**Prompt:**

Four question-answer pairs in a clean 2×2 grid.

**Title (top):** "Before the Pipeline — After the Pipeline"

**Top-left — Before:** "Where are the bottlenecks?" **After:** "Here are the 10 articulation points — by name — with the specific components they connect."

**Top-right — Before:** "Which stories dominate?" **After:** "Seven value dimensions mapped. `Cultural identity` dominates discourse (13 claims) and funding (€17.6M). `Community autonomy` has €0."

**Bottom-left — Before:** "What if a key actor leaves?" **After:** "Stress test shows the core shrinks 11% faster under targeted removal than random. Here are the three actors whose exit would fragment the network."

**Bottom-right — Before:** "Is change possible?" **After:** "Change readiness score: 0.28 (low). The ecosystem is fragmented (263 small components). Blockage score is 1.0 — you need new bridges, not stronger existing ones."

**Footer (bold, centered):** "These are not guesses. They are computed from the data, framed by political science, and traceable back to source."

---

## SLIDE 9 — Where I Fit in the Programme

**Prompt:**

Split layout: left 60%, right 40%.

**Left column — "I can contribute to:" with three items:**

🎤 **Session 1 — "What does KTool actually do?"** (09:30-11:00, Day 1)
- 5-minute live demo of the pipeline on real data
- Show the network map, the claim extraction, the stress test, the change readiness scores
- Concrete answers to "what does it offer that other tools don't?"

🧠 **AI Working Protocol** (15:00-16:00, Day 1)
- I have built verification protocols for every AI layer
- Every output carries a provenance label (`source_data` or `ai_inferred`) and a confidence score
- I can speak to where AI adds value vs where it creates noise — and what governance that implies

🗺️ **Strategic Mapping** (16:00-17:00, Day 1)
- The team exercise asks "who are our partners, where is our density, what are our gaps?"
- If ALC's own stakeholder data is available in KTool, I can pre-run the pipeline and bring a draft organisational map as the starting point
- Turn a 45-minute discussion into a 15-minute validation

**Right column — callout box with light background:**
**"Interested? Let's set up a 10-minute demo during lunch or coffee."**
Small text below: "The dashboard runs on any browser. No installation needed."

---

## SLIDE 10 — Closing

**Prompt:**

Minimal, centred.

A single sentence, large: **"The tool stores data. The pipeline models the system. The dashboard enables the decision."**

Below, two lines of smaller text:

"Arthur Amalir — [email if appropriate]"

"Slides, dashboard link, and full methodology available on request."

Bottom: ALC logo.

(Optional: QR code linking to the deployed dashboard URL.)
