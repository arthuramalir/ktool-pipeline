# Introduction

## An Introduction to K-Tool

K-Tool is a digital platform developed by the Agirre Lehendakaria Center
(ALC) to support systemic innovation. It enables teams to collect,
organize, and analyze information about complex social ecosystems
throughout ALC's methodology.

The platform supports four main phases:

- **Mapping** -- Identify the ecosystem and its actors.

- **Listening** -- Collect qualitative information and identify shared
  perceptions, challenges, and opportunities.

- **Sensemaking** -- Interpret the collected information to understand
  systemic dynamics.

- **Co-creation** -- Design initiatives that respond to the identified
  challenges.

K-Tool stores a wide range of interconnected information, including
actors, organizations, projects, interviews, quotes, perceptions,
challenges, opportunities, values, documents, and the relationships
between them. While this creates a valuable knowledge base, several
limitations prevent the platform from functioning as a true social
digital twin.

## Current Limitations

Although K-Tool contains AI-assisted features and a rich collection of
interconnected information, its underlying architecture remains
fundamentally record-based. Data are distributed across multiple
relational tables, meaning relationships exist only where they have been
explicitly entered by users. Consequently, the platform cannot model the
ecosystem as a unified network, discover latent structural
relationships, or support graph-based analyses of the system.

The Listening phase also presents a significant scalability challenge.
As the number of interviews, quotations, documents, and multimedia
sources increases, manually extracting perceptions, challenges,
opportunities, and other qualitative labels becomes increasingly
time-consuming and difficult to maintain consistently.

Furthermore, many of these labels rely heavily on human interpretation.
Different analysts may assign different perceptions or challenges to the
same piece of evidence, and the platform currently provides no
quantitative metrics to evaluate the confidence, consistency, or quality
of these annotations.

Another limitation is the incompleteness of the available data. Many
entities lack structured attributes such as geographic location,
temporal information, funding, budgets, or other contextual metadata,
reducing the ability to perform comprehensive quantitative analyses.

Most importantly, the platform lacks a mathematical representation of
the ecosystem. Relationships between actors, projects, narratives,
perceptions, and other conceptual layers are stored as database records
rather than mathematical objects. As a result, the platform cannot
directly leverage modern graph algorithms, network science, graph
machine learning, or large-scale semantic reasoning, all of which are
essential components of an accurate digital representation of this kind
of ecosystem.

The key contribution is not a single model but a pipeline that makes the
underlying data usable for graph analysis, narrative interpretation, and
controlled synthetic experimentation. The dataset contains initiatives,
agents, channels, perceptions, quotes, thematic objects, and relation
records. Those objects are not meaningful in isolation; they matter
because of how they connect.

The current 173 analysis shows a network with real structure but
substantial fragmentation. The source data contains roughly 350 records;
after graph construction, claim extraction, and normalisation, the final
graph contains 497 nodes and 653 edges, distributed across 202
connected components. The largest connected component contains 289
nodes (58 percent of the network). The system also contains 199
isolated nodes, 45 articulation points, and 59 bridges. These results
indicate that a relatively small number of nodes carry a
disproportionate share of the network's connectivity.

Perception diagnostics add a complementary interpretive layer. One
perception appears robust, while several others have low purity, low
support, or underdeveloped evidence. This suggests that the narrative
layer contains meaningful structure, but that the interpretive
categories are unevenly supported across the dataset.

The synthetic 173 layer is a controlled experimental extension. It keeps
the same topology as the original 173 dataset but adds invented budget
and investment values so that financial questions can be explored
without treating the synthetic values as empirical truth. In the present
report, this layer is used as a research sandbox rather than as evidence
of real funding conditions.

The dashboard integrates all outputs into a single Streamlit
application with tabs for the decision brief, overview map, geography
and evolution, network layers, health checks, listening data,
AI-generated semantic links, story clusters, narrative space,
perceptions, claims, GNN predictions, narrative simulation, structural
change, and (for synthetic platforms) budget and finance.

# Methods

The entire pipeline is written in Python, with dashboard elements in
Streamlit and Plotly. This section follows the analytical sequence of
the dashboard.

## Data recovery and normalization

The pipeline starts from the cleaned export of data hosted on the KTool
backoffice, and reorganises it into structured tables. This is necessary
because the source database stores relationships in nested, hidden
structures---a single project record may contain embedded references to
people, organisations, perceptions, and thematic tags, all mixed
together. If we flatten these relations too early, we would lose the
connections between them. The pipeline preserves the original structure.

After recovery, the data is normalised into separate entity tables, one
for each kind of record in the ecosystem: people and organisations
(agents), projects and pilots, communication channels, quotes and
information items, perceptions and values, challenges, and thematic
areas. Normalisation makes it possible to join tables reliably, validate
completeness, and run the same analyses repeatedly. Because every table
carries a platform identifier, the pipeline can be run on any platform
at any time, making it straightforward to compare how a platform's
ecosystem evolves, or compare multiple platforms.

When the KTool source data has gaps such as missing locations, budgets,
descriptions, or dates, we can fill them with plausible values. For the
`173_synthetic` dataset, for example, the pipeline assigns Irish town
and county names drawn from a curated list of 84 towns mapped to 28
counties, together with realistic latitude/longitude coordinates, so
that geographic analyses can run even on records that arrived without
location information. Every filled value is tagged with a provenance
label, so the reader can always distinguish original data from synthetic
data. For platform 173, the raw data contains 384 nodes across 12 types
and 496 declared edges; after normalisation all 384 nodes have at least
a label and a node type.

### Example

On the dashboard's Overview tab, the network map shows 384 items and
496 links, with node colours distinguishing agents (blue) from projects
(green) and perceptions (red).

### Limitations

The normalisation step depends on the completeness of the source
export. If relation tables are missing or sparsely populated, the
normalised output will undercount edges. The mapping of nested KTool
fields into flat tables is deterministic but not exhaustive: some
obscure field types in the source schema may be dropped silently.

## Graph construction

The second stage converts the normalized data into a graph
representation. Nodes represent entities, and edges represent relations
between entities. The introduction of edges to the KTool dataset is the
main difference from how it was previously organised. The graph is not a
single undifferentiated network. It is a layered structure with five
distinct edge families.

The first family is declared relational links, which come directly from
the source database: an agent is linked to a project because they
participate in it, or a project is linked to a perception because it was
tagged with it. These 215 edges form the structural backbone.

The second family is listening links, which trace the evidence chain
from a communication channel to an information item (a quote or
observation) to the values expressed in that item. These 136 edges
capture who said what and what values it reflected.

The third family is qualitative narrative links, which connect quotes
through semantic similarity, causal language, contradiction, and shared
themes. These 191 edges exist only for the synthetic dataset because
they require the quote extraction step that runs only on enriched data.

The fourth family is narrative claim edges, produced by the three-layer
claim extraction pipeline. Each claim node is connected to the source
node (quote or perception) that generated it, and to any operational
entities (agents, projects) identified as subject or object of the
claim. These 307 edges add a structured semantic layer on top of the raw
text.

The fifth family is interpretive edges, which are AI-inferred relations
between nodes that share thematic or semantic overlap. These 21 edges
are the most speculative and are clearly labelled so they can be
excluded from any analysis by selecting "Source only" in the dashboard.

This layered design avoids collapsing all evidence into one generic
network. It preserves the origin of each connection and makes it
possible to compare the ecosystem from different analytical angles. Each
edge family can be toggled on or off in the dashboard, and every view
can be further filtered by time window, location, value dimension, and
node type.

The sidebar provides dropdowns for time range (based on best available
date per node) and node type, a radio button to toggle AI-generated data
on or off, and a text input to switch platforms. For example, selecting
"Source only" removes any edge or node flagged as AI-generated
(narrative claims and qualitative narratives), leaving only declared
relational, listening, and interpretive edges.

### Example

For the synthetic dataset, the graph has 498 nodes and 1071 edges across
five families. Toggling "Source only" in the sidebar removes 494
AI-generated edges and leaves 577, revealing the structural core that
exists without any AI inference.

### Limitations

The layered design preserves edge provenance but does not address the
fundamental sparsity of the graph. With only 1071 edges among 498 nodes,
the edge density is 0.004, meaning most node pairs have no recorded
relationship regardless of family. The interpretation of centrality and
community metrics on such a sparse graph should be treated as indicative
rather than precise.

## Structural diagnostics

Before adding narrative layers, simulations, or predictions, the
pipeline runs a set of health checks on the raw graph. These answer
basic questions about the ecosystem's shape.

The first check counts connected components. Platform 173's graph
contains 202 components. The largest holds 289 nodes (58 percent
of the network), meaning most activity is concentrated in a single
cluster while the remaining 208 nodes are scattered across small
isolated groups.

The second check measures edge density: how many connections exist
relative to what would be possible. The 173 graph has a density of 0.005
(0.5 percent of all possible edges are present), which is typical for a
real-world ecosystem---sparse but structured.

The third check identifies orphan nodes that have no relationships at
all. Platform 173 has 199 isolated nodes, giving an orphan rate of 40
percent. These are inactive records, incomplete extractions, or
genuinely disconnected entities.

The fourth check computes degree distributions and identifies the most
connected nodes. Rethink Ireland has the highest betweenness centrality,
followed by a handful of Dublin-based organisations. Removing the top 30
nodes by betweenness centrality fragments the core more than twice as fast
as removing the same number at random, confirming that the network
depends heavily on a small set of gatekeepers.

The pipeline also records node type counts and edge type counts so the
reader can see which entity categories dominate. For platform 173,
agents and projects form the bulk of the mapping layer, while
perceptions and challenges dominate the narrative layer.

### Example

Rethink Ireland has a betweenness centrality of 0.12, the highest in the
network. The dashboard's Health Check tab shows that removing the top
30 such nodes shrinks the largest component by 32.2 percent, versus
13.8 percent for random removal.

### Limitations

Structural diagnostics describe the current state of the graph but do
not distinguish between missing data and genuine disconnection. An
orphan node may be an isolated actor or simply a record that was never
linked. The robustness simulation measures decay under idealised
conditions (sequential removal of the most central nodes) and does not
model real-world failure modes such as simultaneous collapse or
cascading defunding.

## Geographic analysis

From a governance perspective, knowing where activity happens is only
the first question. The more useful question is what kind of activity
happens where, and whether the geographic distribution of attention
matches the distribution of need or opportunity.

Geographic data comes from two sources. For the real 173 dataset, nodes
that have a location field are mapped directly. For the synthetic
dataset, `00_enrich_locations.py` assigns Irish town and county names
together with latitude/longitude coordinates to projects, agents, and
channels that lack them. A separate enrichment step
(`28_enrich_value_dimensions.py`) assigns each node a county name by
looking up its location in a hand-curated mapping of 84 Irish towns to
28 counties, and backfills a value dimension from the node's thematic
areas using a rule-based keyword list of 7 dimensions and 34 keywords.
The same script adds a `sim_year` column for time-lapse animation,
spreading nodes deterministically across years 2019 to 2025 using a hash
of the node's global ID. This allows the time-lapse view to show how
value priorities shift over time even when only one year of real dates
is available.

On the dashboard, the geography tab shows every mapped entity as a point
on an Open Street Map. The default view colours each point by entity
type (project, agent, channel, quote), so the reader can see at a glance
whether a county is dominated by funded projects, by listening channels,
or by organisational actors. Switching to the concentration view scales
each marker by the number of entities at that location, revealing the
urban bias of the ecosystem: Dublin accounts for roughly a quarter of
all mapped nodes, followed by Cork, Limerick, and Galway. Rural counties
such as Leitrim, Longford, and Carlow appear as isolated points.

The underserve view goes a step further by calculating a simple support
ratio: how many agents and channels exist per project in each location.
A low ratio suggests a place where projects outnumber the organisational
infrastructure to support them.

+infleunce

Two additional views connect geography to the value framework. The
county heatmap shades each Republic of Ireland county by its dominant
value dimension, using scattermap markers at county centroids with the
GeoJSON boundary file (26 counties from OpenStreetMap data). The
time-lapse view animates the same county-level data across the years
2019 to 2025. For the synthetic dataset, 101 nodes have both a county
and a value dimension, spread across 13 of the 26 ROI counties.

### Example

On the Geography tab, switching from the default point view to the
county heatmap colours Dublin dark blue (collaboration), Cork green
(social justice), and Galway orange (cultural identity), showing how
value orientations cluster spatially.

### Limitations

The location-to-county mapping is a hand-curated list of 84 towns; any
node with a town name outside this list is left unassigned. The
sim_year values are deterministic hashes, not real dates, so the
time-lapse shows structural change potential rather than observed
temporal trends. The county boundary GeoJSON (26 ROI counties) excludes
Northern Ireland counties entirely.

## Value dimensions

A thematic area such as "youth support" or "community inclusion"
describes what a project is about. A value dimension describes the
deeper rationale: whether the work is primarily about fairness (social
justice), about coordination (collaboration), about new approaches
(innovation drive), about evidence and measurement (evidence based),
about identity and heritage (cultural identity), about local control
(community autonomy), or about doing more with less (austerity and
scarcity). This distinction matters for decision making because two
projects with the same thematic tag can express very different value
orientations, and those orientations shape how stakeholders perceive
success.

The pipeline assigns every project, quote, and perception one of the
seven value dimensions by mapping its thematic areas through a rule-
based lookup of 34 keywords. The mapping is transparent and can be
inspected in the enrichment script.

### Limitations

The value dimension mapping is keyword-based and has no disambiguation.
A project tagged "economy" maps to innovation drive even if the work is
about poverty reduction.

## Financial analysis (synthetic dataset only)

The real 173 dataset contains no budget or investment information — it
is purely a social mapping. The synthetic 173_synthetic dataset was
extended with deterministic financial columns so that budget-allocation
questions could be explored without requiring sensitive real-world
financial data. Every project, pilot, and prototype receives a budget
via a hash-based deterministic formula (projects: €180K–€4.8M, pilots:
€70K–€1.6M, prototypes: €25K–€650K). Every agent receives an
investment_eur_estimate (€15K–€4M) and an investment level label. These
values are reproducible — the same global ID always produces the same
budget — but they have no empirical basis.

Four analyses run on this enriched graph:

**A. Value leverage matrix.** Every initiative is scored by dividing its
betweenness centrality by its budget in millions of euros. The result,
called the leverage score, answers: how much network bridging value does
each euro buy? High leverage = high structural importance per euro spent.

**B. Stranded asset detection.** Initiatives with budget ≥ €500K,
betweenness ≤ 0.001, and degree ≤ 3 are flagged as stranded — expensive
projects that are structurally isolated.

**C. Financial-weighted narrative diffusion.** Standard PageRank is
compared to a personalised PageRank seeded by each agent's
investment_eur_estimate. The financial bias (positive = capital-heavy
narrative zone, negative = structurally prominent but underfunded)
shows where financial capital flows versus where conversations happen.

**D. Budget reallocation simulation.** Budgets are shuffled across
initiatives 1,000 times. If the observed mean leverage is above the 95th
percentile of random allocations, the allocation is "above random." For
the synthetic dataset, the allocation falls in the indeterminate range.

**Prototype project candidates.** A separate script scores existing
projects on five normalised dimensions: budget (32%), betweenness
centrality (28%), perception count (16%), agent count (12%), and topic
count (12%). Projects above 0.6 are flagged as scaling candidates.

### Limitations

Every value in this section is pipeline-generated and has no empirical
basis. The prototype candidate weights (32%/28%/16%/12%/12%) are
arbitrary and unvalidated. The dashboard's budget-by-cluster
cross-tabulation uses fuzzy topic matching. For the real 173 dataset,
no financial information exists and the Budget tab is disabled.

## Quote extraction and semantic linking

The listening layer of the pipeline starts by extracting relevant quotes
from the information table. Every text node (quote, description, or
title) is scored on a 0-to-1 scale using a composite formula based on
text length, the presence of action-oriented language, value-laden
terms, contrast or tension markers, whether the quote is linked to a
channel, to a perception, or to a thematic area. The scoring formula is
deliberately simple: length adds up to 0.35, action markers add 0.25,
value markers add 0.20, contradiction markers add 0.20, and each
structural link (channel, perception, theme) adds 0.10, all capped at
1.0. Quotes scoring below 0.55 are excluded. For the synthetic dataset,
72 quotes pass this threshold with a mean score of 0.83.

Once the quote candidates are selected, the pipeline detects semantic
relationships between them using TF-IDF cosine similarity combined with
linguistic markers. Every pair of quotes from different channels is
compared. If their cosine similarity exceeds 0.15 or they share a
thematic area, an edge is created. The edge type is determined by marker
detection: causal markers (because, therefore, enables, causes, etc.)
produce a causality edge, contradiction markers (but, however, despite,
barrier, etc.) produce a contradiction edge, and neither produces a
similarity edge. Edges with shared themes but low similarity (below 0.3)
are classified as frequency edges. Within the same channel, quotes are
linked in sequence edges ordered by their information ID, capturing the
flow of conversation.

These rules produce 191 semantic edges for the synthetic dataset: 57
sequence edges within channels, 54 causality edges, 40 contradiction
edges, 34 frequency edges, and 6 similarity edges. Every edge is tagged
with the edge family "qualitative_narrative" and a methodological phase
of "listening", so the dashboard can include or exclude them using the
AI-data toggle or the layer selector.

### Example

On the AI-Generated Links tab, selecting "causality" filters the map to
54 purple edges connecting quotes like "funding cuts led to service
closures" and "short-term contracts create instability."

### Limitations

The marker-based edge detection captures only explicit linguistic
signals (because, therefore, but, however). Indirect causality (a quote
implies a cause without using causal language) is missed. The TF-IDF
similarity threshold of 0.15 is arbitrary; raising it would increase
precision but reduce recall. This step currently runs only for platforms
with enriched quote data (the synthetic dataset), because the real 173
data lacks the text coverage needed for meaningful cross-channel
comparison.

## Three-layer claim extraction

The purpose of narrative extraction is to surface what people in the
ecosystem are saying and thinking, so that analysts can understand the
opinion landscape at a glance and use it to create or refine perception
profiles. Perceptions themselves are human-made constructs that
encapsulate population viewpoints; the pipeline's role is to make the
evidence behind them legible, quantifiable, and comparable.

The extraction step processes every node of type information (quotes
from listening channels) that has at least 10 characters of text. For
each such node, a dependency parser converts the sentence into a
subject-verb-object triple. The parser used is hyperbase alphabeta, a
formal grammar engine, not a large language model: it applies a
predefined set of linguistic rules to identify the main predicate and
its arguments. This is deliberately conservative. It avoids the
hallucination risk of generative models and produces claims that are
directly traceable to the source text. The trade-off is coverage: quotes
with complex syntax, incomplete sentences, or non-standard grammar may
fail to parse and produce no claim at all. For the synthetic dataset,
the parser processes about 300 narrative nodes and produces 122 claims.

Alongside each surface claim, the pipeline checks for implicit meanings
using four linguistic markers. Negation markers (not, never, lack of,
fails to) invert the surface assertion: "we do not have funding" becomes
"system lacks funding." Emergency markers (crisis, at risk,
unsustainable) recast the statement as a need for protection: "youth
services are at risk" becomes "youth services need protection."
Conditional markers (if, unless, provided that) reframe it as a
dependency. Emotion markers capture urgency, frustration, hope, and fear
through a curated lexicon. If a quote triggers one or more of these
markers, an implicit claim is created alongside the surface claim. These
inference rules are rigid by design---negation always becomes "lacks",
emergency always becomes "needs_protection"---which makes them
predictable and auditable but means they miss contextual nuance such as
sarcasm.

Every claim is then classified into one or more of the seven value
dimensions using a keyword-based lookup. Each dimension has 15-20
indicative terms. Every keyword match increments a score, and the
dimension with the highest score is assigned. The approach is
transparent but has no disambiguation: a sentence about "cutting-edge
research" matches both innovation drive and evidence based, and a
sentence about "budget cuts" matches austerity and scarcity even if the
speaker is criticising those cuts. The pipeline records the score and
assigns the highest-scoring dimension without applying a threshold.

Once claims are extracted, they are linked to the entities they mention.
Entity linking uses a composite scoring system. First, named entities
and noun phrases are extracted from the claim text using spaCy's
named-entity recognition model and noun chunker, plus a fallback
capitalised-phrase detector. Each mention is then scored against every
known operational entity in the graph (agents, projects, pilots, and
prototypes) using a weighted combination of exact label match (score
1.5), substring match (score 1.0), token overlap (Jaccard index scaled
to 1.2), embedding similarity from a sentence transformer model
(all-MiniLM-L6-v2, 384 dimensions, scaled to 0.5), and a graph context
bonus of 0.2 if the entity already has an edge to the source node. Links
with a composite score above the 0.6 threshold are kept. The threshold
was chosen heuristically and has not been empirically calibrated, which
means both false positives and false negatives are possible.

The linked claims and entities produce three kinds of narrative claim
edges. First, a directed edge connects the source node (quote) to the claim node it generated. Second and third, directed
edges connect the claim node to the subject entity and the object entity
identified during entity linking. In the synthetic dataset, these
produce 303 edges, all tagged with edge family "narrative_claim" and
with the provenance label `is_ai_generated`, so they can be toggled off
in the dashboard.

Pattern analysis groups claims by their source node. If a single quote
generates claims mapped to more than one value dimension, it is flagged
as internally contradictory. For the synthetic dataset, several
quotes produce claims in both social justice and austerity and
scarcity, indicating contested discursive spaces. These patterns help
analysts see which perceptions may need splitting, merging, or
refinement.

### Example

For a quote about "youth services funding," the hyperbase parser
extracts the surface claim "services support youth" (subject = services,
verb = support, object = youth). The negation marker on "lack of
funding" generates the implicit claim "system lacks funding." The
metanarrative classifier assigns it to both social justice (keywords:
youth, inclusion) and austerity and scarcity (keyword: funding),
flagging it as contradictory on the dashboard's Claims tab.

### Limitations

The hyperbase parser has limited coverage of informal speech,
incomplete sentences, and code-switched language. The entity linking
threshold of 0.6 was chosen heuristically, so both false positives and
false negatives are possible. The implicit claim detection captures only
the four marker types and misses rhetorical questions, analogy, or
understatement. All AI-generated edges are labelled but downstream
metrics treat them identically to source edges unless the user selects
"Source only."

## Story clustering and narrative profiles

The quote clusters produced in this step are the closest the pipeline
comes to a "narrative space" — a map of what people in the ecosystem
are talking about and how those conversations group together. These
narrative spaces are the raw material analysts use to construct
perception profiles: each cluster represents a candidate viewpoint that
may warrant its own perception, and the claims within it reveal the
value dimensions driving that viewpoint.

After semantic edges are created, the pipeline clusters quotes into
story groups using TF-IDF vectorisation and agglomerative clustering.
The number of clusters is set dynamically based on quote count, ranging
from 2 to 8. For the synthetic dataset, 72 quotes are grouped into 8
clusters.

Each cluster receives a label derived from its most frequent thematic
area, a representative quote (the one with the highest selection score),
and a profile of the channels and themes it covers. Across the 8
clusters, the average selection score ranges from 0.68 to 0.95.

These clusters are then linked to the claims extracted in the previous
step by matching the claim's source node ID to the information ID in the
cluster table. The resulting cluster-claim matrix shows each cluster's
value dimension profile. If a single cluster contains claims in more
than one value dimension, it is flagged as internally contradictory. For
the synthetic dataset, some clusters span both social justice and
innovation drive claims. This flags for the analyst whether a cluster
represents a single coherent perception or an amalgamation that may need
splitting.

A separate narrative diffusion analysis (script 26) computes PageRank
for every node in the full graph and a diffusion bias score that
contrasts personalised PageRank between the two largest agent types. The
top-ranked node by PageRank is typically the most connected agent in the
mapping layer, while the bias score reveals which voices carry further
in one agent network versus another.

### Example

Cluster 3 ("youth engagement") contains 14 quotes about education and
social inclusion, with claims split between social justice (8 claims)
and innovation drive (3 claims). It is flagged as internally
contradictory. On the dashboard's Story Clusters tab, the cluster's
representative quote is shown alongside its value dimension profile.

### Limitations

Clustering uses TF-IDF similarity, which captures word overlap but not
synonymy or discourse framing. Clusters with fewer than three quotes
have unreliable profiles. The number of clusters (between 2 and 8) is
set by a heuristic formula, not validated against a gold standard. The
narrative diffusion PageRank is computed on the full graph including
AI-generated edges unless filtered.

## Opinion simulation (Friedkin-Johnsen)

The opinion simulation treats each story cluster as a node in a smaller
network. A cluster's opinion is a seven-dimensional vector, one entry
per value dimension, built by counting how many of its claims fall into
each dimension and normalising the result. Two clusters are connected if
a semantic similarity edge exists between any quote in the first cluster
and any quote in the second. In the synthetic dataset, several clusters
are connected through shared thematic areas even though they express
different value dimensions, creating a cross-cluster influence network.

The simulation uses the Friedkin-Johnsen model, a well-established
framework for how opinions evolve in a connected group. Each cluster has
an anchoring opinion (its own initial profile) and is influenced by the
opinions of the clusters it is connected to. The balance between
anchoring and influence is controlled by a stubbornness coefficient set
proportional to the cluster's claim count: clusters with 15 or more
claims are highly stubborn (stubbornness near 0.9), while clusters with
only 2-3 claims are highly persuadable (stubbornness near 0.2). The
model iterates until the opinions stabilise, typically within 20-50
iterations.

The baseline answers the question: if these story clusters were left to
interact naturally, what would each cluster's final opinion profile look
like? The dashboard visualises this as a radar chart with seven axes,
one per value dimension.

From the baseline, the pipeline runs intervention simulations. For each
cluster in turn, it adds an external agent with a neutral opinion (equal
weight across all seven dimensions) and a low stubbornness of 0.2,
connects it to that cluster, and re-runs the simulation. The question
is: if someone new entered the ecosystem and engaged primarily with this
one cluster, how would every other cluster's opinion shift? For the
synthetic dataset, the largest opinion shifts occur when connecting the
agent to the most central cluster (by cross-cluster edge count), with
max deltas of approximately 0.02 to 0.05 in the affected dimensions.

A second, more detailed simulation (script 25, GBCM-FJ) incorporates
perception nodes directly, using their embedding similarity, value
dimension correlation, and claim counts to compute influence. This
model runs on the full graph (not just clusters) and evaluates
counterfactual edges from the GNN link prediction output. It computes
how disagreement, diversity, and perception PageRank change when a
proposed link is added.

### Example

The Narrative Simulation tab shows an 8-cluster radar chart. Cluster 1
("community development") has high social justice and low innovation
drive. Connecting a neutral agent to it shifts Cluster 4 ("digital
inclusion") toward social justice by 0.04 on the normalised scale---a
small but detectable influence cascade.

### Limitations

The stubbornness formula is proportional to claim count, making
clusters with few claims highly persuadable regardless of whether those
claims are strongly held. The model has no temporal dimension. The
intervention tests only a neutral agent, not a funder, advocate, or
traditionalist. Semantic edges between clusters capture textual
similarity, not organisational ties.

## Perception diagnostics

Perceptions in KTool are human-made profiles that encapsulate the
opinions of a population segment. The pipeline does not generate them
automatically. Instead, it evaluates each existing perception on five
health metrics so analysts know which are well-supported by evidence and
which need revision. Internal coherence measures the mean cosine
similarity of semantic edges between quotes within the same perception.
Purity measures what fraction of each quote's top-5 semantic neighbours
belong to the same perception. Source entropy measures the diversity of
channels feeding each perception. Contradiction density measures the
fraction of intra-perception edges that are contradictions. Each
perception also receives a status flag: "Robust," "Underdeveloped (< 3
quotes)," "Low coherence," "Single channel," "High internal
contradiction," or "Low purity."

The claim extraction output feeds directly into this diagnostic layer:
an analyst reviewing a perception with "Low purity" can drill into the
claims behind its quotes to see which value dimensions are pulling it in
different directions, then decide whether to split it into two more
coherent profiles or merge it with a neighbouring perception. For
platform 173, the diagnostics reveal one robust perception with high
coherence and moderate quote count, alongside several perceptions with
fewer than 3 quotes or low internal coherence. The dashboard's
Perceptions tab displays these results as a colour-coded table.

A second phase evaluates narrative alignment: how distinct each
perception's value dimension profile is from the others. The pipeline
computes a silhouette score (how far a perception's claim profile is
from its nearest neighbour) and flags narratively redundant pairs (claim
cosine similarity above 0.80). This helps identify perception categories
that may be analytically indistinguishable and candidates for merging.

### Example

On the Perceptions tab, "community wellbeing" has an internal coherence
of 0.72 (high), a purity of 0.55 (moderate), and a silhouette score of
0.31---distinct from "youth development" but not sharply separated.

### Limitations

Perception diagnostics depend on the availability of semantic edges
between quotes. If the quote set is small or the semantic edge detector
produced few edges, the coherence and purity estimates are unreliable.
The status flags use heuristic thresholds (coherence below 0.40 is
"low," purity below 0.30 is "low") that have not been validated against
human judgement.

## Link prediction and project recommendations

The graph contains only the relations recovered from source data, but a
government or funding agency can actively create new connections: fund a
project in a new region, broker a partnership between two organisations,
or convene stakeholders around a shared challenge. The pipeline uses a
graph neural network to identify which of these possible new connections
would have the largest structural and narrative impact, and then ranks
them by practical feasibility for decision-makers.

The first step is preparing the graph for the model. Every node is given
a feature vector that combines its position in the network (degree,
centrality, k-core level, whether it is an articulation point), its type
and phase (one-hot encoded), its textual description (converted to a
fixed-size hash vector of 256 dimensions), and its semantic embedding
from the sentence transformer model (384 dimensions). Every edge is
given an omega confidence score: omega = 0.45 x provenance (how the edge
was created) + 0.35 x alignment (cosine similarity of endpoint
embeddings) + 0.20 x variance stability (how consistent the alignment
is). Edges with omega below 0.6 are excluded from training. The GNN
directory also records feature slices (numeric, categorical, text hash,
semantic) so the contribution of each feature type is traceable.

The link prediction model is a graph autoencoder. A two-layer GCN
encoder compresses each node into a 64-dimensional embedding. A dot-
product decoder takes any pair of node embeddings, computes their dot
product, and passes it through a sigmoid function to produce a
probability between 0 and 1. The model is trained on 70 percent of the
existing edges, with 15 percent for validation and 15 percent for
testing. On the synthetic dataset with 498 nodes and 1071 edges (1625 after
bidirectional expansion and omega filtering), it achieves a best
validation AUC of 0.986, but the test AUC drops to 0.72, confirming
overfitting on a small graph.

Once trained, the model scores all possible pairs of nodes that are not
already connected. Only pairs where both endpoints belong to the mapping
layer agents and projects are retained, because these are the entity
types where a new connection translates into an actionable
recommendation. The top 50 candidates are then scored by a structural
impact engine that answers: what would actually happen if this link were
added?

The impact engine temporarily adds each candidate edge to a copy of the
full graph and measures what changes. It runs a breadth-first search
from both endpoints before and after the addition, counting how many new
perception nodes and claim nodes become reachable within three hops. It
checks whether the link unlocks value dimensions that were previously
inaccessible. It classifies the bridge type: a learning bridge unlocks
entirely new value dimensions, a cleavage breach breaks through a
structural divide, a coalition reinforcement connects nodes that already
share values, and a structural link adds no new narrative pathways. It
also checks whether the link merges two separate connected components,
whether either endpoint is an articulation point, and how semantically
compatible the two nodes are.

All of these factors are combined into a composite governance score.
Narrative impact (new pathways and new value dimensions) carries the
most weight at 35 percent, followed by bridge type at 20 percent,
component merger at 15 percent, and degree, articulation-point status,
and domain compatibility at 10 percent each. Each candidate also
receives a feasibility badge: fundable (strong semantic alignment),
cross-domain (moderate alignment with structural value), or topological
(weak alignment, purely structural connection). The dashboard's GNN
Predictions tab displays the top recommendations sorted by this
composite score, with the badge, bridge type, number of new pathways,
and a rationale.

A separate scoring process identifies prototype project candidates from
the full set of project nodes. Each project is scored on a weighted
combination of its budget (32 percent), its betweenness centrality (28
percent), the number of perceptions it is connected to (16 percent), the
number of agents involved (12 percent), and the number of thematic
topics it covers (12 percent). The dashboard's Suggested Projects
section displays the top candidates with their estimated budget, score,
and a suggestion reason.

### Example

The GNN Predictions tab shows a recommended link between "Dublin Youth
Collective" (agent) and "Creative Communities Programme" (project) with
a governance score of 0.81. Adding it would unlock 3 new perception
paths and bridge the social justice and cultural identity value
dimensions. The badge is "Fundable."

### Limitations

The GNN achieves high AUC but on a small graph, so the ranking is more
reliable than the absolute probability values. The narrative impact
scores depend on a BFS horizon of three hops, which may miss longer-
range effects. The feasibility badge reflects semantic compatibility,
not organisational or political readiness. The budget data for the
synthetic dataset is pipeline-generated.

## Structural change feasibility

The question that motivates this section is not "what should change?" but
"given the current relational structure, what kinds of change are even
possible?" This is a fundamentally different question from the link
prediction in the previous section, and it is grounded in political
science rather than graph theory alone. The Advocacy Coalition Framework
(Sabatier and Jenkins-Smith, 1993) holds that policy subsystems are
organised around competing coalitions bound by shared beliefs, and that
major change requires external shocks or the accumulation of minor
perturbations over long periods. Punctuated Equilibrium Theory
(Baumgartner and Jones, 1993) describes how long periods of policy
stability maintained by dense, insular policy monopolies are interrupted
by brief bursts of rapid reconfiguration when excluded actors reframe
the policy image and find new institutional venues. Path dependency
theory (Pierson, 2000) warns that early institutional choices and
funding patterns lock ecosystems into trajectories that become
increasingly costly to reverse. These frameworks provide the interpretive
lens for the metrics below.

The pipeline computes four composite dimensions from the network
metrics, each mapped to a political science construct.

The first is leverage. In network terms, nodes with high betweenness
centrality occupy structural holes and act as gatekeepers of information
flow. In political terms, they are policy brokers and boundary spanners
who can mediate across coalitions, build consensus, and accelerate
narrative diffusion (Gould and Fernandez, 1989). The pipeline identifies
the top betweenness nodes and top bridge agents as potential leverage
points. For the synthetic dataset, the leverage score is 0.775,
indicating a moderately centralised ecosystem with clear hubs---most
betweenness is concentrated in a handful of Dublin-based organisations.
Strengthening their brokerage role would accelerate diffusion, but
losing them would fragment coordination.

The second is plasticity, which measures the network's spare capacity to
form new connections without collapsing existing structures. Nodes with
degree one or two are peripheral and easy to rewire; nodes with degree
three to five are candidates for new brokering roles. The count of
high-confidence GNN link predictions also contributes, because each
predicted link represents a latent connection that could be activated.
A high plasticity score means the periphery has abundant unused
connection potential---there are many isolated or weakly connected nodes
waiting to be integrated. For the synthetic dataset, the plasticity
score is 1.0, the maximum, reflecting a sparse periphery with ample
rewiring capacity.

The third is blockage. Fragile connectors are articulation points that
also carry bridge edges: removing them would both disconnect the network
and break a unique link. These are institutional bottlenecks, single
points of systemic failure. Isolated nodes have degree zero and
participate in no relationships at all. Small components with three or
fewer nodes cannot sustain meaningful diffusion. Blocked perceptions are
perceptions that have no path to any information node in the graph,
trapping them in discursive silos---a structural analogue of the
discursive cleavages and ideological polarisation described in
polarisation research. The blockage score is 0.562, meaning the network
has a moderate fraction of single points of failure.

The fourth is lock-in or path dependency. A network with a very dense
core (a high fraction of nodes in the maximum k-core layer) resembles a
policy monopoly: a dominant coalition controls the most connected
positions, and new entrants or alternative viewpoints struggle to gain
traction (Baumgartner and Jones's concept of the "policy monopoly"
maintained by a supporting policy image and an insulated venue). The
robustness gap---how much faster the network decays under targeted
attack versus random failure---reinforces this picture: if removing the
most central nodes causes disproportionate damage, the network depends
heavily on those hubs and is structurally brittle. The synthetic dataset
has a lock-in score of 0.161 (low, meaning the core is small: only 2
percent of nodes are in the innermost k-core of 8) and an overall change
readiness score of 0.743.

These four dimensions are normalised to scores between zero and one and
combined into an overall readiness score, where blockage carries the
highest weight. The dashboard's Decision Brief tab displays all four
scores as horizontal bars with colour coding (green for favourable,
amber for moderate, red for concerning), each accompanied by a plain-
language explanation: leverage becomes "fraction of nodes that are well-
connected hubs," blockage becomes "fraction of nodes that are single
points of failure," and lock-in becomes "how stuck the network is."

From these scores, the pipeline generates named hypotheses grounded in
the political science frameworks above. A hypothesis about a high-
betweenness node cites policy brokerage and boundary spanning theory
(Gould and Fernandez). A hypothesis about the innermost k-core layer
cites the Advocacy Coalition Framework and the concept of a policy
monopoly (Sabatier and Jenkins-Smith; Baumgartner and Jones). A
hypothesis about blocked perceptions cites discursive cleavages and
ideological polarisation. Each hypothesis is assigned a confidence level
(high or medium) based on thresholds derived from the data distribution,
and only high-confidence hypotheses are promoted to recommendations. The
dashboard's Structural Change tab lists these hypotheses alongside their
framework reference, confidence level, and the specific node identifiers
they reference.

The overall assessment for the synthetic dataset reflects an ecosystem
with moderate structural openness (readiness 0.743), a handful of high-
leverage nodes concentrated in Dublin-based organisations, a sparse
periphery that offers ample rewiring capacity, and a small but dense
core that would resist rapid reconfiguration. The most interesting
structural question is not whether the network can change---the metrics
say it can---but whether the core's resistance and the periphery's
fragmentation will produce a punctuated equilibrium dynamic (long
stability punctuated by brief reconfiguration) or a gradual accretion of
new ties. Answering that requires temporal data that does not yet exist.

### Example

The Structural Change tab shows leverage at 0.78 (green), plasticity at
1.0 (green), blockage at 0.56 (amber), and lock-in at 0.16 (green). The
caption reads: "Network is open to reconfiguration. Blockages (fragile
connectors, small components) need attention."

### Limitations

The change readiness scores are computed from a static graph snapshot.
Without temporal data, the pipeline cannot distinguish between a network
that is slowly rewiring and one that is frozen in place. The political
science mappings (betweenness = broker, dense core = policy monopoly)
are analogical, not proven by the data---they provide an interpretive
frame, not a causal test. The Gould-Fernandez brokerage typology is not
fully implemented (the pipeline uses simple betweenness centrality, not
group-attribute brokerage classification). The blockage score weights
each type of blockage equally, which may not reflect their relative
policy significance.

## Dashboard implementation

The dashboard is the main user-facing interface for the outputs. It is
built with Streamlit and Plotly and presents the analysis through 14 or
15 tabs, depending on whether the platform has synthetic financial data.
The design goal is to make the analytical layers understandable to
readers who do not need to inspect code or raw tables.

### Example

The sidebar's time filter defaults to the full date range of the data
(2019 to 2025 for the synthetic dataset). Selecting "Source only" and
narrowing to 2023-2024 immediately reduces the graph from 498 nodes to
the subset with real or synthetic dates in that window, and removes all
AI-generated edges from the view.

### Limitations

The dashboard reads pre-computed CSV and JSON files from the analysis
directory. If a script has not been run for the selected platform, the
corresponding tab shows a "data not available" message rather than an
error. Performance degrades for very large graphs (above 1,000 nodes)
because the network visualisation uses a force-directed layout computed
in real time. The dashboard is designed for exploratory analysis, not
for reproducible reporting: it does not export figures or log user
interactions.

# Discussion

The main substantive finding is that the ecosystem has meaningful
structure, but that structure is unevenly distributed. There is a clear
core, there are many bridges, and there are many isolated nodes. This
combination means the network is neither trivial nor fully integrated.
It is a functioning socio-semantic system with visible structural
dependencies.

From a research perspective, this matters in three ways. First, it shows
that graph analysis is appropriate for the dataset because there is
enough structure to study. Second, it shows that narrative and
perception layers add value because they reveal interpretive patterns
that are not visible in the graph alone. Third, it shows that the
current system still has data limitations, especially where relation
coverage is incomplete or where categories remain weakly supported.

The results also imply that interventions should probably focus on
improving relational completeness, strengthening bridge entities, and
standardizing how narrative evidence is linked to structured records. In
a practical sense, a better-connected data model would make later
analysis more reliable and more interpretable.

# Limitations

The current report and pipeline are limited by the available source
data. Some relation families are only partially populated, and some
interpretive categories have weaker support than others. That means the
results should be read as a careful description of the present dataset,
not as a full account of the entire underlying ecosystem.

A second limitation is that the synthetic financial layer is
experimental. It is useful for scenario analysis, but it should not be
mistaken for real-world financial evidence. Any paper using this layer
should state that the budgets and investment values were invented to
support controlled testing.

A third limitation is that the current outputs are strongest for
description and structure. They are less able to support causal claims
about change, because the project is not yet based on a longitudinal
sequence of repeated snapshots. For that reason, the present report
should be treated as a foundation for future work rather than as a final
causal explanation.

# Conclusion

The ALC K-Tool project provides a structured way to move from a raw
ecosystem record dump to an interpretable analytical system. The value
of the work is in the combination of recovery, normalization, graph
construction, narrative analysis, and dashboard presentation.

The Platform 173 results show a sparse and fragmented network with a
small but meaningful core, many isolated nodes, and substantial
sensitivity to targeted node removal. The perception results show that
the narrative layer contains both robust and weakly supported
categories. The synthetic 173 layer adds a controlled way to test
financial-style questions without changing the original topology.

Taken together, these results support a straightforward conclusion: the
dataset is now rich enough for serious descriptive and interpretive
analysis, and the pipeline is strong enough to support future research
on structure, narrative, and investment alignment. The most important
next step is not a more complicated model for its own sake, but better
data coverage, clearer relational labeling, and repeated observations
over time.
