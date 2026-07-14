# Deep Research: Three-Level Narrative Extraction via Semantic Hypergraphs for Policy Network Analysis

## Project Context

I am building a decision-support pipeline for a government innovation agency (ALC, Ireland). The pipeline:

1. **Builds a heterogeneous graph** from ~6 years of stakeholder consultation data. Nodes = agents (organisations, community groups), initiatives (projects, pilots, prototypes), information items (quotes, testimonies), perceptions (themes/opinions from workshops), challenges, channels (media, reports), values. Edges = declared links, listening evidence chains, and NLP-inferred relationships.

2. **Runs analytics** — centrality, k-core, articulation points, GNN link prediction (R-GCN), perception diagnostics, robustness simulation, intervention simulation.

3. **Visualises** in a Streamlit dashboard for non-technical government decision-makers.

## The Missing Piece: Narrative Depth

Currently, my NLP pipeline has two disconnected levels:

**Level 1 — Shallow similarity (working):**
I encode node descriptions with SentenceTransformer (all-MiniLM-L6-v2), producing 384-dim embeddings stored as node features for the GNN. This gives 100% semantic coverage but only captures topical relatedness — cosine similarity between bag-of-words sentence embeddings. No structure, no claims, no relations.

**Level 2 — Meaning extraction (broken):**
I have a hand-rolled script (`10_semantic_hypergraph_claims.py`) that uses spaCy dependency parsing to extract subject-verb-object triples and formats them as graphbrain-style hypergraph expressions `(verb/P subject/C object/C)`. It produces ~139 claims per platform but:
- Entity linking is naive string matching — most claims anchor to "Implicit Concept" because operational node labels ("Rethink Ireland") rarely appear verbatim in narrative text ("the funding body")
- The SVO extraction is basic, noisy, and doesn't handle negation, conditionals, or implied claims
- The output is never fed into the GNN — it just sits in a CSV

## The Framework: Three Narrative Levels

I want to extract and integrate three levels of narrative from the consultation text, drawing on discourse analysis and the Advocacy Coalition Framework (Sabatier & Jenkins-Smith, 1993):

### 1. Surface Narrative
What participants explicitly say. Direct claims extractable from syntax:
> "The HSE should fund community mental health projects."
→ `(fund/Pd.so HSE/Cp (community/Cc mental_health/Cc projects/Cc))`

**Extraction method:** Dependency parsing + semantic role labeling. This is what hyperbase-parser-ab does natively.

### 2. Implicit Narrative
What participants suggest, fear, or leave unsaid, detectable in their discourse through:
- **Negated claims** ("We're not getting enough support") → implies `(withhold/Pd.so support_system/Cc us/Cp)`
- **Conditional claims** ("If the funding doesn't come soon, we'll have to close") → implies `(threaten/Pd.so funding_cut/Cc closure/Cc)`
- **Emotion/stance markers** ("It's devastating that youth services are being cut") → implies `(harm/Pd.so funding_cut/Cc youth_services/Cc)`
- **Contrastive discourse** ("They focus on Dublin, but rural communities need it most") → implies spatial inequity claim
- **Lexical choice as stance** — calling something a "pilot" vs a "programme" vs a "scheme" reveals different assumptions about its permanence and legitimacy

**Extraction method:** Inference rules over the surface hypergraph + stance/emotion lexicons + LLM prompting for implicit claim detection.

### 3. Metanarrative / Deep-Seated Belief
Underlying values that organize surface and implicit claims into coherent worldviews:
- **Social justice** — claims about equity, inclusion, marginalised groups, fair distribution
- **Innovation as driving force** — claims about novelty, transformation, disruption, piloting
- **Cultural identity** — claims about community, tradition, local knowledge, place
- **Austerity/scarcity** — claims about limited resources, efficiency, value for money, sustainability
- **Subsidiarity** — claims about local control, bottom-up decision making, community autonomy

**Extraction method:** Clustering surface and implicit claims by shared value dimensions. Each cluster defines a metanarrative. This can be done via:
- Topic modeling over claim embeddings
- Dictionary-based value coding (value lexicons mapped to ACF belief levels)
- Graph propagation — which claims co-occur in the same actor's discourse, which claims are structurally central in the semantic hypergraph

## The Tool: Semantic Hypergraphs (Hyperbase)

The `hyperbase` library (successor to graphbrain, actively maintained, v0.10.0, April 2026) with the `hyperbase-parser-ab` plugin (alphabeta parser, v0.3.0) provides:

1. **Alpha stage** — a multilingual neural token classifier (DistilBERT-based) that assigns one of 39 semantic atom types to each token: concepts (C), predicates (Pd), modifiers (M), builders (B), conjunctions (J), triggers (T), etc.

2. **Beta stage** — a rule-based engine that combines classified atoms into ordered, recursive hyperedges: `((should/Mm fund/Pd.so) (the/Md hse/Cp) (+/B.am/. community/Cc (youth/M projects/Cc)))`

3. **Hypergraph database** — persistent storage, pattern search, inference, knowledge agents (taxonomy inference, coreference resolution).

**Tested and working on my machine** (Python 3.12, Windows). The alphabeta parser loads `en_core_web_lg` and produces structured hyperedges from real consultation text.

## The Integration Challenge

How to translate semantic hypergraph claims into my **existing heterogeneous networkx graph topology** in a way that the GNN can consume meaningfully — not as an abstract disconnected layer, but as concrete, queryable semantic structure.

### Design Options for Graph Representation

#### Option A: Claims as Separate Nodes
Each hypergraph claim becomes a node in the heterogeneous graph, connected to:
- Its subject actor (agent, project) via `makes_claim` edge
- Its object actor via `claim_references` edge
- Related claims via `entails`, `contradicts`, `supports` edges
- Perception nodes via `expresses_similar_narrative` edge

The GNN learns claim-to-actor, claim-to-claim, and claim-to-perception relationships.

**Pros:** Claims are first-class citizens. The GNN can reason about which actors share which claims, which claims are contested, which narratives diffuse across which paths.

**Cons:** Increases graph size significantly (139 claims → potentially thousands). Claims are sparsely connected initially. Requires the GNN to handle a new node type with different edge semantics.

#### Option B: Claims as Hyperedges in a Separate Semantic Layer
Keep claims in a hyperbase hypergraph database, separate from the networkx graph. Query it on-demand for the dashboard (e.g., "show me all claims made by actors in this k-core") and for feature engineering (e.g., "aggregate claim topics per actor as GNN features").

**Pros:** Clean separation of concerns. Hyperbase handles semantic inference natively. No GNN architecture changes needed.

**Cons:** The GNN cannot reason about claims directly. Semantic insight is limited to pre-computed features. Misses the chance for the GNN to discover novel claim-to-structure relationships.

#### Option C: Claims as Hyperedge Features on Existing Edges
Each existing edge (e.g., agent → project) gets a set of claim vectors as edge attributes — the claims that both endpoints share or that express the nature of their relationship.

**Pros:** Minimal graph restructuring. Directly augments what the GNN already sees.

**Cons:** Loses the claim-as-entity semantics. Cannot model claim-to-claim relationships. Hard to reason about orphan claims not tied to existing edges.

#### Option D: Hybrid — Hyperbase + Networkx Bridge
Use hyperbase as the semantic backend and build a bidirectional bridge:
- Claims live in hyperbase (pattern search, inference, knowledge agents)
- A translation layer converts claims to a simplified graph representation for the GNN
- The GNN's link predictions feed back into hyperbase as inferred knowledge
- The dashboard queries both layers through a unified API

**Pros:** Best of both worlds. Hyperbase handles the semantic complexity; networkx/GNN handles the structural analysis.

**Cons:** Two systems to maintain. Latency in the bridge layer. Requires defining the translation schema carefully.

### The Three-Level Narrative in Graph Terms

For each narrative level, I need to define how it maps to the graph:

| Level | Graph Element | Attributes | Connected To | Detection Method |
|---|---|---|---|---|
| Surface | Claim node | `hyperedge` (string), `verb`, `subject`, `object`, `modality`, `confidence`, `source_text` | Actor nodes (via `makes_claim`), perception nodes (via `expresses`) | hyperbase-parser-ab |
| Implicit | Claim node | Same structure + `inference_rule`, `stance` (-1 to 1), `emotion_tone` | Surface claims (via `implies`), challenge nodes (via `expresses_concern`) | Inference rules + LLM prompting + emotion lexicons |
| Metanarrative | Meta-claim node (aggregate) | `label` (e.g. "social_justice"), `value_dimension`, `belief_level` (ACF: deep_core / policy_core / secondary / material), `coherence_score` | Surface claims (via `expresses_value`), implicit claims (via `aligned_with`) | Clustering + value dictionary + graph propagation |

## The Question

I need a systematic approach to:

1. **Extract the three narrative levels** from consultation text using hyperbase/hyperbase-parser-ab (or alternative tools), with specific attention to:
   - Entity linking that works in the policy domain (resolving "the funding body" → "Rethink Ireland", "the health service" → "HSE")
   - Implicit claim detection via inference rules, stance analysis, and discourse markers
   - Metanarrative clustering — grouping claims by shared value dimensions and ACF belief levels
   - Quality filtering — how to distinguish a well-formed claim from a parse error

2. **Represent the claims in the existing graph** to maximize GNN utility, covering:
   - Which of the four design options (A-D above) is most appropriate for a policy network GNN
   - How to handle the sparsity issue (many actors make zero explicit claims)
   - How to keep the claim graph updated as new consultation data arrives
   - How to use the resulting semantic features in the GNN's link prediction task

3. **Operationalize the three narrative levels for decision-makers**, specifically:
   - Dashboard visualizations that show narrative structure (not just "hairball" graphs)
   - Metrics that translate narrative analysis to governance action (e.g., "coalition belief coherence score", "narrative diffusion barriers", "value alignment gap")
   - How to surface implicit claims and metanarratives to decision-makers without overclaiming certainty

## Output Format

Please provide:

1. **Recommended architecture** — specific tool choices, graph representation design, pipeline stages with effort estimates, with rationale for why this approach fits policy network analysis specifically

2. **Entity linking strategy** for the policy domain — methods that work with short consultation text where operational entity names rarely appear verbatim

3. **Implicit claim detection** — practical techniques (rule-based, LLM-based, hybrid) with tradeoffs for each, prioritized by ease of implementation

4. **Metanarrative extraction** — clustering and value classification approaches that map to ACF belief levels

5. **GNN integration design** — how the three narrative levels become node features, edge types, or additional graph layers, with specific attention to the R-GCN architecture already in place

6. **Existing code assessment** — specific recommendations for fixing or replacing `10_semantic_hypergraph_claims.py` to use hyperbase properly, and where in the pipeline orchestrator (`01_run_all_platform.py`) it should be inserted

7. **Key literature** — 10-15 papers spanning semantic hypergraphs, policy discourse analysis, narrative extraction, and belief system classification

8. **Implementation roadmap** — phased plan from quick-wins (surface level, weeks) to full integration (three levels + GNN, months)

## Current Constraints

- Python 3.12, Windows
- Existing R-GCN in PyTorch
- hyperbase v0.10.0 + hyperbase-parser-ab v0.3.0 installed and working
- spaCy en_core_web_lg installed
- sentence-transformers (all-MiniLM-L6-v2) installed
- ~375 nodes per platform, ~140 existing claims, ~1000 edges
- No labeled training data for belief classification
- Domain: Irish social innovation funding (mental health, youth services, community development, social enterprise)
- Dashboard audience: non-technical government decision-makers
