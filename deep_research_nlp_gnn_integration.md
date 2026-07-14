# Deep Research: Integrating NLP Meaning Extraction into a GNN for Policy Network Analysis

## Project Context

I am building a decision-support pipeline for a government innovation agency (ALC in Ireland). The pipeline:

1. **Extracts** stakeholder engagement data from the KTool platform (~6 years of consultations, workshops, listening exercises across mental health, youth services, community development, and social innovation funding).

2. **Builds** a heterogeneous network graph where nodes = agents (organisations, community groups), initiatives (projects, pilots, prototypes), perceptions (themes/opinions from workshops), challenges, information items (quotes/testimonies), channels (media, reports), values, patterns. Edges = declared links (who leads what, who partners with whom), listening evidence chains (channel → information → value), interpretive edges (NLP-inferred similarity, contradiction, causality, sequence between quotes), and perception-to-initiative links.

3. **Runs** an analytical pipeline producing centrality metrics, k-core decomposition, articulation points, robustness/stress testing, GNN-based link prediction (R-GCN), intervention simulation, and perception diagnostics.

4. **Visualises** everything in a Streamlit dashboard for non-technical government decision-makers.

## The Gap

My current NLP pipeline exists in two disconnected pieces:

**Piece A — Shallow similarity (working):**
- `08_nlp_semantic_alignment.py`: Encodes node descriptions with SentenceTransformer (all-MiniLM-L6-v2), computes cosine similarity, generates `nlp_semantic_edges.csv` between listening-layer nodes (information, perception, challenge, value) and operational nodes (agent, project, pilot, prototype).
- I recently modified it to also save per-node 384-dim embeddings to `node_semantic_embeddings.parquet`, which the GNN preparation script loads as node features.
- This is now fully integrated: semantic coverage = 100%, the GNN uses these embeddings in its feature matrix.
- **Problem:** This is surface-level semantic similarity. It captures topical relatedness but not structured meaning.

**Piece B — Meaning extraction (broken and disconnected):**
- `10_semantic_hypergraph_claims.py`: Uses spaCy dependency parsing to extract subject-verb-object triples from narrative text (information nodes, perceptions, challenges). Formats them as graphbrain-style hypergraph expressions like `(funds/P agency/C project/C)`.
- Outputs `socio_semantic_hypergraph_claims.csv` with ~139 claims per platform.
- **Problems:** (1) Entity linking is naive string-matching — most claims have `anchored_target_label = "Implicit Concept"`. (2) Claims are never used by the GNN — they just sit in a CSV. (3) The SVO extraction is basic and noisy.

**Additional NLP components (all producing edges, not node features):**
- `14_alc_advanced_semantic_edges.py`: Uses SentenceTransformer + rule-based heuristics to detect similarity, contradiction, causality, and sequence between quote pairs. Outputs `quote_semantic_edges.csv`.
- `15-17`: Quote extraction → semantic edge detection → profile clustering (for the "Story Clusters" tab).
- `09_perception_diagnostics.py`: Computes perception coherence, purity, source entropy using quote-to-information mapping.

**The critical gap:** There is no pipeline stage that:
- Extracts *structured claims* from text with reliable entity resolution to graph nodes
- Classifies claims by belief hierarchy level (deep core, policy core, secondary belief, material interest — from Sabatier's Advocacy Coalition Framework)
- Classifies claims by stance (supportive, oppositional, neutral) toward specific policies, actors, or initiatives
- Connects these claims back to the heterogeneous graph as either node features, edge types, or a separate claim-graph layer that the GNN can reason over

## The GNN Architecture

The current GNN is an R-GCN (Relational Graph Convolutional Network) for link prediction:
- **Node features:** Numeric (centrality, k-core, component size) + categorical (node type one-hot, methodological phase) + text hash (TF-IDF-like label+description hashing) + **semantic embeddings** (384-dim SentenceTransformer, just added)
- **Edge types:** 8-20 relation types depending on platform (agent→project, project→perception, information→channel, etc.)
- **Training:** Link prediction with negative sampling, binary cross-entropy loss
- **Outputs:** Link probability scores for node pairs, perception-space effect analysis, intervention simulation

## The Research Question

I want to integrate **meaning-level NLP** into this GNN pipeline so the model can reason about:

1. **What actors believe** — not just who is connected to whom, but what claims they make, what stances they hold, which beliefs they share with other actors
2. **How narratives diffuse** — which claims spread across which network paths, which actors serve as narrative bridges
3. **Coalition structure** — which actors share the same belief hierarchy (deep core + policy core alignment), not just topological proximity
4. **Contested claims** — where different actors make contradictory claims about the same topic, indicating coalition boundaries

## Specific Directions I Need Researched

### 1. Claim Extraction and Entity Linking for Policy Networks

What is the state of the art for extracting structured claims from semi-structured policy/consultation text and linking subjects/objects to known graph entities?

- **Current approach:** spaCy dependency parsing with naive string matching. The entity linking fails because operational node labels ("Rethink Ireland", "HSE") rarely appear verbatim in narrative text ("the health service", "the funding body").
- **What I need:** Methods for — coreference resolution in short, domain-specific text; fuzzy entity linking that maps "the health service" → "HSE (Health Service Executive)"; claim normalization (canonical forms of similar claims); handling of negated claims, conditional claims, and hypothetical claims common in consultation discourse.
- **Specific question:** Is there a proven approach for claim extraction + entity resolution in the context of *governance/policy networks* specifically, not just general IE benchmarks?

### 2. Representing Claims in the Graph

Once extracted, how should claims be represented in the heterogeneous graph for a GNN to consume?

**Options I can see:**
- **a) Claims as node features:** Aggregate claims per actor into a claim-vector (e.g., frequency of verb types, topic distributions). Loses relational structure between claims.
- **b) Claims as separate nodes:** Each claim is a node in the graph, connected to its subject-actor and object-actor via typed edges. The GNN learns claim-to-claim, claim-to-actor, and claim-to-perception relationships. Scales with number of claims.
- **c) Claims as hyperedges:** A single hyperedge connects (subject, verb, object) as a n-ary relation. Requires a GNN architecture that supports hypergraphs.
- **d) Claims as edge types:** A pair of actors can share a "makes-claim-about" typed edge with the claim text as an edge attribute. Limited to binary relations.

**Key tradeoffs:** Graph complexity vs expressiveness, scalability (139 claims now, could be 5000+), GNN architecture compatibility (current R-GCN expects binary edges with types).

### 3. Belief Hierarchy Classification

The Advocacy Coalition Framework (Sabatier & Jenkins-Smith, 1993) distinguishes three levels of belief:
- **Deep core:** Fundamental values (equity, liberty, efficiency) — highly stable, cross-domain
- **Policy core:** Problem diagnosis, policy positions, institutional preferences — semi-stable, domain-specific
- **Secondary:** Instrumental choices, budget allocations, implementation details — plastic, adaptive

And the deep research report added:
- **Material interest:** Short-term organizational survival, funding preservation

**Can claims extracted from consultation text be reliably classified into this hierarchy?**

- What methods exist for classifying policy beliefs from text? (supervised classification, LLM prompting, rule-based keyword + syntax patterns?)
- What is the minimum labeled data needed for a reasonable classifier in a narrow domain (social innovation funding)?
- How reliable are LLM-based classifiers (few-shot GPT-4, Claude, etc.) for this specific task vs fine-tuned smaller models?
- Are there existing corpora or taxonomies for belief classification in social innovation / community development domains?

### 4. Stance Detection in Policy Discourse

Beyond "what claim is made," I need **whose claim aligns with whose** — stance toward policies, other actors, and proposals.

- What is the best approach for multi-target stance detection in policy discourse (actor A stance toward policy B, toward actor C, toward claim D)?
- How to handle implicit stance (e.g., one actor prioritizing funding for mental health while another prioritizes youth services implies a stance gradient, not a binary for/against)?
- Can stance be encoded as a continuous dimension (alignment score between -1 and 1) rather than a discrete label, and if so, how does the GNN use it?

### 5. GNN Architecture Modifications for Semantic Awareness

Given that claims, beliefs, and stances can be represented as:
- Additional node types (claim nodes)
- Additional edge types (endorses, contests, implies)
- Additional node features (claim-vectors, stance-vectors)

**What GNN architecture changes does this imply?**

- Can a standard R-GCN handle claim nodes as first-class citizens, or does the relation count explode?
- Would a heterogeneous graph transformer (e.g., HAN, HGT) be more appropriate for semantic-attention over different claim types?
- How to handle the sparsity issue — many agents make zero explicit claims in the data, while a few agents mention many claims?
- Are there existing examples of GNNs over policy/consultation graphs with semantic node types?

### 6. Pipeline Integration

Given practical constraints (Python, PyTorch, existing R-GCN implementation, Streamlit dashboard):
- What is the most practical *next step* that delivers meaningful improvement within 1-2 months?
- What would an ideal "phase 2" architecture look like (6-12 months)?
- What should be avoided (approaches that look promising but fail on real policy text)?

## Output Format

Please provide:
1. **Synthesised approach** — recommended architecture for integrating NLP meaning extraction into this GNN pipeline, with rationale
2. **Specific methods** for each component (claim extraction, entity linking, belief classification, stance detection, graph representation) with citations to relevant papers or implementations
3. **Practical integration plan** — step-by-step, starting from the current codebase, with effort estimates
4. **Alternative approaches** — what we should NOT do, and why
5. **Key papers** — 10-15 most relevant papers spanning NLP, network science, and political science methods
6. **Code/implementation references** — existing open-source tools, libraries, or repositories that approximate one or more components

## Current Constraints

- Python 3.12, PyTorch (for GNN), sentence-transformers, spaCy, networkx, pandas, numpy
- GPU available but limited (RTX-level, not A100)
- No labeled training data currently — would need to create or bootstrap
- Domain: social innovation funding in Ireland (mental health, youth services, community development, social enterprise)
- Text sources: workshop transcripts (1-5 paragraphs), survey responses (1-3 sentences), project descriptions (1-2 paragraphs), quote extractions (1-2 sentences)
- The GNN is trained for link prediction, not node classification — but this could change
- Dashboard audience: non-technical government decision-makers who need actionable insights, not NLP metrics
