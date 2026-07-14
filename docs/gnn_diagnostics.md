# GNN Link Prediction — Diagnostics & Limitations

> Use this document to understand why the GNN makes certain recommendations,
> where it fails, and how to interpret the results in a governance context.

---

## 1. Model Architecture

**R-GCN (Relational Graph Convolutional Network)**
- 2 layers, 64 hidden units
- Input features: degree, betweenness centrality, k-core, node type one-hot, methodological phase
- 4 relations: `collaborates_with`, `funded_by`, `implements`, `addresses`
- Decoder: dot product on z-vectors → link probability
- Training: negative sampling (5:1 negative:positive), binary cross-entropy, sampled every 500 edges

## 2. Known Limitations

### 2a. Zero Embedding Vectors

53 / 375 nodes (14%) have **zero-vector** sentence-transformer embeddings
(`np.linalg.norm(v) == 0.0`). These nodes appear in 2 of the 3 GNN recommendations:

| Recommendation | Nodes with zero embedding | Impact |
|---|---|---|
| Surf Club → Mental Health | Surf Club (`project_4570`) | Cannot use semantic domain filter |
| Corporate Workshops → Donegal CC | Corporate Workshops (`project_4574`) | Cannot use semantic domain filter |

**Why are embeddings zero?** The sentence-transformer model used in `08_nlp_semantic_alignment.py`
produces zero vectors when the input text contains only stop words, punctuation, or very short strings.
This typically happens for nodes whose `description` or `label` fields are empty or non-informative.

**Diagnosis steps:**
1. Check `node_semantic_embeddings.parquet` for zero-norm rows
2. Cross-reference with `nodes.csv` to see which nodes have missing/incomplete text
3. Consider re-running 08 with a fallback strategy (e.g., concatenating label + node_type + sector)

### 2b. Topological False Positives

The GNN is trained on **graph structure** (who connects to whom), not on operational domain semantics.
This creates false positives when:

1. **Structural homophily**: Two agents (e.g., Focus Ireland and Liquid Therapy) share similar
   structural roles (same node_type, similar degree/betweenness/k-core profiles) despite being in
   completely different operational domains (homelessness vs youth surfing).
2. **Generic vocabulary**: Node descriptions use generic social-sector vocabulary that overlaps
   across domains.
3. **Sparse supervision**: Only 8 training edges per platform for some relations → model has weak
   signal about what constitutes a "good" vs "bad" link.

### 2c. No Perception/Claim Awareness at Training Time

The GNN is trained on the **mapping layer** graph (project/agent/challenge nodes and their
relations). Perception and claim nodes are added later during narrative impact simulation
(script 14). This means:

- The GNN has **zero information** about perception statistics, sentiment, or claim content
- All "narrative impact" is computed post-hoc in `14_structural_impact_prediction.py`
- The GNN cannot distinguish between linking two agents in the same narrative space vs
  linking two agents with opposing perceptions

### 2d. Semantic Domain Filter (Mitigation)

A post-hoc **domain compatibility score** is applied in `14_structural_impact_prediction.py`:

| Condition | Score | Meaning |
|---|---|---|
| Cosine sim > 0.5 | 1.0 (compatible) | Same operational domain |
| Cosine sim 0.3–0.5 | 0.8 (neutral) | Possibly compatible, low confidence |
| Cosine sim < 0.3 | 0.5 (distant) | Likely different domains — penalise |
| Zero-embedding + same type | 0.8 (fallback) | Same node type, no semantic signal |
| Zero-embedding + diff type | 0.6 (fallback) | Different types, no semantic signal |

**Effect on current recommendations:**

| Recommendation | Cosine sim | Domain score | Governance interpretation |
|---|---|---|---|
| Surf Club → Mental Health | 0.000 (zero vec) | 0.8 (fallback) | Both projects — plausible |
| Focus Ireland → Liquid Therapy | 0.363 | 0.8 (neutral) | Low semantic overlap, structurally driven |
| Corporate Workshops → Donegal CC | 0.000 (zero vec) | 0.8 (fallback) | Both projects — plausible |

The Focus Ireland ↔ Liquid Therapy case remains a concern: both are agents with similar
structural profiles but very different domains. The domain filter catches this as "neutral"
rather than "compatible" but doesn't fully reject it.

## 3. Recommendation Breakdown

### 3a. Edge Addition Recommendations (from GNN)

| # | Source | Target | GNN Score | Category | Bridge Type |
|---|---|---|---|---|---|
| 1 | Surf Club | Mental Health | high | alignment | learning_bridge |
| 2 | Focus Ireland | Liquid Therapy | medium | alignment | coalition_reinforcement |
| 3 | Corporate Workshops | Donegal Council | high | alignment | structural_only |

### 3b. Node Addition Proposals (from 14)

Generated when an endpoint has ≤ 2 neighbors or zero narrative/perception reachability.
Sources: GNN endpoints + narrative gap detection.

### 3c. Node Invention (from 14)

Only for `structural_only` bridge type edges where both endpoints have non-zero embeddings.

**Current inventions:** 0 (all structural_only endpoints have zero embeddings on at least one side)

## 4. When to Trust / Not Trust Recommendations

| Scenario | Trust | Why |
|---|---|---|
| Same value dimension, different components | ✅ | Narrative bridge creates measurable new connections |
| Different value dimensions with cosine sim > 0.5 | ✅ | Cross-belief dialogue initiated |
| Same node_type, cosine sim 0.3–0.5 | ⚠️ | Structurally plausible, semantically weak |
| Different node_type, cosine sim < 0.3 | ❌ | Likely topological false positive |
| Zero embedding on either node | ⚠️ | No semantic signal to verify |

## 5. Data Quality Checks

- [ ] How many zero-embedding nodes exist? (53 in 173)
- [ ] Do any recommendation endpoints have zero embeddings? (2/6 in 173)
- [ ] What is the random-pair baseline cosine similarity for embeddings?
- [ ] Do perception nodes carry useful semantic signal?
- [ ] Are claim nodes being assigned embeddings? (currently 0/122)

## 6. Future Improvements

1. **Train perception-aware GNN**: Include perception node features (or a perception proximity
   metric) in the GNN input features
2. **Incorporate claim embeddings**: Run claim text through sentence-transformers and add as
   node features
3. **Sector/domain as explicit feature**: If sector data becomes available (>5% coverage), add
   as a one-hot feature to the R-GCN model
4. **Domain pre-filter**: Before GNN training, filter candidate edges by domain compatibility
   (require cosine sim > 0.3)
5. **Re-train with more edges**: The current 8 training edges per relation is very sparse;
   adding more ground-truth edges would improve robustness
