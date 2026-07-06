"""Political science framework data from deep research output.

Bridges graph-theoretic metrics to governance/political constructs
for the Structural Change dashboard tab.
"""

METRIC_MAP = [
    {
        "metric": "Maximal k-Core",
        "concept": "Policy Monopoly / Dominant Coalition",
        "interpretation": "Entrenched inner circle of highly cohesive, well-connected regime actors who control resources and enforce the status quo.",
        "relevance": "High coreness = high resistance to change. Interventions targeting the core are often futile unless coupled with external shocks.",
        "source": "Sabatier & Jenkins-Smith (1993); Heemskerk, Daolio & Tomassino (2013)",
    },
    {
        "metric": "Betweenness Centrality",
        "concept": "Policy Brokerage / Boundary Spanning",
        "interpretation": "Actors who mediate information and resource flows across structural holes. Vital for building consensus and compromises.",
        "relevance": "High-betweenness nodes are high-leverage intervention points. Strengthening them accelerates narrative diffusion.",
        "source": "Gould & Fernandez (1989)",
    },
    {
        "metric": "Articulation Points",
        "concept": "Institutional Bottlenecks / Structural Cleavages",
        "interpretation": "Fragile actors whose removal de-links different sectors, creating isolated blocks. Single points of system failure.",
        "relevance": "Vulnerability points that block change propagation. Engineer redundancy by building alternative bridges.",
        "source": "Baumgartner & Jones (1993)",
    },
    {
        "metric": "Targeted Node Decay",
        "concept": "Governance Fragility / Coalition Vulnerability",
        "interpretation": "How vulnerable communication pathways are to strategic removal of central gatekeepers.",
        "relevance": "Brittle systems = centralized hierarchy dominated by few actors. Indicates high path dependency.",
        "source": "Cairney (2019)",
    },
    {
        "metric": "PageRank Centrality",
        "concept": "Reputational Prestige / Cognitive Authority",
        "interpretation": "Actors considered highly credible because they are connected to other influential actors.",
        "relevance": "High PageRank + low formal power = untapped cognitive authorities for narrative reframing.",
        "source": "Provan & Kenis (2007)",
    },
    {
        "metric": "Semantic Betweenness",
        "concept": "Discursive Bridges / Narrative Framings",
        "interpretation": "Concepts that connect otherwise disconnected discourse communities or competing coalitions.",
        "relevance": "High semantic betweenness = ideas with capacity to build coalitions across deep ideological splits.",
        "source": "Fischer & Maggetti (2020)",
    },
    {
        "metric": "Perception Coherence",
        "concept": "Ideological Consolidation",
        "interpretation": "Tightness and consistency of a coalition's shared policy beliefs. High coherence = unified, resilient advocacy frame.",
        "relevance": "Hyper-coherent coalitions resist change. Low coherence = coalition vulnerable to internal division.",
        "source": "Sabatier & Jenkins-Smith (1993)",
    },
    {
        "metric": "Stranded Asset Detection",
        "concept": "Institutional Exclusion / Niche Isolation",
        "interpretation": "Projects or organizations with high alignment to agency goals but no relational ties to the central core.",
        "relevance": "Wasted capacity. Requires targeted link addition to integrate these nodes into the broader ecosystem.",
        "source": "Provan & Kenis (2007)",
    },
]

GOVERNANCE_MATRIX = [
    {
        "finding": "Innermost k-Core contains only established organizations",
        "interpretation": "Elite Capture / Institutional Monopolization",
        "risk": "Path dependency & stagnation: new community-led initiatives are starved of capital and excluded from planning.",
        "action": "Venue Diversification: direct funding to independent regional intermediaries to bypass the core.",
    },
    {
        "finding": "High Betweenness concentrated in a single intermediary",
        "interpretation": "Centralized Brokerage / Communication Bottleneck",
        "risk": "Systemic brittleness: exit or failure of this node fragments the ecosystem.",
        "action": "Redundancy Engineering: fund liaison brokerage pathways across adjacent sectors to build resilient connections.",
    },
    {
        "finding": "Low network density + high modularity",
        "interpretation": "Ecosystem Fragmentation / Tribalized Coalitions",
        "risk": "Coordination failure: high duplication, low resource sharing, low collective action capacity.",
        "action": "Core Consolidation: form a Network Administrative Organization with cross-sectoral mandate.",
    },
    {
        "finding": "High semantic distance between sectors",
        "interpretation": "Discursive Cleavages / Ideological Polarization",
        "risk": "Political stalemate: competing coalitions block collaborative policy implementation.",
        "action": "Boundary Spanning: fund platforms targeting discursive bridges with high semantic betweenness.",
    },
    {
        "finding": "High GNN link prediction scores between isolated niches and core",
        "interpretation": "Latent Collaborative Capacity",
        "risk": "Stranded assets: high-value projects ready for integration but relationally invisible.",
        "action": "Targeted Link Addition: design joint initiatives that mandate partnerships between core and niche actors.",
    },
]

FRAMEWORKS = [
    {
        "name": "Advocacy Coalition Framework (ACF)",
        "author": "Sabatier & Jenkins-Smith (1993)",
        "core_idea": "Policy subsystems contain competing coalitions bound by shared belief systems and coordinated activity. Change happens when external shocks are exploited by minority coalitions.",
        "belief_hierarchy": {
            "Deep Core": "Fundamental values — highly resistant to change, span across subsystems.",
            "Policy Core": "Problem severity, causes, solutions — primary filter for ally/opponent selection.",
            "Secondary": "Operational details, budgets, implementation — most plastic, adapt through learning.",
        },
        "dashboard_use": "Use to interpret perception coherence: high coherence on policy core = stable coalition, hard to shift. Alignment only on secondary beliefs = fragile partnership.",
        "key_insight": "Material beliefs (funding survival) interact with purposive beliefs (societal goals) — both must be mapped to understand coalition stability.",
    },
    {
        "name": "Punctuated Equilibrium Theory (PET)",
        "author": "Baumgartner & Jones (1993)",
        "core_idea": "Long stability (equilibrium) interrupted by brief bursts of non-linear change. Policy monopolies maintain control via a supporting narrative + insulated venue.",
        "pillars": {
            "Policy Image": "Dominant framing that makes the issue seem technical or successfully managed, discouraging public attention.",
            "Policy Venue": "Closed institutional arrangement restricting decision-making to a small elite circle.",
        },
        "dashboard_use": "A dense k-core + low semantic betweenness = a policy monopoly. Change requires reframing the issue + venue shopping to more favorable forums.",
        "key_insight": "Positive feedback loops mean the rich get richer — resources flow to the core, entrenching the status quo. Punctuation requires excluded actors to reframe the dominant image.",
    },
    {
        "name": "New Public Governance (NPG)",
        "author": "Provan & Kenis (2007)",
        "core_idea": "Public services are co-produced through multi-sector networks, not hierarchical command. Relies on trust, collaborative contracts, and relational capital.",
        "governance_modes": {
            "Self-governed": "Decentralized, shared decision-making — works for small, high-trust networks.",
            "Lead-organization": "One hub coordinates — efficient but creates bottlenecks (articulation points).",
            "NAO": "Network Administrative Organization — dedicated entity manages the network, best for large fragmented systems.",
        },
        "dashboard_use": "Use to design intervention typology: fragmented ecosystem → NAO recommendation. Centralized brittle system → redundancy engineering.",
        "key_insight": "Path dependency locks in early institutional choices. Transition costs rise exponentially — systems converge toward stable 'basins of attraction'.",
    },
]

INTERVENTION_MATURITY = [
    {
        "subsystem_type": "Mature, Rigid",
        "profile": "Entrenched high-k core, policy monopoly, high path dependency",
        "goal": "Disrupt status quo, enable niche innovations",
        "strategy": "Outsider Brokerage + Venue Shopping",
        "alc_role": "Independent Liaison/Itinerant broker — fund boundary organizations, mediate conflicts, align material interests, fund peripheral advocacy groups to reframe the policy image.",
    },
    {
        "subsystem_type": "Nascent, Plastic",
        "profile": "Low density, fragmented coalitions, fragile articulation points, no established monopoly",
        "goal": "Build stability and coordination capacity",
        "strategy": "Core Consolidation",
        "alc_role": "Fund a Network Administrative Organization as a central hub. Grant integrated budget and service-planning authority to build trust and stabilize communication.",
    },
]


def framework_narrative(score_name: str, value: float) -> str:
    """Return a political science interpretation for a change readiness score."""
    narratives = {
        "leverage_score": (
            "**Political interpretation:** High betweenness actors are brokerage points — the 'boundary spanners' who "
            "mediate across structural holes (Gould & Fernandez, 1989). Strengthening them accelerates narrative "
            "diffusion and coalition building. Low scores mean change needs new bridges, not existing ones."
        ),
        "plasticity_score": (
            "**Political interpretation:** A network rich in peripheral nodes (degree 1-2) and GNN-predicted latent "
            "links has spare capacity to accept new connections without triggering systemic resistance. Low scores "
            "indicate saturation — the network is rigid, and link addition has diminishing returns (Provan & Kenis, 2007)."
        ),
        "blockage_score": (
            "**Political interpretation:** Articulation points represent institutional bottlenecks (Baumgartner & Jones, 1993). "
            "Fragile connectors + small components = blocked propagation paths for narratives and resources. "
            "Blocked perceptions (no path to information nodes) indicate discursive cleavages — whole coalitions "
            "talking past each other."
        ),
        "lockin_score": (
            "**Political interpretation:** A dense k-core maps to a policy monopoly (Sabatier & Jenkins-Smith, 1993). "
            "High lock-in = positive feedback loops funneling resources to the core. The robustness gap (targeted vs "
            "random decay) reveals whether the system depends on a few irreplaceable hubs — classic path dependency. "
            "Change must start at the periphery (Cairney, 2019)."
        ),
        "overall_readiness": (
            "**Political interpretation:** Composite of all four dimensions. High readiness (>0.6) = the ecosystem is "
            "structurally open to reconfiguration. Low readiness (<0.3) = the system is locked in, requiring either "
            "an external shock or a long-term venue-shopping strategy to punctuate the equilibrium."
        ),
    }
    return narratives.get(score_name, "")


def governance_action(metric: str, value: float, threshold: float = 0.5) -> str:
    """Return an actionable governance recommendation based on metric + value."""
    actions = {
        "leverage_score": (
            f"**If value > {threshold}:** Identify the top betweenness nodes and strengthen their brokerage capacity — "
            f"fund their coordination role, protect them from turnover. "
            f"**If value < {threshold}:** Don't rely on existing bridges. Create new ones — fund boundary-spanning "
            f"platforms that connect disconnected components."
        ),
        "plasticity_score": (
            f"**If value > {threshold}:** The network can absorb new links. Use GNN predictions to add strategic "
            f"connections between compatible but unlinked actors. "
            f"**If value < {threshold}:** The network is saturated. Prune redundant or low-value connections before "
            f"adding new ones. Focus on rewiring, not just adding."
        ),
        "blockage_score": (
            f"**If value > {threshold}:** Map the fragile connectors and blocked perceptions. Build redundant pathways "
            f"around articulation points. Fund bridging initiatives that connect isolated components. "
            f"**If value < {threshold}:** Blockages are minor. Monitor articulation points for turnover risk."
        ),
        "lockin_score": (
            f"**If value > {threshold}:** The core is entrenched. Don't challenge it directly. Fund venue shopping — "
            f"peripheral advocacy groups that can reframe the issue and shift debates to alternative forums. "
            f"**If value < {threshold}:** The topology is open. Core consolidation may be appropriate to build stability."
        ),
    }
    return actions.get(metric, "")
