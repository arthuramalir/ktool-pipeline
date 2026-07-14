## The Project

We have 6+ years of stakeholder engagement data — workshops, consultations, listening exercises — collected through KTool. I've built a pipeline that turns all of this into a network map: who works with whom, what themes people are discussing, how information flows, where the money goes.

**What the pipeline does:**

- Extracts everything from KTool and builds a heterogeneous graph
- Runs structural analysis: most connected actors, weak points, cascade effects if key people leave, etc.
- Applies semantic analysis to the listening data — identifies narrative clusters, extracts claims (surface / implicit / metanarrative), maps how stories relate (agree, contradict, cause/effect) using the ALC Urdaibai methodology. Currently working with a small platform so the results are tractable and human-verifiable.
- Trains a Graph Neural Network that predicts missing connections (for the mapping layer) and simulates how new links would affect the system — a "what if" engine with recommendations.
- Synthetic financial augmentation: the real platforms are small and data-sparse (good for testing AI link quality via human verification, bad for prediction tasks), so I built a generator that fills in realistic financial categories to test budget allocation and perform financial analysis.
- Working on the relationship between financial flows and perception spaces, and how suggested links / projects affect both.
- All of this could feed into narrative change detection if we had snapshots of opinion over time for the same platform.

**The dashboard** puts it all in one place: network maps, story clusters, stress tests, budget analysis. I can deploy it to Streamlit Cloud and send you a link, whichever is easier.

## What I'm Thinking About Next

The deeper question I keep returning to: **given this network structure, is change even possible?** I have all these analytical layers — centrality, dominant narratives, blockages — but I want to move from describing the network to understanding what kinds of structural change are feasible.

Specifically:

- Where would a small intervention have the biggest effect?
- What's blocking change from propagating?
- Is the network locked into its current configuration?

I'm starting to research this from a political science angle (network theory of policy change, advocacy coalitions, path dependency). If you have any resources, reading recommendations, or contacts working on similar questions, I'd be very grateful for pointers.

Happy to walk through the dashboard whenever you have 20 minutes.
