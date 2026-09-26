export const navLinks = [
  { to: "/platform", label: "Platform" },
  { to: "/research", label: "Research" },
  { to: "/benchmarks", label: "Benchmarks" },
  { to: "/simulation", label: "Simulation" },
  { to: "/studio", label: "Studio" },
  { to: "/docs", label: "Docs" },
  { to: "/roadmap", label: "Roadmap" },
] as const;

export const platformTracks = [
  {
    key: "neural",
    label: "Neural routing",
    eyebrow: "LEARN",
    title: "Policies that learn the shape of a problem.",
    body: "Attention models, hierarchical reinforcement learning, and meta-learning explore how routing policies can generalize beyond one fixed distribution.",
    color: "eco" as const,
  },
  {
    key: "classical",
    label: "Classical optimization",
    eyebrow: "PROVE",
    title: "Exactness and heuristics in the same laboratory.",
    body: "Gurobi, ALNS, HGS, OR-Tools, and branch-and-price-and-cut provide rigorous baselines and practical search for profit-aware and capacitated routes.",
    color: "amber" as const,
  },
  {
    key: "simulation",
    label: "Scenario simulation",
    eyebrow: "TEST",
    title: "Stress a policy across time, not just one instance.",
    body: "Multi-day waste collection scenarios expose overflow, capacity, uncertainty, and operational trade-offs before a policy reaches deployment.",
    color: "route" as const,
  },
] as const;

export const solvers = [
  { name: "Gurobi", kind: "Exact" },
  { name: "ALNS", kind: "Metaheuristic" },
  { name: "HGS", kind: "Genetic" },
  { name: "OR-Tools", kind: "Constraint" },
  { name: "PyVRP", kind: "Heuristic" },
] as const;

export const metrics = [
  {
    label: "Problem families",
    value: "VRPP · CWC VRP · SCWCVRP",
    note: "profit, capacity, stochastic fill",
  },
  {
    label: "Approaches",
    value: "Neural + Classical",
    note: "one shared experimental surface",
  },
  {
    label: "Simulation",
    value: "Multi-day scenarios",
    note: "31–365 day horizons",
  },
  {
    label: "Primary surface",
    value: "Studio / Desktop",
    note: "Tauri 2 research workbench",
  },
] as const;

export const researchNotes = [
  {
    title: "Vehicle Routing Problem with Profits",
    body: "Select profitable stops under route-length and capacity constraints — when visiting everything is not the right answer.",
  },
  {
    title: "Capacitated Waste Collection VRP",
    body: "Serve bins under vehicle capacity, depot returns, and operational constraints that municipalities actually face.",
  },
  {
    title: "Stochastic multi-day operations",
    body: "Fill rates evolve; overflow risk accumulates. Policies must plan across days, not only single static instances.",
  },
  {
    title: "Neural combinatorial optimization",
    body: "Constructive attention models, graph encoders, hierarchical RL, and meta-learning against classical OR baselines.",
  },
] as const;

export const docs = [
  {
    title: "Architecture",
    path: "docs/ARCHITECTURE.md",
    kind: "SYSTEM",
    blurb: "How logic, Studio, and pipelines fit together.",
  },
  {
    title: "Configuration guide",
    path: "docs/CONFIGURATION_GUIDE.md",
    kind: "START",
    blurb: "Hydra composition, CLI overrides, and multi-run sweeps.",
  },
  {
    title: "Benchmarks",
    path: "docs/BENCHMARKS.md",
    kind: "EVIDENCE",
    blurb: "Methodology, comparison surfaces, and caveats.",
  },
  {
    title: "Studio guide",
    path: "app/docs/README.md",
    kind: "PRODUCT",
    blurb: "Desktop app pages, workflows, and development notes.",
  },
  {
    title: "Feature roadmap",
    path: "docs/FEATURE_ROADMAP.md",
    kind: "DIRECTION",
    blurb: "Active research and product directions.",
  },
  {
    title: "Glossary",
    path: "docs/GLOSSARY.md",
    kind: "REFERENCE",
    blurb: "Shared language for problems, policies, and metrics.",
  },
] as const;

export const roadmapItems = [
  {
    phase: "NOW",
    title: "Make runs easier to compare",
    body: "Unify benchmark and simulation evidence around shared run metadata and clearer Studio comparison views.",
  },
  {
    phase: "NEXT",
    title: "Strengthen policy generalization",
    body: "Expand cross-distribution and multi-period evaluation so policies are tested beyond a single training shape.",
  },
  {
    phase: "RESEARCH",
    title: "Learn from operations",
    body: "Explore hybrid neural and classical decision loops under uncertainty, with overflow and capacity in the loop.",
  },
] as const;

export const githubRepo = "https://github.com/ACFHarbinger/WSmart-Route";
