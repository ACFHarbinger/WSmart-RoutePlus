"""WSmart-Route Core Logic Package.

This package contains the core business logic, simulation environment, solvers,
neural models, configuration layers, and testing suites.

Directory Structure and Module Descriptions:
--------------------------------------------

1. benchmark
   - Benchmark suites, evaluation frameworks, and performance baseline scripts
     for comparing various waste collection routing algorithms.

2. configs
   - Configuration management module for Hydra YAML definitions.
   - Subdirectories:
     - envs: Environment specific variables (e.g. CVRP, PTP).
     - models: Architecture defaults for ML policies (e.g. Attention Models).
     - policies: Solver parameters and heuristics settings (e.g. ALNS, HGS, PSOMA).
     - tasks: Config overrides defining specific workflows (e.g. train, test_sim).
     - tracking: Target definitions for telemetry (e.g. WandB, MLflow).
     - ui: Graphical UI configuration templates.

3. controllers
   - Workflow dispatchers and application orchestration layer.
   - Subdirectories:
     - manager: Automated sequential batch run coordinator, including
       pre/post-processing tasks, matrix generation, and VCS hooks.

4. docs
   - Architectural guides, design documents, and documentation pipelines.
   - Subdirectories:
     - source: Sphinx configuration files and source ReST files.

5. examples
   - Jupyter notebooks, API walkthroughs, and tutorials demonstrating
     usage patterns of environments, pipelines, and solver policies.

6. migrations
   - Relational database schema adjustments, table creations, and SQL script
     migrations for persistent run tracking.

7. scripts
   - Standalone diagnostic scripts, data generator executables, distance
     matrix calculation utilities, and post-simulation analysis runners.

8. src
   - Core Python package source code.
   - Subdirectories:
     - cli: Command-line interface parsers, registries, and validations.
     - configs: Hydra schema definition classes mapping to YAML structures.
     - constants: Global settings, default file paths, and mathematical mappings.
     - data: Pipelines for raw network data engineering, Google Maps integration,
       and bin-level fill distribution modeling.
     - enums: Standard enum declarations for system tasks and problem status.
     - envs: Gym-compliant environments simulating multi-day waste fill cycles.
     - interfaces: Protocol and abstract contract definitions enforcing API limits.
     - models: Neural net configurations, encoders, and decoding mechanisms.
     - pipeline: Training coordinators, evaluation modules, and Lightning callbacks.
     - policies: Heuristics, metaheuristics, and machine learning decision models.
     - tracking: Telemetry managers, data lineage callbacks, and logger bindings.
     - utils: General utility library (file IO, security operations, typing helper).

9. test
   - Package verification and testing suites.
   - Subdirectories:
     - e2e: Integration end-to-end full simulation trials tests.
     - fixtures: Mock data structures and parameters for tests.
     - integration: Component interaction tests.
     - properties: Hypothesis-based property validations.
     - unit: Component-level unit test definitions.
"""

__all__: list[str] = []
