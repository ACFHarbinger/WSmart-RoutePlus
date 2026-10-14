# WSmart-Route Glossary

This document provides a reference for common abbreviations, acronyms, and domain-specific terms used throughout the WSmart-Route codebase.

## Technical Abbreviations

| Abbreviation | Expanded Form         | Description                                                                                     |
| :----------- | :-------------------- | :---------------------------------------------------------------------------------------------- |
| **ptr**      | Pointer               | Usually refers to Pointer Networks or pointer-based selection mechanisms.                       |
| **moe**      | Mixture of Experts    | A machine learning technique where multiple "expert" networks are used with a gating mechanism. |
| **hgnn**     | Heterogeneous GNN     | Heterogeneous Graph Neural Network, used for graphs with multiple node/edge types.              |
| **mpnn**     | Message Passing NN    | Message Passing Neural Network, a framework for Graph Neural Networks.                          |
| **mdam**     | Multi-Decoder AM      | Multi-Decoder Attention Model, a variant of the Attention Model.                                |
| **nar**      | Non-Autoregressive    | Refers to models or decoding strategies that do not generate output step-by-step.               |
| **am**       | Attention Model       | The standard Transformer-based constructive routing model.                                      |
| **tam**      | Temporal AM           | Time-dependent Attention Model for multi-day scenarios.                                         |
| **ddam**     | Deep Decoder AM       | A variant of AM with a deeper decoder stack.                                                    |
| **hrl**      | Hierarchical RL       | Reinforcement Learning with a high-level manager and low-level workers.                         |
| **hgs**      | Hybrid Genetic Search | A state-of-the-art metaheuristic for VRP.                                                       |
| **alns**     | ALNS                  | Adaptive Large Neighborhood Search, a popular metaheuristic.                                    |
| **swc_tcf**  | SWC-TCF               | Smart Waste Collection - Two-Commodity Flow, standard policy for VRPP.                          |

## Problem Domain Terms

| Term        | Expanded Form                | Description                                                                   |
| :---------- | :--------------------------- | :---------------------------------------------------------------------------- |
| **VRP**     | Vehicle Routing Problem      | The core optimization task of finding optimal routes for vehicles.            |
| **VRPP**    | VRP with Profits             | A variant where nodes have associated profits and not all need to be visited. |
| **WCVRP**   | Waste Collection VRP         | VRP specifically tailored for bin collection with various constraints.        |
| **CWC VRP** | Capacitated WCVRP            | Waste collection with strict vehicle capacity constraints.                    |
| **SCWCVRP** | Stochastic Capacitated WCVRP | Waste collection where bin fill levels are uncertain.                         |

## Code Semantics

| Term           | Context      | Guidance                                                  |
| :------------- | :----------- | :-------------------------------------------------------- |
| **num_loc**    | Config/Env   | Number of locations (excluding depot).                    |
| **graph_size** | Config/Model | Total number of nodes (usually `num_loc + 1`).            |
| **n_nodes**    | Math/Tensors | Alias for `graph_size` in tensor operations.              |
| **B**          | Tensors      | Often used for Batch size dimension.                      |
| **N**          | Tensors      | Often used for Number of nodes dimension.                 |
| **cur_loc**    | Simulation   | The current location index of a vehicle.                  |
| **packing**    | Simulation   | Locations that _must_ be visited today (e.g., full bins). |
