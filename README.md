# Capacitated Vehicle Routing Problem Research

> Venkatesh Duraiarasan, DA24C021

## 1. Motivation and Problem Statement
Capacitated Vehicle Routing Problem (CVRP) is a generalization of the Traveling Salesman Problem
(TSP) where multiple vehicles with limited capacity must serve customer demands. The capacity
constraints significantly increase the complexity of the problem, making exact approaches slow for
large instances. This project evaluates several classical and learning-based techniques for solving
CVRP and compares their performance.

## 2. Project Objectives
The project benchmarks a mix of exact, metaheuristic and reinforcement learning (RL) methods for
CVRP. The main objectives are:
- Evaluate the effectiveness of RL in producing near-optimal solutions compared to traditional
  techniques.
- Analyse computational efficiency in terms of solution quality, optimality gap and run time.
- Highlight strengths and limitations of each approach on instances of varying size and difficulty.

## 3. Repository Structure
```
├── src
│   ├── sol_1_mip         # Pyomo model solved with SCIP
│   ├── sol_1_mip_bcp     # Branch-and-cut using VRPSolverEasy
│   ├── sol_3_mh_hgs      # Hybrid genetic search (hygese)
│   ├── sol_4_or_tools    # OR-Tools implementation
│   ├── sol-3-rl          # Reinforcement learning prototype (PyTorch)
│   ├── data-collection   # Scripts for downloading benchmark data
│   └── common            # Shared helpers
├── results               # Pickled outputs for each solver
├── Report.pdf            # Detailed report
└── README.md             # This file
```

## 4. Data
Benchmark instances from **VRPLIB** can be downloaded using the helper script:
```bash
python src/data-collection/download-standard-dataset.py
```
The script collects several standard CVRP sets (A, B, E, F, M) and stores them under `data/vrplib`
【F:src/data-collection/download-standard-dataset.py†L1-L28】.

## 5. Solvers
### 5.1 Mixed Integer Programming (MIP)
`src/sol_1_mip` models CVRP with Pyomo and solves it with the SCIP optimizer. The model constructs
binary arc variables, capacity constraints and subtour elimination, then minimises travel cost
【F:src/sol_1_mip/main.py†L51-L82】.

Run on a pickled dataset:
```bash
python src/sol_1_mip/main.py path/to/dataset.pkl -o results.pkl
```

### 5.2 Branch-and-Cut (BCP)
`src/sol_1_mip_bcp` provides a branch-and-cut implementation using VRPSolverEasy. After loading an
instance, vehicle types, depot and customers are added, distances computed, and the solver is invoked
for up to 500 seconds【F:src/sol_1_mip_bcp/bcp.py†L7-L52】.

```bash
python src/sol_1_mip_bcp/main.py path/to/dataset.pkl -o results.pkl
```

### 5.3 Hybrid Genetic Search
The metaheuristic solver relies on the `hygese` library. Algorithm parameters include a configurable
`timeLimit`, and the solver returns best cost, routes and elapsed time【F:src/sol_3_mh_hgs/hgs.py†L1-L13】.

```bash
python src/sol_3_mh_hgs/main.py path/to/dataset.pkl 60 -o results.pkl
```

### 5.4 Google OR-Tools
An OR-Tools model with guided local search builds routing and capacity constraints via callback
functions and reports total distance and per-vehicle routes【F:src/sol_4_or_tools/model.py†L1-L85】.

```bash
python src/sol_4_or_tools/main.py path/to/dataset.pkl -o results.pkl
```

### 5.5 Reinforcement Learning Prototype
`src/sol-3-rl` contains a prototype RL environment and decoder built with PyTorch, including custom
embedding and attention modules for sequential decision making【F:src/sol-3-rl/main.py†L1-L40】.
The RL approach is experimental and intended for further research.

## 6. Results
Example outputs for each solver are stored under `results/<solver-name>/` as pickle files. Each entry
contains `[total_cost, routes, execution_time, instance_id]`. These artefacts enable comparative
analysis across solvers.

## 7. Report
The full research methodology, experiments and findings are documented in [Report.pdf](Report.pdf).
Refer to this report for a comprehensive discussion of the results and conclusions.

## 8. Getting Started
1. Install project dependencies (Pyomo, VRPSolverEasy, hygese, OR-Tools, PyTorch, NumPy, SciPy).
2. Download datasets using the data-collection script.
3. Execute one of the solver scripts with a dataset file to reproduce results.

---
For questions or contributions, please open an issue or submit a pull request.
