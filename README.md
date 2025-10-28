# AUPO - Abstracted until proven otherwise: A reward distribution based abstraction algorithm

## Purpose

This is the repository accompanying the paper "AUPO - Abstracted until proven otherwise: A reward distribution based abstraction algorithm" which
contains the code to reproduce the experiments and the results of the paper.

## Cite this work

If you use this work, please cite it as:

```bibtex
@misc{schmöcker2025aupoabstractedproven,
      title={AUPO - Abstracted Until Proven Otherwise: A Reward Distribution Based Abstraction Algorithm}, 
      author={Robin Schmöcker and Alexander Dockhorn and Bodo Rosenhahn},
      year={2025},
      eprint={2510.23214},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.23214}, 
}
```

## Abstract
We introduce a novel, drop-in modification to Monte Carlo Tree Search's (MCTS) decision policy 
that we call AUPO. Comparisons based on a range of IPPC benchmark problems show that 
AUPO clearly outperforms MCTS. AUPO is an automatic action abstraction algorithm that solely 
relies on reward distribution statistics acquired during the MCTS. Thus, unlike other 
automatic abstraction algorithms, AUPO requires neither access to transition
probabilities nor does AUPO require a directed acyclic search graph to build its abstraction, 
allowing AUPO to detect symmetric actions that state-of-the-art frameworks like ASAP 
struggle with when the resulting symmetric states are far apart in state space. Furthermore, as 
AUPO only affects the decision policy, it is not mutually exclusive with other abstraction 
techniques that only affect the tree search.

## Installation

To build the project from source, you will need a C++ compiler supporting the C++20 standard or higher (a lower standard probably works too but we have not tested that). The project
is self-contained and does not require any additional installation.

To compile with [CMake](https://cmake.org/) you need to have CMake installed on your system. A `CMakeLists.txt` file is already provided for configuring the build.

**Steps:**

1. **Clone the repository:**
    ```bash
    git clone https://github.com/codebro634/Aupo.git
    cd Aupo
    ```

2. **Create a build directory (optional but recommended):**
    ```bash
    mkdir build
    cd build
    ```

3. **Generate build files using CMake:**
    ```bash
    cmake -DCMAKE_CXX_COMPILER=/path/to/your/c++-compiler -DCMAKE_C_COMPILER=/path/to/your/c-compiler ..
    ```

4. **Compile the project:**
    ```bash
    cmake --build .
    ```
   *This will invoke the underlying build system (e.g., `make` or `ninja`) to compile the source code.*

If no errors occur, the compiled binary `Aupo` should now be available in the `build` directory.

## Usage

The program is called with the following arguments:

`--seed`: The seed for the random number of generator. Running the program with the same seed will produce the same results.

`--n`: The number of episodes to run.

`--model`: The abbreviation for the model to use. Possible values are
`sa` for SysAdmin, `gol` for Game of Life, `aa` for AcademicAdvising,  `tam` for Tamarisk, `mab` for Multi-armed Bandit, and `wf` for Wildfire.

`--margs`: The arguments for the model. Mostly this is a game map to be specified which can be found in the
`resources` folder.

`--agent`:  Which agent to use. The only options are `mcts` and `aupo`.

`--aargs`: The arguments for the agent. A list of required and optional arguments can be found in main.cpp in getAgent.

The following shows an example call of running Aupo with 500 Mcts-iterations, uniform policy at the root,
a confidence of 0.9, a distribution tracking depth of 3, and using no additional filters
for 200 episodes on a Game of Life map.

```bash
--seed 42 -n 200  --model gol --margs map=../resources/GameOfLifeMaps/3.txt  --agent aupo --aargs confidence=0.9 --aargs distribution_layers=3 --aargs filter_by_std=0 --aargs use_rollout_distribution=0 --aargs iterations=500
```

