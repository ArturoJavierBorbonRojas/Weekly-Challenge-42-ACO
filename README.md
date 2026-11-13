# Weekly Challenge 42: Combinatorial Metaheuristics (Ant Colony Optimization from Scratch)

## Description
For Week 42, I shifted my focus from continuous space prediction algorithms back to the realm of **Advanced Optimization**, specifically targeting discrete combinatorial problems. I programmed an **Ant Colony Optimization (ACO)** engine entirely from scratch using pure NumPy.

While algorithms like Gradient Descent or PSO are excellent for continuous mathematical surfaces, they fail at discrete routing challenges like the Traveling Salesperson Problem (TSP) or school bus routing. Trying every possible combination of a 15-node route takes centuries of computation. ACO solves this by mimicking biological ants: deploying artificial agents that explore paths probabilistically, leaving "mathematical pheromones" on the most efficient routes.

## How it works (The Math)
The algorithm relies on two primary equations:

1. **State Transition Probability:** An ant decides which node to visit next based on a weighted roulette wheel. The probability $P$ of moving from node $i$ to node $j$ is calculated by combining the pheromone trail ($\tau$) and the heuristic visibility ($\eta = 1/distance$):
   $$P_{ij} = \frac{[\tau_{ij}]^\alpha [\eta_{ij}]^\beta}{\sum_{k \in \text{unvisited}} [\tau_{ik}]^\alpha [\eta_{ik}]^\beta}$$

2. **Pheromone Update Rule:** At the end of each epoch, two things happen to the pheromone matrix:
   * **Evaporation:** All pheromones decay by a rate of $\rho$ to prevent the algorithm from getting stuck in local optima.
   * **Deposit:** Each ant deposits new pheromones inversely proportional to the total distance of its completed path (shorter paths get more pheromone).
   $$\tau_{ij} = (1 - \rho)\tau_{ij} + \sum_{k=1}^{m} \Delta \tau_{ij}^k$$

## Technical Highlights
* **Pure Matrix Operations:** The heuristic visibility matrix and probability normalizations are fully vectorized using NumPy.
* **Stochastic Convergence:** The algorithm reliably converges on the global optimum sequence for a 10-node network in under 50 epochs by perfectly balancing exploration ($\rho$ evaporation) and exploitation (pheromone deposits).

## Dependencies
* Python 3.14.3
* NumPy
