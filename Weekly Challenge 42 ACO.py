import numpy as np

# Weekly Challenge 42: Combinatorial Optimization - Ant Colony Optimization (ACO) from Scratch
# Author: Ing. Arturo Javier Borbon Rojas

print("Initializing Ant Colony Optimization (ACO) Engine...")

class AntColony:
    def __init__(self, distances, n_ants, n_epochs, alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
        self.distances = distances
        self.n_nodes = distances.shape[0]
        self.n_ants = n_ants
        self.n_epochs = n_epochs
        
        # Hyperparameters
        self.alpha = alpha  # Importance of the pheromone trail (tau)
        self.beta = beta    # Importance of the heuristic visibility (eta = 1/distance)
        self.rho = evaporation_rate
        self.Q = Q          # Pheromone deposit factor
        
        # Initialize Pheromone matrix evenly
        self.pheromones = np.ones((self.n_nodes, self.n_nodes))
        
        # Heuristic information (Visibility matrix: 1 / distance)
        # Add a tiny epsilon to diagonal to prevent division by zero
        with np.errstate(divide='ignore'):
            self.visibility = 1.0 / (self.distances + np.eye(self.n_nodes) * 1e-10)
            
        self.global_best_path = None
        self.global_best_distance = float('inf')

    def optimize(self):
        print(f"Deploying {self.n_ants} artificial ants across {self.n_nodes} logistical nodes for {self.n_epochs} epochs:")
        
        for epoch in range(self.n_epochs):
            all_paths = []
            all_distances = []
            
            # 1. Ants construct solutions
            for ant in range(self.n_ants):
                path = self._construct_path()
                path_distance = self._calculate_path_distance(path)
                
                all_paths.append(path)
                all_distances.append(path_distance)
                
                # Update global best
                if path_distance < self.global_best_distance:
                    self.global_best_distance = path_distance
                    self.global_best_path = path
            
            # 2. Update Pheromones
            self._update_pheromones(all_paths, all_distances)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   ➔ Epoch {epoch+1:03d} | Shortest route found: {self.global_best_distance:.2f} km")
                
        return self.global_best_path, self.global_best_distance

    def _construct_path(self):
        # Start at a random node
        start_node = np.random.randint(self.n_nodes)
        path = [start_node]
        unvisited = set(range(self.n_nodes))
        unvisited.remove(start_node)
        
        current_node = start_node
        
        while unvisited:
            next_node = self._select_next_node(current_node, list(unvisited))
            path.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
            
        return path

    def _select_next_node(self, current_node, unvisited_nodes):
        # Transition probability equation: (tau^alpha) * (eta^beta) / Sum of all options
        probabilities = np.zeros(len(unvisited_nodes))
        
        for i, target in enumerate(unvisited_nodes):
            tau = self.pheromones[current_node, target] ** self.alpha
            eta = self.visibility[current_node, target] ** self.beta
            probabilities[i] = tau * eta
            
        # Normalize to create a valid probability distribution
        probabilities /= np.sum(probabilities)
        
        # Roulette wheel selection based on probabilities
        return np.random.choice(unvisited_nodes, p=probabilities)

    def _calculate_path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distances[path[i], path[i+1]]
        # Return to start to complete the loop
        distance += self.distances[path[-1], path[0]]
        return distance

    def _update_pheromones(self, all_paths, all_distances):
        # 1. Evaporation: all pheromones decay naturally
        self.pheromones *= (1 - self.rho)
        
        # 2. Deposit: ants leave pheromones on the edges they traversed
        for path, distance in zip(all_paths, all_distances):
            deposit_amount = self.Q / distance
            for i in range(len(path) - 1):
                self.pheromones[path[i], path[i+1]] += deposit_amount
                self.pheromones[path[i+1], path[i]] += deposit_amount # Symmetric TSP
            # Complete the loop
            self.pheromones[path[-1], path[0]] += deposit_amount
            self.pheromones[path[0], path[-1]] += deposit_amount

# --- Synthetic Logistics Dataset (School Transportation Network) ---
# Generate 10 random coordinate points (nodes)
np.random.seed(42)
coords = np.random.rand(10, 2) * 100

# Calculate the Euclidean distance matrix between all points
n_points = coords.shape[0]
distances = np.zeros((n_points, n_points))
for i in range(n_points):
    for j in range(n_points):
        distances[i, j] = np.linalg.norm(coords[i] - coords[j])

# Deploy the ACO algorithm
aco = AntColony(distances=distances, n_ants=20, n_epochs=50, alpha=1.0, beta=2.0, evaporation_rate=0.4)
best_route, best_dist = aco.optimize()

print("-" * 65)
print("LOGISTICAL OPTIMIZATION COMPLETE:")
print(f"Optimal Routing Sequence: {best_route}")
print(f"Total Circuit Distance:   {best_dist:.2f} km")