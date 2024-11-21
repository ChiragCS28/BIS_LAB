import numpy as np

# Example of a network routing objective function:
# Assume network is represented as an adjacency matrix with weights as cost/latency between nodes
def objective_function(route, adj_matrix):
    """
    Objective function to optimize the total cost (e.g., latency or energy) of a given route.
    route: A list of node indices representing the route.
    adj_matrix: The adjacency matrix of the network, where adj_matrix[i][j] is the cost from node i to j.
    """
    total_cost = 0
    for i in range(len(route) - 1):
        total_cost += adj_matrix[route[i], route[i+1]]  # Add the cost of each edge in the path
    return -total_cost  # We aim to minimize the total cost, so we negate it for maximization

# Generate initial nests (paths) randomly within network bounds
def initialize_nests(num_nests, num_nodes):
    # Each nest is a random route from node 0 to the destination (node n-1)
    nests = []
    for _ in range(num_nests):
        route = np.random.permutation(num_nodes).tolist()  # Random route
        nests.append(route)
    return nests

# Levy flight for generating new solutions
def levy_flight(Lambda, dim):
    sigma_u = (np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2) /
               (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))) ** (1 / Lambda)
    sigma_v = 1.0
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, sigma_v, size=dim)
    step = u / np.abs(v) ** (1 / Lambda)
    return step

# Cuckoo Search Algorithm for Network Routing
def cuckoo_search(adj_matrix, source, destination):
    # User inputs
    num_nests = int(input("Enter the number of nests (population size): "))
    num_nodes = adj_matrix.shape[0]  # Number of nodes in the network
    max_iterations = int(input("Enter the number of iterations: "))
    pa = float(input("Enter the fraction of nests to abandon (0 to 1): "))
    Lambda = float(input("Enter the step-size parameter for Levy flights (1.5 is common): "))
    
    # Initialize nests and fitness
    nests = initialize_nests(num_nests, num_nodes)
    fitness = np.array([objective_function(nest, adj_matrix) for nest in nests])
    best_nest = nests[np.argmax(fitness)]
    best_fitness = np.max(fitness)
    
    # Main loop
    for iteration in range(max_iterations):
        # Generate new solutions via Levy flight
        for i in range(num_nests):
            step = levy_flight(Lambda, num_nodes)
            # Generate a new nest by perturbing the current best nest
            new_nest = nests[i] + step * (nests[i] - best_nest)
            # Ensure new nest is a valid route (valid node sequence)
            new_nest = np.clip(new_nest, 0, num_nodes - 1).astype(int)
            new_fitness = objective_function(new_nest, adj_matrix)
            
            # Update nest if new solution is better
            if new_fitness > fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fitness

        # Sort nests by fitness and abandon a fraction of the worst nests
        num_abandon = int(pa * num_nests)
        if num_abandon > 0:
            worst_indices = np.argsort(fitness)[:num_abandon]
            for idx in worst_indices:
                # Reinitialize the worst nests with new random routes
                nests[idx] = np.random.permutation(num_nodes).tolist()
                fitness[idx] = objective_function(nests[idx], adj_matrix)

        # Update global best
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_nest = nests[current_best_idx]
            best_fitness = fitness[current_best_idx]

        print(f"Iteration {iteration + 1}/{max_iterations}: Best Fitness = {best_fitness:.4f}")
    
    print("\nBest solution found:")
    print(f"Route: {best_nest}")
    print(f"Total Cost (Fitness): {best_fitness:.4f}")

# Example: Running the Cuckoo Search Algorithm for a simple network
# Create a simple network adjacency matrix (costs between nodes)
adj_matrix = np.array([
    [0, 10, 15, 20, 25],
    [10, 0, 35, 25, 30],
    [15, 35, 0, 30, 5],
    [20, 25, 30, 0, 15],
    [25, 30, 5, 15, 0]
])

source = 0
destination = 4

# Run the Cuckoo Search Algorithm for network routing
cuckoo_search(adj_matrix, source, destination)

