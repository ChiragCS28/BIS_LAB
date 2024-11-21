import numpy as np
import random

# Genetic Algorithm Parameters
population_size = 20
generations = 100
mutation_rate = 0.1

# Functions for the Genetic Algorithm
def calculate_route_cost(route, distance_matrix):
    """Calculate total distance of a single route."""
    cost = 0
    for i in range(len(route) - 1):
        cost += distance_matrix[route[i]][route[i + 1]]
    return cost

def calculate_total_cost(solution, distance_matrix):
    """Calculate total cost of all routes in the solution."""
    total_cost = 0
    for route in solution:
        total_cost += calculate_route_cost(route, distance_matrix)
    return total_cost

def generate_initial_population(customers, demands, vehicle_capacity):
    """Generate an initial population of solutions."""
    population = []
    for _ in range(population_size):
        random.shuffle(customers)
        solution = []
        current_route = [0]  # Start at the depot
        current_load = 0
        for customer in customers:
            if current_load + demands[customer] <= vehicle_capacity:
                current_route.append(customer)
                current_load += demands[customer]
            else:
                current_route.append(0)  # Return to depot
                solution.append(current_route)
                current_route = [0, customer]
                current_load = demands[customer]
        current_route.append(0)  # Return to depot for the last route
        solution.append(current_route)
        population.append(solution)
    return population

def crossover(parent1, parent2):
    """Perform crossover between two parents."""
    child = []
    for route1, route2 in zip(parent1, parent2):
        split_point = len(route1) // 2
        new_route = route1[:split_point] + [x for x in route2 if x not in route1[:split_point]]
        child.append(new_route)
    return child

def mutate(solution):
    """Perform mutation on a solution."""
    for route in solution:
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(1, len(route) - 1), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]

def select_parents(population, distance_matrix):
    """Select two parents using tournament selection."""
    tournament_size = 5
    selected = random.sample(population, tournament_size)
    selected.sort(key=lambda sol: calculate_total_cost(sol, distance_matrix))
    return selected[0], selected[1]

# User Input
print("Welcome to the Vehicle Routing Problem Solver!")
num_locations = int(input("Enter the number of locations (including the depot): "))
print(f"Provide the distance matrix for {num_locations} locations:")

distance_matrix = []
for i in range(num_locations):
    row = list(map(int, input(f"Row {i + 1}: ").split()))
    distance_matrix.append(row)

demands = list(map(int, input(f"Enter the demands for {num_locations - 1} customers (separated by spaces): ").split()))
demands = [0] + demands  # Depot has no demand

num_vehicles = int(input("Enter the number of vehicles: "))
vehicle_capacity = int(input("Enter the capacity of each vehicle: "))

# Run Genetic Algorithm
customers = list(range(1, num_locations))  # Exclude depot
population = generate_initial_population(customers, demands, vehicle_capacity)

for generation in range(generations):
    new_population = []
    for _ in range(population_size):
        parent1, parent2 = select_parents(population, distance_matrix)
        child = crossover(parent1, parent2)
        mutate(child)
        new_population.append(child)
    population = sorted(new_population, key=lambda sol: calculate_total_cost(sol, distance_matrix))[:population_size]

# Output Best Solution
best_solution = population[0]
print("\nOptimized Routes:")
for i, route in enumerate(best_solution, start=1):
    print(f"Vehicle {i}: {' -> '.join(map(str, route))}")
print("Total Distance:", calculate_total_cost(best_solution, distance_matrix))
