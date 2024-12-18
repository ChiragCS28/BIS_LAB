#TSP

import numpy as np
import random

# Problem Parameters
CITIES = [
    (0, 0),  # City 1
    (1, 3),  # City 2
    (4, 3),  # City 3
    (6, 1),  # City 4
    (3, 0)   # City 5
]
POPULATION_SIZE = 50
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.8
GENERATIONS = 100

def calculate_distance(city1, city2):
    """Calculate Euclidean distance between two cities."""
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Precompute the distance matrix
DISTANCE_MATRIX = [
    [calculate_distance(c1, c2) for c2 in CITIES] for c1 in CITIES
]

def fitness(route):
    """Calculate the total distance of a route (lower is better)."""
    return -sum(DISTANCE_MATRIX[route[i]][route[i + 1]] for i in range(len(route) - 1)) - DISTANCE_MATRIX[route[-1]][route[0]]

def create_population():
    """Generate an initial population of random routes."""
    return [random.sample(range(len(CITIES)), len(CITIES)) for _ in range(POPULATION_SIZE)]

def select(population, fitness_scores):
    """Perform roulette wheel selection based on fitness."""
    total_fitness = sum(fitness_scores)
    probabilities = [f / total_fitness for f in fitness_scores]
    return random.choices(population, weights=probabilities, k=POPULATION_SIZE)

def crossover(parent1, parent2):
    """Perform ordered crossover between two parents."""
    if random.random() < CROSSOVER_RATE:
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [-1] * len(parent1)
        child[start:end + 1] = parent1[start:end + 1]
        fill_values = [x for x in parent2 if x not in child]
        idx = 0
        for i in range(len(child)):
            if child[i] == -1:
                child[i] = fill_values[idx]
                idx += 1
        return child
    return parent1

def mutate(route):
    """Perform mutation by swapping two cities."""
    if random.random() < MUTATION_RATE:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

def genetic_algorithm():
    """Solve the TSP using Genetic Algorithm."""
    population = create_population()
    best_route = None
    best_fitness = float('-inf')

    for generation in range(GENERATIONS):
        fitness_scores = [fitness(route) for route in population]
        max_fitness = max(fitness_scores)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_route = population[fitness_scores.index(max_fitness)]

        print(f"Generation {generation}: Best Fitness = {-best_fitness:.2f}")
        
        # Selection, crossover, and mutation
        selected_population = select(population, fitness_scores)
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1, parent2 = selected_population[i], selected_population[(i + 1) % POPULATION_SIZE]
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            next_generation.extend([child1, child2])

        population = next_generation

    return best_route, -best_fitness

# Run the Genetic Algorithm
best_route, best_distance = genetic_algorithm()
print("\nBest Route Found:", best_route)
print("Distance of Best Route:", best_distance)

