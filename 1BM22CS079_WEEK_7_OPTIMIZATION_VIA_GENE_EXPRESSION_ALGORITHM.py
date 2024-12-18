#DNA SEQUENCE ALIGNMENT

import random

# Scoring constants
MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -2

# Problem Definition: DNA sequences
SEQ1 = "ACGTG"
SEQ2 = "ACTG"

# Define the scoring function
def calculate_fitness(sequence1, sequence2, alignment):
    score = 0
    for i in range(len(alignment)):
        if alignment[i][0] == alignment[i][1]:
            score += MATCH_SCORE
        elif "-" in alignment[i]:
            score += GAP_PENALTY
        else:
            score += MISMATCH_PENALTY
    return score

# Generate initial random population
def initialize_population(pop_size, seq1, seq2):
    population = []
    max_len = max(len(seq1), len(seq2))
    for _ in range(pop_size):
        alignment = []
        for i in range(max_len):
            # Randomly align bases from both sequences, using gaps where necessary
            base1 = seq1[i] if i < len(seq1) else '-'
            base2 = seq2[i] if i < len(seq2) else '-'
            alignment.append((base1, base2))
        population.append(alignment)
    print(f"Initial Population Size: {len(population)}")  # Debug print
    return population

# Selection based on fitness
def selection(population, seq1, seq2):
    fitness_scores = [calculate_fitness(seq1, seq2, alignment) for alignment in population]
    
    # Ensure all fitness scores are greater than zero by adding a small constant if necessary
    min_fitness = min(fitness_scores)
    if min_fitness <= 0:
        fitness_scores = [score - min_fitness + 1 for score in fitness_scores]  # Adjust so all scores are positive
        print("Adjusted Fitness Scores to avoid zero or negative scores.")
    
    print(f"Fitness Scores: {fitness_scores}")  # Debug print
    # Selection should still happen even if fitness scores are the same
    selected = random.choices(population, weights=fitness_scores, k=len(population) // 2)
    print(f"Selected Population Size: {len(selected)}")  # Debug print
    return selected

# Crossover: Swap alignment fragments between two parents
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# Mutation: Randomly alter an alignment
def mutate(alignment, seq1, seq2):
    for i in range(len(alignment)):
        if random.random() < 0.1:  # Mutation rate = 10%
            if random.random() > 0.5:
                # Randomly select new bases from both sequences
                base1 = seq1[random.randint(0, len(seq1) - 1)]
                base2 = seq2[random.randint(0, len(seq2) - 1)]
            else:
                # Randomly select gaps for mutations
                base1 = '-'
                base2 = '-'
            alignment[i] = (base1, base2)
    return alignment

# Main GEA function
def gene_expression_algorithm(seq1, seq2, generations=50, population_size=20):
    # Step 1: Initialize population
    population = initialize_population(population_size, seq1, seq2)
    best_alignment = None
    best_fitness = float('-inf')
    
    # Step 2: Iterate through generations
    for gen in range(generations):
        # Check if population is empty
        if not population:
            print(f"Error: Population is empty at generation {gen}")
            break
        
        # Evaluate fitness
        fitness_scores = [calculate_fitness(seq1, seq2, alignment) for alignment in population]
        
        # Track best solution
        current_best_fitness = max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_alignment = population[fitness_scores.index(current_best_fitness)]
        
        # Selection
        selected_population = selection(population, seq1, seq2)
        
        # Crossover
        new_population = []
        for i in range(0, len(selected_population), 2):
            if i + 1 < len(selected_population):
                child1, child2 = crossover(selected_population[i], selected_population[i + 1])
                new_population.extend([child1, child2])
        
        # Mutation
        new_population = [mutate(alignment, seq1, seq2) for alignment in new_population]
        
        # Replace old population
        population = new_population
        print(f"Population Size at Generation {gen}: {len(population)}")  # Debug print
    
    return best_alignment, best_fitness

# Run the GEA
best_alignment, best_fitness = gene_expression_algorithm(SEQ1, SEQ2)

# Output the results
print("Best Alignment:")
for pair in best_alignment:
    print(pair)
print("Best Fitness Score:", best_fitness)
