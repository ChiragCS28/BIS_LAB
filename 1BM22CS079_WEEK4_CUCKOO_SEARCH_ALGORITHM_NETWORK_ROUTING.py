import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Problem Parameters
AREA_SIZE = (100, 100)  # Target area dimensions (100x100)
NUM_SENSORS = 20        # Number of sensors to place
COVERAGE_RADIUS = 10    # Coverage radius of each sensor
NUM_NESTS = 50          # Number of possible solutions (nests)
PA = 0.25               # Probability of discovery (abandonment)
MAX_ITER = 10           # Number of iterations

# Optimized Fitness Function (Coverage + Energy Efficiency)
def calculate_fitness(sensors, area_size, coverage_radius):
    """
    Optimized Fitness Function based on coverage and energy consumption.
    """
    covered_area = np.zeros(area_size)
    energy_consumption = 0
    for sensor in sensors:
        x, y = sensor
        x_min, y_min = max(0, int(x - coverage_radius)), max(0, int(y - coverage_radius))
        x_max, y_max = min(area_size[0], int(x + coverage_radius)), min(area_size[1], int(y + coverage_radius))
        
        # Check coverage in the bounding box around the sensor
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                if np.sqrt((i - x) ** 2 + (j - y) ** 2) <= coverage_radius:
                    covered_area[i, j] = 1
        energy_consumption += np.sqrt(x**2 + y**2)  # Energy is proportional to distance from origin
    
    # Maximize coverage and minimize energy consumption
    fitness = np.sum(covered_area) - energy_consumption
    return fitness

# Lévy flight function for generating new solutions
def levy_flight(dim):
    # Generate a random Lévy flight step size
    beta = 1.5  # Parameter for Lévy flight
    sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
             (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    step = np.random.normal(0, sigma, dim)
    return step

# Cuckoo Search Algorithm
def cuckoo_search():
    # Initialize nests (random sensor placements)
    nests = [np.random.rand(NUM_SENSORS, 2) * AREA_SIZE for _ in range(NUM_NESTS)]
    fitness = [calculate_fitness(nest, AREA_SIZE, COVERAGE_RADIUS) for nest in nests]
    
    best_nest = nests[np.argmax(fitness)]  # Best solution found
    best_fitness = max(fitness)

    for iteration in range(MAX_ITER):
        # Generate new nests using Lévy flights
        new_nests = []
        for nest in nests:
            step = levy_flight(2)  # Move in 2D space
            new_nest = nest + step
            new_nest = np.clip(new_nest, 0, AREA_SIZE)  # Keep the sensors within the area
            new_nests.append(new_nest)

        # Evaluate fitness of new nests
        new_fitness = [calculate_fitness(nest, AREA_SIZE, COVERAGE_RADIUS) for nest in new_nests]
        
        # Abandon worst nests and replace them with new nests
        for i in range(NUM_NESTS):
            if new_fitness[i] > fitness[i]:
                nests[i] = new_nests[i]
                fitness[i] = new_fitness[i]
        
        # Discovery process (abandon some nests with probability PA)
        for i in range(NUM_NESTS):
            if random.random() < PA:
                nests[i] = np.random.rand(NUM_SENSORS, 2) * AREA_SIZE  # Reinitialize the nest
                fitness[i] = calculate_fitness(nests[i], AREA_SIZE, COVERAGE_RADIUS)
        
        # Update the best solution found
        best_nest = nests[np.argmax(fitness)]
        best_fitness = max(fitness)

        print(f"Iteration {iteration + 1}: Best Fitness = {best_fitness}")

    return best_nest, best_fitness

# Run Cuckoo Search
best_nest, best_fitness = cuckoo_search()

# Displaying the results
print("Best Fitness Achieved:", best_fitness)

# Plot the best sensor placement
x, y = zip(*best_nest)
plt.figure(figsize=(8, 8))
plt.scatter(x, y, color='red', label='Sensors')
plt.xlim(0, AREA_SIZE[0])
plt.ylim(0, AREA_SIZE[1])
plt.title('Optimal Sensor Placement')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

