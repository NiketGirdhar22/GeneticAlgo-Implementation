import random
import numpy as np

# Function to create a random tour
def create_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

# Function to calculate the total distance of a tour
def total_distance(tour, distance_matrix):
    distance = 0
    for i in range(len(tour)):
        distance += distance_matrix[tour[i-1]][tour[i]]
    return distance

# Function to create an initial population
def create_population(population_size, n):
    return [create_tour(n) for _ in range(population_size)]

# Function to perform tournament selection
def tournament_selection(population, distance_matrix, k=3):
    selected = random.sample(population, k)
    selected.sort(key=lambda x: total_distance(x, distance_matrix))
    return selected[0]

# Function to perform ordered crossover
def ordered_crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]

    for item in parent2:
        if item not in child:
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = item
                    break
    return child

# Function to perform mutation
def mutate(tour, mutation_rate=0.01):
    for i in range(len(tour)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(tour) - 1)
            tour[i], tour[j] = tour[j], tour[i]
    return tour

# Genetic Algorithm
def genetic_algorithm(distance_matrix, population_size=100, generations=500, mutation_rate=0.01):
    n = len(distance_matrix)
    population = create_population(population_size, n)
    
    for generation in range(generations):
        new_population = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population, distance_matrix)
            parent2 = tournament_selection(population, distance_matrix)
            child1 = mutate(ordered_crossover(parent1, parent2), mutation_rate)
            child2 = mutate(ordered_crossover(parent2, parent1), mutation_rate)
            new_population.extend([child1, child2])
        
        population = new_population
        
        best_tour = min(population, key=lambda x: total_distance(x, distance_matrix))
        best_distance = total_distance(best_tour, distance_matrix)
        print(f"Generation {generation + 1}: Best Distance = {best_distance}")

    best_tour = min(population, key=lambda x: total_distance(x, distance_matrix))
    return best_tour, total_distance(best_tour, distance_matrix)

# Example usage
distance_matrix = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

best_tour, best_distance = genetic_algorithm(distance_matrix, population_size=100, generations=500, mutation_rate=0.01)
print("Best tour:", best_tour)
print("Best distance:", best_distance)