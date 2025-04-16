"""
Project 26. Simple genetic algorithm
Description:
A genetic algorithm (GA) is an optimization technique inspired by natural selection and genetics. It evolves a population of candidate solutions over generations using operators like selection, crossover, and mutation. In this project, we implement a basic GA to maximize a mathematical function — a classic optimization problem.

About:
🧬 What This GA Does:
Maximizes the function f(x) = x * sin(x) over the interval [0, 10]

Evolves a population over multiple generations

Applies selection, crossover, and mutation

Tracks and prints the best solution each generation
"""

import numpy as np
 
# Objective function: Maximize f(x) = x * sin(x)
def fitness_function(x):
    return x * np.sin(x)
 
# Parameters
population_size = 10
generations = 30
mutation_rate = 0.1
x_bounds = (0, 10)
 
# Generate initial population (random real values)
population = np.random.uniform(low=x_bounds[0], high=x_bounds[1], size=(population_size,))
 
# Genetic Algorithm loop
for generation in range(generations):
    # Evaluate fitness for each individual
    fitness = fitness_function(population)
 
    # Select the top 50% of the population (tournament selection)
    sorted_indices = np.argsort(fitness)[::-1]
    parents = population[sorted_indices[:population_size // 2]]
 
    # Crossover (simple average crossover)
    offspring = []
    while len(offspring) < population_size // 2:
        p1, p2 = np.random.choice(parents, 2, replace=False)
        child = (p1 + p2) / 2
        offspring.append(child)
 
    # Mutation: randomly modify some offspring
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] += np.random.normal(0, 0.5)  # small random noise
            offspring[i] = np.clip(offspring[i], *x_bounds)
 
    # Form new generation
    population = np.concatenate([parents, offspring])
 
    # Print the best solution so far
    best_fitness = np.max(fitness_function(population))
    best_individual = population[np.argmax(fitness_function(population))]
    print(f"Generation {generation+1}: Best x = {best_individual:.4f}, f(x) = {best_fitness:.4f}")
 
# Final result
print("\nFinal Best Solution:")
print(f"  x = {best_individual:.4f}")
print(f"  f(x) = {best_fitness:.4f}")