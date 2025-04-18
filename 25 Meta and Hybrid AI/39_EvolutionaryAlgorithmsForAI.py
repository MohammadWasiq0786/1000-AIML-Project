"""
Project 999: Evolutionary Algorithms for AI
Description
Evolutionary algorithms (EAs) are a family of optimization algorithms inspired by the process of natural selection. These algorithms are used to solve complex optimization problems by evolving a population of potential solutions over time. In this project, we will implement an evolutionary algorithm to optimize the hyperparameters of a machine learning model, demonstrating how evolutionary strategies can be used for model selection and optimization.

Key Concepts Covered:
Evolutionary Algorithms: An optimization technique inspired by natural selection, where a population of potential solutions evolves over generations.

Genetic Algorithm: A type of evolutionary algorithm that uses crossover (combining solutions) and mutation (randomly altering solutions) to explore the solution space.

Hyperparameter Optimization: Using evolutionary algorithms to find the best set of hyperparameters for a machine learning model.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
 
# Generate a population of hyperparameters for RandomForest
def generate_population(pop_size, param_ranges):
    population = []
    for _ in range(pop_size):
        individual = {
            'n_estimators': random.randint(param_ranges['n_estimators'][0], param_ranges['n_estimators'][1]),
            'max_depth': random.randint(param_ranges['max_depth'][0], param_ranges['max_depth'][1]),
            'min_samples_split': random.randint(param_ranges['min_samples_split'][0], param_ranges['min_samples_split'][1])
        }
        population.append(individual)
    return population
 
# Evaluate a set of hyperparameters (individual) using cross-validation
def evaluate_individual(individual, X_train, y_train, X_test, y_test):
    # Create RandomForest model with the individual's hyperparameters
    model = RandomForestClassifier(n_estimators=individual['n_estimators'],
                                   max_depth=individual['max_depth'],
                                   min_samples_split=individual['min_samples_split'])
    model.fit(X_train, y_train)
    # Evaluate model on the test set
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)
 
# Select two parents from the population based on their fitness (accuracy)
def select_parents(population, X_train, y_train, X_test, y_test):
    fitness_scores = [evaluate_individual(individual, X_train, y_train, X_test, y_test) for individual in population]
    # Select two parents based on their fitness (higher fitness = better chance of selection)
    parents = np.random.choice(population, size=2, p=np.array(fitness_scores) / np.sum(fitness_scores))
    return parents
 
# Perform crossover between two parents to create offspring
def crossover(parents):
    parent1, parent2 = parents
    offspring = {}
    for param in parent1:
        # Choose randomly from the two parents
        offspring[param] = random.choice([parent1[param], parent2[param]])
    return offspring
 
# Perform mutation on an offspring (randomly change one hyperparameter)
def mutate(offspring, param_ranges):
    mutation_param = random.choice(list(offspring.keys()))
    # Mutate the selected parameter by randomly changing its value
    offspring[mutation_param] = random.randint(param_ranges[mutation_param][0], param_ranges[mutation_param][1])
    return offspring
 
# Genetic Algorithm for optimizing hyperparameters
def genetic_algorithm(X_train, y_train, X_test, y_test, generations=10, pop_size=10):
    param_ranges = {
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10)
    }
    
    # Generate initial population
    population = generate_population(pop_size, param_ranges)
    
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        # Select parents
        parents = select_parents(population, X_train, y_train, X_test, y_test)
        
        # Crossover to generate offspring
        offspring = crossover(parents)
        
        # Mutation
        offspring = mutate(offspring, param_ranges)
        
        # Evaluate the new offspring
        offspring_fitness = evaluate_individual(offspring, X_train, y_train, X_test, y_test)
        
        # Replace the worst-performing individual with the new offspring
        fitness_scores = [evaluate_individual(individual, X_train, y_train, X_test, y_test) for individual in population]
        worst_individual_idx = np.argmin(fitness_scores)
        population[worst_individual_idx] = offspring
        
        print(f"Best Accuracy in Generation {generation + 1}: {max(fitness_scores):.4f}")
    
    # Return the best individual after all generations
    best_individual_idx = np.argmax(fitness_scores)
    best_individual = population[best_individual_idx]
    print(f"Best Hyperparameters: {best_individual}")
    return best_individual
 
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
 
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Run the Genetic Algorithm for hyperparameter optimization
best_params = genetic_algorithm(X_train, y_train, X_test, y_test, generations=10, pop_size=10)
 
# Train the final model with the best hyperparameters
final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                     max_depth=best_params['max_depth'],
                                     min_samples_split=best_params['min_samples_split'])
final_model.fit(X_train, y_train)
 
# Evaluate the final model
y_pred_final = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"Final Model Accuracy: {final_accuracy:.4f}")