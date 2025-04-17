"""
Project 483: Portfolio Optimization with AI
Description:
Portfolio optimization aims to select the best mix of assets to maximize returns and minimize risk. In this project, we'll implement a simple mean-variance optimization using historical returns of a set of assets. We’ll then use a genetic algorithm to optimize the weights of each asset in the portfolio.

✅ What It Does:
Simulates portfolio optimization by using Genetic Algorithms to select optimal asset weights.

Optimizes for the highest Sharpe ratio by evaluating annualized return and volatility.

Plots the portfolio growth over time to show how the strategy would perform.

Key Extensions and Customizations:
Multi-objective optimization: You can optimize for both return and risk-adjusted return (e.g., using Sharpe ratio).

Use real financial data: Integrate with live stock data APIs (e.g., Yahoo Finance, Alpha Vantage) for real-time portfolio updates.

Add transaction costs: Include fees or slippage in the strategy for more realistic performance.

Expand with more features: Integrate more advanced strategies like mean-variance optimization, machine learning-based methods, or even Black-Litterman model.

You can later extend this project by:

Using real stock data from APIs like Yahoo Finance, Alpha Vantage, or Quandl

Implementing advanced models such as Black-Litterman, Markowitz optimization, or machine learning-based approache
"""


import numpy as np
import pandas as pd
import yfinance as yf
import random
import matplotlib.pyplot as plt
 
# 1. Download historical stock data for a portfolio of assets (e.g., Apple, Microsoft, Google)
assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
stock_data = yf.download(assets, start="2015-01-01", end="2021-01-01")['Adj Close']
 
# 2. Calculate daily returns
returns = stock_data.pct_change().dropna()
 
# 3. Simulate portfolio optimization using a genetic algorithm
 
# Helper function to calculate portfolio performance
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized volatility
    return portfolio_return, portfolio_volatility
 
# Fitness function (negative Sharpe Ratio) for genetic algorithm
def fitness_function(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    p_return, p_volatility = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility  # We want to minimize this
 
# 4. Genetic algorithm for portfolio optimization
def genetic_algorithm(returns, population_size=50, generations=100, mutation_rate=0.1):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Generate an initial random population of portfolios
    population = []
    for _ in range(population_size):
        weights = np.random.random(len(assets))
        weights /= np.sum(weights)
        population.append(weights)
    
    # Evolution process
    for gen in range(generations):
        fitness_scores = [fitness_function(weights, mean_returns, cov_matrix) for weights in population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population))]
        
        # Keep the best half
        population = sorted_population[:population_size//2]
        
        # Crossover and mutation to create the next generation
        next_generation = []
        while len(next_generation) < population_size:
            parent1, parent2 = random.choices(population[:5], k=2)  # Select two parents
            crossover_point = random.randint(1, len(assets)-1)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            
            # Mutation
            if random.random() < mutation_rate:
                mutation_point = random.randint(0, len(assets)-1)
                child[mutation_point] = random.random()
                child /= np.sum(child)  # Normalize to sum to 1
            
            next_generation.append(child)
        
        population = next_generation
    
    # Best portfolio from the final generation
    best_weights = population[0]
    return best_weights
 
# 5. Run genetic algorithm
best_weights = genetic_algorithm(returns)
 
# 6. Calculate portfolio performance for the best portfolio
mean_returns = returns.mean()
cov_matrix = returns.cov()
p_return, p_volatility = calculate_portfolio_performance(best_weights, mean_returns, cov_matrix)
 
# 7. Display results
print(f"Optimal Portfolio Weights: {best_weights}")
print(f"Annualized Return: {p_return:.2f}")
print(f"Annualized Volatility: {p_volatility:.2f}")
 
# 8. Visualize portfolio performance
portfolio_value = 1000000  # Start with $1,000,000 investment
investment = portfolio_value * np.array(best_weights)
portfolio_growth = (returns.dot(best_weights) + 1).cumprod() * portfolio_value
 
plt.figure(figsize=(14, 7))
plt.plot(portfolio_growth, label='Optimized Portfolio')
plt.title("Portfolio Performance")
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USD)')
plt.legend()
plt.show()