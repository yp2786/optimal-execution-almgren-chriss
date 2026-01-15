# Optimal Trade Execution: Almgren-Chriss Framework
Optimal liquidation of large equity positions using the Almgren-Chriss framework with CVXPY optimization

## Overview
This repository contains an implementation of the **Almgren-Chriss (2000)** model for optimal liquidation. The project demonstrates the application of **Stochastic Calculus** and **Convex Optimization** to solve the fundamental problem of institutional trading: minimizing the total cost of execution (Implementation Shortfall) while managing timing risk.

## Mathematical Formulation

I model the price process $S_t$ as a stochastic process influenced by the trader's own actions through **Market Impact**.

### 1. Price Dynamics
The price at time $k$ is governed by:
$$S_k = S_{k-1} + \sigma \sqrt{\tau} \xi_k - \tau \gamma n_k$$

Where:
* $S_k$: Fundamental price at time $k$.
* $\sigma$: Arithmetic volatility.
* $\xi_k$: Standard normal random variable $\mathcal{N}(0,1)$.
* $\gamma$: **Permanent Impact** coefficient.
* $n_k$: Number of shares sold in interval $k$.

### 2. Execution Price
The actual price received per share is affected by **Temporary Impact** $\eta$:
$$\tilde{S}_k = S_k - \eta \frac{n_k}{\tau}$$

### 3. The Objective Function
As an Operations Research problem, we minimize the **Mean-Variance Utility** function of the Total Shortfall $X$:
$$\min_{n} \mathbb{E}[X] + \lambda \mathbb{V}[X]$$

The expected cost (Shortfall) and variance are derived as:
$$\mathbb{E}[X] = \frac{1}{2}\gamma X^2 + \left( \frac{\eta}{\tau} - \frac{1}{2}\gamma \right) \sum_{k=1}^N n_k^2$$
$$\mathbb{V}[X] = \sigma^2 \sum_{k=1}^N \tau x_k^2$$

where $x_k$ is the remaining inventory at time $k$.

## Methodology
* **Optimization:** `CVXPY` (Second-Order Cone Programming)
* **Analysis:** `NumPy`, `SciPy`
* **Visualization:** `Matplotlib`
* **Data Handling:** `Pandas`

## Repository Structure
* `/src`: Contains the `AlmgrenChrissOptimizer` class.
* `/notebooks`: 
    * `01_Market_Dynamics`: Simulation of impact functions.
    * `02_The_Optimizer`: Implementation of the CVXPY quadratic solver.
    * `03_Efficient_Frontier`: Monte Carlo analysis of the Risk-Reward tradeoff.

## Key Insights
* **Risk Aversion ($\lambda$):** As $\lambda$ increases, the optimal strategy front-loads trades to minimize the variance of the final shortfall, even at the cost of higher market impact.
* **The Efficient Frontier:** The model identifies the "Optimal Execution Frontier," allowing a desk to choose a strategy based on their specific risk level.
