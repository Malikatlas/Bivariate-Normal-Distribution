# Bivariate Normal Distribution – Math vs Physics Scores

This project analyzes the **joint distribution** of Math and Physics scores using **bivariate normal theory**. Both analytical calculations and Python-based simulation are provided.

---

### 📌 Problem Overview

#### ✅ Q1: Theoretical Calculations

Given:

mu_Math = 70
mu_Physics = 75
sigma_Math = 10
sigma_Physics = 12
rho = 0.7 (correlation coefficient between Math and Physics scores)

Tasks:

Write the joint probability density function (PDF) of the bivariate normal distribution for Math and Physics scores.
Calculate the probability that a student scores more than 80 in both Math and Physics:
P(Math > 80 and Physics > 80)
If a student scores 80 in Math, compute:
The expected Physics score (E[Physics | Math = 80])
The conditional variance of Physics given Math
Find the probability that a student scoring 80 in Math scores more than 85 in Physics:
P(Physics > 85 | Math = 80)
If 200 students are surveyed, estimate how many are expected to score more than 90 in Physics.

---

#### ✅ Q2: Python Simulation

Simulate 10,000 student scores and calculate:

- Sample correlation coefficient
- Proportion of students scoring above 80 in both subjects
- Scatter plot of Math vs Physics scores

---

### 🧮 Mathematical Highlights


**Conditional Expectation:**  
E[Y | X = x] = mu_Y + rho * (sigma_Y / sigma_X) * (x - mu_X)


- **Conditional Variance:**
Var(Y | X) = sigma_Y^2 * (1 - rho^2)

f(x, y) = [1 / (2 * pi * sigma_X * sigma_Y * sqrt(1 - rho^2))] 
         * exp{ -1 / [2 * (1 - rho^2)] * Q }

where:
Q = ((x - mu_X) / sigma_X)^2 
    - 2 * rho * ((x - mu_X) / sigma_X) * ((y - mu_Y) / sigma_Y) 
    + ((y - mu_Y) / sigma_Y)^2


---

### 📊 Python Simulation (Snippet)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, pearsonr

mu = [70, 75]
sigma_math = 10
sigma_physics = 12
rho = 0.7

cov = [
    [sigma_math**2, rho * sigma_math * sigma_physics],
    [rho * sigma_math * sigma_physics, sigma_physics**2]
]

# Simulation
np.random.seed(42)
samples = np.random.multivariate_normal(mu, cov, 10000)
math_scores = samples[:, 0]
physics_scores = samples[:, 1]

# Plot
plt.scatter(math_scores, physics_scores, alpha=0.4, s=10)
plt.xlabel("Math Scores")
plt.ylabel("Physics Scores")
plt.title("Scatter Plot: Math vs Physics")
plt.grid(True)
plt.show()

# Analysis
corr, _ = pearsonr(math_scores, physics_scores)
prop_above_80 = np.mean((math_scores > 80) & (physics_scores > 80))
print(f"Sample Correlation: {corr:.3f}")
print(f"Proportion scoring above 80 in both: {prop_above_80:.3f}")
