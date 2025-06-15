# Bivariate Normal Distribution – Math vs Physics Scores

This project analyzes the **joint distribution** of Math and Physics scores using **bivariate normal theory**. Both analytical calculations and Python-based simulation are provided.

---

### 📌 Problem Overview

#### ✅ Q1: Theoretical Calculations

Given:

- \( \mu_{\text{Math}} = 70 \)
- \( \mu_{\text{Physics}} = 75 \)
- \( \sigma_{\text{Math}} = 10 \)
- \( \sigma_{\text{Physics}} = 12 \)
- \( \rho = 0.7 \)

Tasks:

1. **Joint PDF** of the bivariate normal distribution
2. \( P(\text{Math} > 80 \ \text{and} \ \text{Physics} > 80) \)
3. \( \mathbb{E}[\text{Physics} \mid \text{Math} = 80] \) and conditional variance
4. \( P(\text{Physics} > 85 \mid \text{Math} = 80) \)
5. Estimated number of students scoring >90 in Physics out of 200

---

#### ✅ Q2: Python Simulation

Simulate 10,000 student scores and calculate:

- Sample correlation coefficient
- Proportion of students scoring above 80 in both subjects
- Scatter plot of Math vs Physics scores

---

### 🧮 Mathematical Highlights

- **Conditional Expectation:**
  \[
  \mathbb{E}[Y \mid X = x] = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X}(x - \mu_X)
  \]

- **Conditional Variance:**
  \[
  \text{Var}(Y \mid X) = \sigma_Y^2(1 - \rho^2)
  \]

- **Joint PDF (2D Gaussian):**
  \[
  f(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1 - \rho^2}} \exp\left(-\frac{1}{2(1 - \rho^2)} Q \right)
  \]
  where  
  \[
  Q = \left(\frac{x - \mu_X}{\sigma_X}\right)^2 - 2\rho\left(\frac{x - \mu_X}{\sigma_X}\right)\left(\frac{y - \mu_Y}{\sigma_Y}\right) + \left(\frac{y - \mu_Y}{\sigma_Y}\right)^2
  \]

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
