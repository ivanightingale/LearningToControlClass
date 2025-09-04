# Class 2 — 08/29/2025

**Presenter:** Arnaud Deza

**Topic:** Numerical optimization for control (gradient/SQP/QP); ALM vs. interior-point vs. penalty methods

---

## Overview

This class covers the fundamental numerical optimization techniques essential for optimal control problems. We explore gradient-based methods, Sequential Quadratic Programming (SQP), and various approaches to handling constraints including Augmented Lagrangian Methods (ALM), interior-point methods, and penalty methods.

## Interactive Materials

The class is structured around four interactive Jupyter notebooks:

1. **[Part 1: Minimization via Newton's Method](part1_minimization.html)**
   - Unconstrained optimization fundamentals
   - Newton's method implementation
   - Hessian matrix and regularization

2. **[Part 1: Root Finding & Backward Euler](part1_root_finding.html)**
   - Root-finding algorithms for implicit integration
   - Fixed-point iteration vs. Newton's method
   - Application to pendulum dynamics

3. **[Part 2: Equality Constraints](part2_eq_constraints.html)**
   - Lagrange multiplier theory
   - KKT conditions for equality constraints
   - Quadratic programming implementation

4. **[Part 3: Interior-Point Methods](part3_ipm.html)**
   - Inequality constraint handling
   - Barrier methods and log-barrier functions
   - Comparison with penalty methods

## Additional Resources

- **[Lecture Slides (PDF)](ISYE_8803___Lecture_2___Slides.pdf)** - Complete slide deck
- **[Demo Script](penalty_barrier_demo.py)** - Python demonstration of penalty vs. barrier methods
- **[LaTeX Source](main.tex)** - Source code for lecture slides

## Key Learning Outcomes

- Understand gradient-based optimization methods
- Implement Newton's method for minimization
- Apply root-finding techniques for implicit integration
- Solve equality-constrained optimization problems
- Compare different constraint handling methods
- Implement Sequential Quadratic Programming (SQP)

## Prerequisites

- Solid understanding of linear algebra and calculus
- Familiarity with Julia programming
- Basic knowledge of differential equations
- Understanding of optimization concepts from Class 1

## Next Steps

This class provides the foundation for advanced topics in subsequent classes, including Pontryagin's Maximum Principle, nonlinear trajectory optimization, and stochastic optimal control.

