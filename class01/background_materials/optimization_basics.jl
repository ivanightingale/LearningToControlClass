### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 881eed45-e7f0-4785-bde8-530e378d7050
begin
using Pkg; Pkg.activate("..")
Pkg.instantiate()
end

# ╔═╡ 9f5675a3-07df-4fb1-b683-4c5fd2a85002
begin
	using PlutoUI
	using Random
	using LinearAlgebra
	using HypertextLiteral
	using PlutoTeachingTools
	using ShortCodes, MarkdownLiteral
	using Random
	using Plots
	Random.seed!(8803)
end

# ╔═╡ 9ce52307-bc22-4f66-a4af-a4e4ac382212
begin
    using JuMP
    using HiGHS        # Solver for LPs and MILPs
    using Ipopt        # Solver for NLPs
    using Test         # For quick validation checks
    # const MOI = JuMP.MathOptInterface
end

# ╔═╡ 0df8b65a-0527-4545-bf11-00e9912bced0
md"""

| | | |
|-----------:|:--|:------------------|
|  Lecturer   | : | Rosemberg, Andrew |
|  Date   | : | 28 of July, 2025 |

# Background – Modeling Optimization Problems in JuMP 🏗️

This short Pluto notebook walks you through three small optimisation models of increasing
difficulty:

1. **Linear program (LP)**
2. **Mixed‑integer linear program (MILP)**
3. **Non‑linear program (NLP)** – a taste of what shows up constantly in **optimal‑control** and simulaing **non‑linear systems**.

For every task you will:

* Write down the mathematical formulation.
* Translate it into a JuMP model.
* Solve it.
* Run the provided `@testset` to make sure your implementation is correct.  
  When the tests are green ✅ you can be confident that your model is producing the expected answer.
"""

# ╔═╡ 6f67ca7c-1391-4cb9-b692-cd818e037587
md"""
---

## 1. Linear program – Production planning

A workshop makes **widgets** \(w\) and **gadgets** \(g\).

|            | Machine‑hours | Labour‑hours | Profit (\$) |
|------------|---------------|--------------|--------------|
| Widget (\(w\)) | 2             | 3            | 3            |
| Gadget (\(g\)) | 4             | 2            | 5            |

Resources available this week: **100 machine‑hours** and **90 labour‑hours**.

### 1.1  Your tasks
1. Write the mathematical model *(maximization)*.
2. Fill in the JuMP code in the next cell.
3. Run the tests.
"""

# ╔═╡ 49042d6c-cf78-46d3-bfee-a8fd7ddf3aa0
begin
# === Your LP model goes below ===
# Replace the contents of this cell with your own model.
model_lp = Model(HiGHS.Optimizer)

# Suggested variable names
# @variable(model_lp, w >= 0)
# @variable(model_lp, g >= 0)

# --- YOUR CODE HERE ---

# optimize!(model_lp)
end

# ╔═╡ 6fb672d0-5a18-4ccc-b7b3-184839c2401b
begin
# === Quick check ===
@testset "LP check" begin
    @test termination_status(model_lp) == MOI.OPTIMAL
    @test isapprox(objective_value(model_lp), 135.0; atol = 1e-3)
end
end

# ╔═╡ 808c505d-e10d-42e3-9fb1-9c6f384b2c3c
md"""
---

## 2. MILP – 0‑1 Knapsack

You have a backpack that can carry at most **10 kg**.  
There are three items:

| Item | Value | Weight |
|------|-------|--------|
| 1    | 10    | 4      |
| 2    | 7     | 3      |
| 3    | 5     | 2      |

### 2.1  Your tasks
1. Write the mathematical model with **binary** decision variables \(x_i \in \{0,1\}\).
2. Complete the JuMP model and solve it.
3. Pass the tests.
"""

# ╔═╡ 39617561-bbbf-4ef6-91e2-358dfe76581c
begin
# === Your MILP model goes below ===
# Replace the contents of this cell with your own model.
model_milp = Model(HiGHS.Optimizer)

# Example:
# @variable(model_milp, x[1:3], Bin)

# --- YOUR CODE HERE ---

# optimize!(model_milp)
end

# ╔═╡ 01367096-3971-4e79-ace2-83600672fbde
begin
# === Quick check ===
@testset "MILP check" begin
    @test termination_status(model_milp) == MOI.OPTIMAL
    @test isapprox(objective_value(model_milp), 22.0; atol = 1e-3)
end
end

# ╔═╡ 5e3444d0-8333-4f51-9146-d3d9625fe2e9
md"""
---

## 3. Non‑linear program – Rosenbrock valley

Non‑linear models dominate **optimal control** because discretising the differential equations that
describe a physical system almost always yields a **non‑linear program (NLP)**.

A classic (and benign) test problem is the **Rosenbrock** function

\[
\min_{x,\,y\in\mathbb R}  \; f(x,y)= (1-x)^{2} + 100\,(y - x^{2})^{2}.
\]

It has a single global optimum at \((x^{\star},y^{\star}) = (1,1)\) with \(f^{\star}=0\).

### 3.1  Your tasks
1. Build and solve the model with **Ipopt**.
2. Inspect the solution and objective.
3. Check your work below.
"""

# ╔═╡ 00728de8-3c36-48c7-8520-4c9f408a7c5f
begin
# === Your NLP model goes below ===
# Replace the contents of this cell with your own model.
model_nlp = Model(Ipopt.Optimizer)

# --- YOUR CODE HERE ---

# optimize!(model_nlp)
end

# ╔═╡ 254b9a87-17f9-4fea-8b28-0e3873b58fe2
begin
# === Quick check ===
@testset "NLP check" begin
    @test termination_status(model_nlp) == MOI.LOCALLY_SOLVED || termination_status(model_nlp) == MOI.OPTIMAL
    @test isapprox(objective_value(model_nlp), 0.0; atol = 1e-6)
end
end

# ╔═╡ 147fe732-fe65-4226-af43-956b33a75bff
md"""
---

## Why non‑linear models matter in optimal control 🚀

When you discretise a continuous‑time optimal‑control problem (for example with **direct collocation**)
you obtain an optimisation problem whose variables are the states, controls, and possibly parameters
at many discrete time points:

```math
\begin{aligned}
&\min_{x_{k},u_{k}} && \sum_{k=0}^{N-1} \; \ell(x_{k},u_{k}) \\
&\text{s.t.} && x_{k+1} = x_{k} + h\,f(x_{k},u_{k}), \qquad k=0,\dots,N-1, \\
& && g(x_{k},u_{k}) \le 0, \\
& && x_{0}=x_{\text{init}}, \; x_{N}=x_{\text{goal}}.
\end{aligned}
```

Even when \(f\) and \(g\) are **polynomial** the resulting constraints are *non‑linear* in the decision variables.
Hence your optimisation solver must tackle *general NLPs*.  
Getting comfortable with modelling and debugging small nonlinear examples like Rosenbrock will pay off
when you step up to thousands of variables in real control problems!
"""

# ╔═╡ Cell order:
# ╟─881eed45-e7f0-4785-bde8-530e378d7050
# ╟─9f5675a3-07df-4fb1-b683-4c5fd2a85002
# ╟─0df8b65a-0527-4545-bf11-00e9912bced0
# ╠═9ce52307-bc22-4f66-a4af-a4e4ac382212
# ╟─6f67ca7c-1391-4cb9-b692-cd818e037587
# ╠═49042d6c-cf78-46d3-bfee-a8fd7ddf3aa0
# ╠═6fb672d0-5a18-4ccc-b7b3-184839c2401b
# ╠═808c505d-e10d-42e3-9fb1-9c6f384b2c3c
# ╠═39617561-bbbf-4ef6-91e2-358dfe76581c
# ╠═01367096-3971-4e79-ace2-83600672fbde
# ╠═5e3444d0-8333-4f51-9146-d3d9625fe2e9
# ╠═00728de8-3c36-48c7-8520-4c9f408a7c5f
# ╠═254b9a87-17f9-4fea-8b28-0e3873b58fe2
# ╟─147fe732-fe65-4226-af43-956b33a75bff
