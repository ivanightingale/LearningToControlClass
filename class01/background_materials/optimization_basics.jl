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
	using CairoMakie
	Random.seed!(8803)
end

# ╔═╡ 9ce52307-bc22-4f66-a4af-a4e4ac382212
begin
    using JuMP
    using HiGHS        # Solver for LPs and MILPs
    using Ipopt        # Solver for NLPs
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
* Pluto will run tests to check your answer.
  When the tests are green ✅ you can be confident that your model is producing the expected answer.
"""

# ╔═╡ 6f67ca7c-1391-4cb9-b692-cd818e037587
md"""
---

## 1. Linear program – Production planning

A workshop makes **widgets** \(w\) and **gadgets** \(g\). The table below specifies how many machine--hours and human labour hours are required to produce each product and how much profit the company makes after selling the product.

|            | Machine‑-hours | Labour‑-hours | Profit (\$) |
|------------|---------------|--------------|--------------|
| Widget (\(w\)) | 2             | 3            | 3            |
| Gadget (\(g\)) | 4             | 2            | 5            |

Resources available this week: **100 machine‑hours** and **90 labour‑hours**.

### 1.1  Your tasks
1. Write the mathematical model that maximizes profit! Let's assume we can sell continuous fractions of our produts. 
2. Fill in the JuMP code in the next cell.
"""

# ╔═╡ 49042d6c-cf78-46d3-bfee-a8fd7ddf3aa0
begin
# === Your LP model goes below ===
# Replace the contents of this cell with your own model.
model_lp = Model(HiGHS.Optimizer)

# Required variable names (used for testing)
@variable(model_lp, w >= 0) # number of Widgets
@variable(model_lp, g >= 0) # number of Gadgets

# --- YOUR CODE HERE ---

# optimize!(model_lp) # uncomment to optimize

# Let's look at our model
println(model_lp)
end

# ╔═╡ 1d3edbdd-7747-4651-b650-c6b9bf87b460
md"Tests will automatically fetch the optimal values from your solved model."

# ╔═╡ 248b398a-0cf5-4c2b-8752-7b9cc4e765d6
question_box(md"Did we get partial products?")

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
1. Write the mathematical model with **binary** decision variables $x_i \in \{0,1\}$ that maximizes the acumulated value of the items in your bag. No partial items allowed.
2. Complete the JuMP model and solve it.
3. Pass the tests.
"""

# ╔═╡ 39617561-bbbf-4ef6-91e2-358dfe76581c
begin
# === Your MILP model goes below ===
# Replace the contents of this cell with your own model.
model_milp = Model(HiGHS.Optimizer)

# Variables should be a vector named x (used for testing)
# @variable(model_milp, x[1:3] ...

# --- YOUR CODE HERE ---

# optimize!(model_milp)

# Let's look at our model
println(model_milp)
end

# ╔═╡ 01367096-3971-4e79-ace2-83600672fbde
begin
 	ground_truth_2 = (x = [1.0, 1.0, 1.0], obj = 22.0)

	ans2 = missing
    try
        ans2 = (
            x   = haskey(model_milp, :x) ? JuMP.value.(model_milp[:x]) : missing,
            obj = objective_value(model_milp),
        )
    catch
        ans2 = missing
    end

    good = !ismissing(ans2) &&
           all(isapprox.(ans2.x, ground_truth_2.x; atol=1e-3)) &&
           isapprox(ans2.obj, ground_truth_2.obj; atol=1e-3)

    if ismissing(ans2)
        still_missing()
    elseif good
        correct()
    else
        keep_working()
    end
end

# ╔═╡ 38b3a8f3-35ae-46da-91ce-0e4ba27ae098
question_box(md"Is the answer the same if we allow partial products?")

# ╔═╡ bca712e4-3f1c-467e-9209-e535aed5ab0a
md"""
### 2.2  Sudoku
(Credit to the similar JuMP Tutorial)

Now let us solve the classic sudoku problem!

1. Write the mathematical model with **binary** decision variables $x_i \in \{0,1\}$ that solves the following sudoku.
2. Complete the JuMP model and solve it.
3. Pass the tests.
"""

# ╔═╡ 3997d993-0a31-435e-86cd-50242746c305
@htl """
<img src="https://raw.githubusercontent.com/jump-dev/JuMP.jl/refs/heads/master/docs/src/assets/partial_sudoku.png"
	 width="800" height="600"
	 style="object-fit:contain;" />
"""		

# ╔═╡ 3f56ec63-1fa6-403c-8d2a-1990382b97ae
begin
# === Your MILP model goes below ===
# Replace the contents of this cell with your own model.
sudoku = Model(HiGHS.Optimizer)

# Variables should be a vector named x_s (used for testing)
@variable(sudoku, x_s[i = 1:9, j = 1:9, k = 1:9], Bin);

# --- YOUR CODE HERE ---

# optimize!(sudoku)

# Let's look at the stats of our model
sudoku
end

# ╔═╡ 0e8ed625-df85-4bd2-8b16-b475a72df566
begin
 	ground_truth_s = (x_ss = [[ 5  3  4  6  7  8  9  1  2];
 [6  7  2  1  9  5  3  4  8];
 [1  9  8  3  4  2  5  6  7];
 [8  5  9  7  6  1  4  2  3];
 [4  2  6  8  5  3  7  9  1];
 [7  1  3  9  2  4  8  5  6];
 [9  6  1  5  3  7  2  8  4];
 [2  8  7  4  1  9  6  3  5];
 [3  4  5  2  8  6  1  7  9]])

	anss = missing
    try
        anss = (
            x_ss   = haskey(sudoku, :x_s) ? JuMP.value.(sudoku[:x_s]) : missing,
        )
    catch
        anss = missing
    end

    goods = !ismissing(anss) &&
           all(isapprox.(anss.x_ss, ground_truth_s.x_ss; atol=1e-3))

    if ismissing(anss)
        still_missing()
    elseif goods
        correct()
    else
        keep_working()
    end
end

# ╔═╡ 5e3444d0-8333-4f51-9146-d3d9625fe2e9
md"""
---

## 3. Non‑linear program – Modified Rosenbrock valley

Non‑linear models dominate **optimal control** because discretising the differential equations that
describe a physical system almost always yields a **non‑linear program (NLP)**.

Let's modify the classic (and benign) **Rosenbrock** function

```math
\begin{aligned}
\min_{x,\,y} \quad & f(x,y) \;=\; -(1-x)^{2} + 100\,(y - x^{2})^{2} + 10000x \\
\text{s.t.}\quad  & -10 \le x \le 10,\\
                  & -60 \le y \le 60.
\end{aligned}
```

It has a single global minimum within the feasible reagion defined by the box constraints on $x$ and $y$.

### 3.1  Your tasks
1. Build a model to find this minimum and solve it with **Ipopt**.
2. Inspect the solution and objective.
3. Pass the tests
"""

# ╔═╡ 0e190de3-da60-41e9-9da5-5a0c7fefd1d7
f(x, y) = -(1-x)^2 + 100 * (y-x^2)^2 + 10000*x

# ╔═╡ cac18d70-b354-48c7-9f37-31ee0c585675
begin
# 1.  Domain grids
xs = range(-10,  10; length = 100)     # 100 x-points
ys = range(-60, 60; length = 100)    # 100 y-points

# 3.  Surface heights — matrix z[y, x]
zs = [f(x, y) for y in ys, x in xs]

# 4.  Create a 1×3 layout
fig = Figure(size = (1500, 500))

# 4a.  3-D surface
ax3d = Axis3(fig[1, 1]; xlabel = "y", ylabel = "x", zlabel = "f(x,y)",
             title = "Surface")
surface!(ax3d, ys, xs, zs; colormap = :viridis)
ax3d.azimuth[]   = deg2rad(-10)        # ≈ camera = (-10°, 30°)
ax3d.elevation[] = deg2rad( 30)

# 4b.  Slice at y = 
y0=40
ax_x = Axis(fig[1, 2]; xlabel = "x", ylabel = "f(x,$(y0))", title = "y = $(y0) slice")
lines!(ax_x, xs, f.(xs, y0))

# 4c.  Slice at x =
x_0 = 0
ax_y = Axis(fig[1, 3]; xlabel = "y", ylabel = "f($(x_0),y)", title = "x = $(x_0) slice")
lines!(ax_y, ys, f.(x_0, ys))

fig 
end

# ╔═╡ 00728de8-3c36-48c7-8520-4c9f408a7c5f
begin
# === Your NLP model goes below ===
# Replace the contents of this cell with your own model.
model_nlp = Model(Ipopt.Optimizer)

# Required named variables
@variable(model_nlp, x)
@variable(model_nlp, y)

# --- YOUR CODE HERE ---

# optimize!(model_nlp)

println(model_nlp)
end

# ╔═╡ 45541136-695a-4260-82c1-66d38ec44dcc
md"""
### 3.2  Intersecting-ellipse constraints  
Find the minimum of the same modified Rosenbrock objective, **but now the feasible
region is the intersection of three ellipses** defined only through their focal points
and the constant sum-of-distances $2a$:

| Ellipse | Focal points $(p_1,\,p_2)$ | Required sum of distances $2a$ |
|---------|---------------------------|--------------------------------|
| $E_1$   | $(-4,\,0),\;(4,\,0)$      | $12$ |
| $E_2$   | $(0,\,-5),\;(0,\,5)$      | $14$ |
| $E_3$   | $(-3,\,-3),\;(3,\,3)$     | $12$ |

Recall that a point $(x,y)$ lies **inside** an ellipse if the **sum of its Euclidean
distances to the two foci is _no greater_ than $2a$**.  
Formulate these three nonlinear constraints and use **Ipopt** to locate the optimal
$(x^*,y^*)$ and corresponding objective value.

1. Implement the model with the three ellipse constraints.  
2. Solve it and report the optimal point and objective.  
3. Verify that the solver stopped at a feasible point (all three distance sums $\le 2a$).
"""

# ╔═╡ b107bcd7-60ca-4f09-aa42-f8335e13233e
begin
	# ── NLP model ────────────────────────────────────────────────────────────────
    model_nlp2 = Model(Ipopt.Optimizer)

    # Decision variables with box bounds
    @variable(model_nlp2, x2)
    @variable(model_nlp2, y2)

	# --- YOUR CODE HERE ---
	
    # Solve
    # optimize!(model_nlp2)

    # Quick report
    println(model_nlp2)
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

# ╔═╡ 87ffc247-3769-4002-a584-c687fd813125
begin
	hidden_answers = Dict(
	    :lp    => (w = 20.0, g = 15.0, obj = 135.0),
	    :milp  => (x = [1,1,1], obj = 22.0),
	    :nlp   => (x = 1.0, y = 1.0, obj = 0.0),
	)
	safeval(model, sym) = haskey(model, sym) ? JuMP.value(model[sym]) : missing
	md""
end

# ╔═╡ 6fb672d0-5a18-4ccc-b7b3-184839c2401b
begin
    # ground truth
    ground_truth = (w = 20.0, g = 15.0, obj = 135.0)

    # student answer
 	ans = missing
    try
        ans = (
            w   = safeval(model_lp, :w),
            g   = safeval(model_lp, :g),
            obj = objective_value(model_lp),
        )
    catch           # objective_value will throw if model_lp not ready
        ans = missing
    end

    # Decide which badge to show
    if ismissing(ans)               # nothing yet
        still_missing()
    elseif isapprox(ans.w,   ground_truth.w;   atol=1e-3) &&
           isapprox(ans.g,   ground_truth.g;   atol=1e-3) &&
           isapprox(ans.obj, ground_truth.obj; atol=1e-3)
        correct()
    else
        keep_working()
    end
end

# ╔═╡ 254b9a87-17f9-4fea-8b28-0e3873b58fe2
begin
    ground_truth_3 = (x = -7.946795, y = 60.00, obj = -78554.7682)

	ans3=missing
    try
        ans3 = (
            x   = safeval(model_nlp, :x),
            y   = safeval(model_nlp, :y),
            obj = objective_value(model_nlp),
        )
    catch
        ans3 = missing
    end

    good_2 = !ismissing(ans3) &&
           isapprox(ans3.x,   ground_truth_3.x;   atol=1e-3) &&
           isapprox(ans3.y,   ground_truth_3.y;   atol=1e-3) &&
           isapprox(ans3.obj, ground_truth_3.obj; atol=1e-3)

    if ismissing(ans3)
        still_missing()
    elseif good_2
        correct()
    else
        keep_working()
    end
end

# ╔═╡ 1fcaedb8-34d0-4faf-9052-fc074d2edda3
begin
    ground_truth_32 = (x = -3.121657, y = 2.875823, obj = -26515.3545)

	ans4=missing
    try
        ans4 = (
            x   = safeval(model_nlp2, :x2),
            y   = safeval(model_nlp2, :y2),
            obj = objective_value(model_nlp2),
        )
    catch
        ans4 = missing
    end

    good_3 = !ismissing(ans4) &&
           isapprox(ans4.x,   ground_truth_32.x;   atol=1e-3) &&
           isapprox(ans4.y,   ground_truth_32.y;   atol=1e-3) &&
           isapprox(ans4.obj, ground_truth_32.obj; atol=1e-3)

    if ismissing(ans4)
        still_missing()
    elseif good_3
        correct()
    else
        keep_working()
    end
end

# ╔═╡ Cell order:
# ╟─881eed45-e7f0-4785-bde8-530e378d7050
# ╟─9f5675a3-07df-4fb1-b683-4c5fd2a85002
# ╟─0df8b65a-0527-4545-bf11-00e9912bced0
# ╠═9ce52307-bc22-4f66-a4af-a4e4ac382212
# ╟─6f67ca7c-1391-4cb9-b692-cd818e037587
# ╠═49042d6c-cf78-46d3-bfee-a8fd7ddf3aa0
# ╟─1d3edbdd-7747-4651-b650-c6b9bf87b460
# ╟─6fb672d0-5a18-4ccc-b7b3-184839c2401b
# ╟─248b398a-0cf5-4c2b-8752-7b9cc4e765d6
# ╟─808c505d-e10d-42e3-9fb1-9c6f384b2c3c
# ╠═39617561-bbbf-4ef6-91e2-358dfe76581c
# ╟─01367096-3971-4e79-ace2-83600672fbde
# ╟─38b3a8f3-35ae-46da-91ce-0e4ba27ae098
# ╟─bca712e4-3f1c-467e-9209-e535aed5ab0a
# ╟─3997d993-0a31-435e-86cd-50242746c305
# ╠═3f56ec63-1fa6-403c-8d2a-1990382b97ae
# ╟─0e8ed625-df85-4bd2-8b16-b475a72df566
# ╟─5e3444d0-8333-4f51-9146-d3d9625fe2e9
# ╠═0e190de3-da60-41e9-9da5-5a0c7fefd1d7
# ╟─cac18d70-b354-48c7-9f37-31ee0c585675
# ╠═00728de8-3c36-48c7-8520-4c9f408a7c5f
# ╟─254b9a87-17f9-4fea-8b28-0e3873b58fe2
# ╟─45541136-695a-4260-82c1-66d38ec44dcc
# ╠═b107bcd7-60ca-4f09-aa42-f8335e13233e
# ╟─1fcaedb8-34d0-4faf-9052-fc074d2edda3
# ╟─147fe732-fe65-4226-af43-956b33a75bff
# ╟─87ffc247-3769-4002-a584-c687fd813125
