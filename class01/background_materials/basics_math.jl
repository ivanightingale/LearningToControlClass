### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 9fffca86-aec9-4c26-bced-3edaa1af9a22
begin
using Pkg; Pkg.activate("..")
end

# ╔═╡ 00cc1e5e-f10e-4559-b9d9-8aba44eb493e
begin
	using PlutoUI
	using Random
	using LinearAlgebra
	using HypertextLiteral
	using PlutoTeachingTools
	using ShortCodes, MarkdownLiteral
	using Random
	Random.seed!(8803)
end

# ╔═╡ a6bd5fd0-f856-4f1e-8e06-4e16639d436c
using Symbolics

# ╔═╡ 44c9e09f-4fb6-42b2-a265-3adedfa9b2d2
using Zygote

# ╔═╡ b129ba7c-953a-11ea-3379-17adae34924c
md"

### Math Foundations

| | | |
|-----------:|:--|:------------------|
|  Lecturer   | : | Rosemberg, Andrew |

# Background Math (_Welcome to Pluto!_)

This background material will use Pluto!
Pluto is a programming environment for _Julia_, designed to be **interactive** and **helpful**.

"

# ╔═╡ 22c65fc5-08f9-4675-89c5-20ff685c35b3
md"We should be active on the environment `LearningToControlClass/class01`"

# ╔═╡ a481b9a8-306a-4434-b5d0-e8fc7bba8482
md"Let's import some useful packages"

# ╔═╡ 4d88b926-9543-11ea-293a-1379b1b5ae64
begin
md"## 1) Calculus and linear algebra"
end

# ╔═╡ c9941b84-575c-4b98-94d2-72b59327a6a8
md" ### a) Partial Derivative

For a vector $w \in R^N$ and another vector $a \in R^N$, consider the function:

```math
f(w) = a^T w
```

"

# ╔═╡ ac439448-a766-4cb2-b76f-b4733cd14195
question_box(md"What is the partial derivative of $f$ with respect to $w_n$ ?")

# ╔═╡ 23ff3286-be75-42b8-8327-46ebb7d8b538
md"Write you answer bellow in place of `missing` as a function of `a` and `n`"

# ╔═╡ a33c9e20-0139-4e72-a6a1-d7e9641195cb
ans=missing

# ╔═╡ c9afb329-2aa8-48e2-8238-8f6f3977b988
begin
N = 10
n = 5
a = rand(N)
println("a=", a)
end

# ╔═╡ 529c2464-4611-405b-8ff7-b9de8158aa2d
md" **Lets check our answer (with Symbolic Differentiation):**"

# ╔═╡ ebd908a2-98c5-4abb-84cf-9a286779a736
begin
	@variables w[1:N]

	∂╱∂w = Differential(w[n])

	f(t) = dot(a, t)
	
	_ans = expand_derivatives(∂╱∂w(f(w)))
end

# ╔═╡ 570f87c7-f808-442e-b22c-d1aaec61922d
begin
	if ismissing(ans)
		still_missing()
	elseif _ans == a[n]
		correct()
	else
		keep_working()
	end
end

# ╔═╡ aeb3a6bc-9540-11ea-0b8f-6d37412bfe68
md"### b) Gradient

Compute the gradient of the function:

```math

f(w_1, w_2, w_3) = w_1 w_2 + w_2 w_3 + w_3 w_1

```

"

# ╔═╡ 611c28fa-9542-11ea-1751-fbdedcfb7690
ans2=missing

# ╔═╡ da942073-32d5-4de5-82c0-271a0cb0e903
md" **Lets check our answer (with Automatic Differentiation - AD):**

Symbolic Differentiation may not always be possible, making it beneficial to have alternative Automatic Differentiation (AD) tools at your disposal.
"

# ╔═╡ 9972495f-79f4-4612-99db-d97c3e9d57b4
zygote_ans = f'(rand(N))

# ╔═╡ e6948424-ce04-4bed-b9d1-ab6c5a513ffa
begin
	if ismissing(ans2)
		still_missing()
	elseif zygote_ans == ans2
		correct()
	else
		keep_working()
	end
end

# ╔═╡ 6f7eecec-9543-11ea-1284-dd52fce3ecca
md"
### c) Equivalence

Suppose we have the linear equation
```math

y = A v

```

"

# ╔═╡ 397ffb27-b0c4-408a-8c89-b022e6924ee0
question_box(md"Argue that $y$ can be written as a linear combination of the columns of $A$.")

# ╔═╡ f1346c3e-727a-488d-bfc8-54ffacd26987
md"### d) Vector Squared"

# ╔═╡ e0642f42-9545-11ea-14ee-fde52cb54ccc
question_box(md" Given any vector $v \in R^N$ with real entries, show that $v^T v = 0 \iff v = 0$")

# ╔═╡ 19ff8d36-9547-11ea-0e08-e5cdd8338673
md"
### e) Matrix Squared

Suppose a matrix $A = B^T B$ where matrix $B_{m,n}$ is not necessarily square ($m \neq n$).
"

# ╔═╡ ee268cb3-c7b4-4696-a8d2-54efe9ab56a7
question_box(md"Is that $A$ is square, symmetric, and PSD? Argue")

# ╔═╡ 9967004a-cd54-4b57-a813-c70634dece88
md" **Check your counter example:**
"

# ╔═╡ b8700fc7-3e81-44ac-abf2-d62bf96dcece
let
	_B = rand(5, 8) # Change here
	_A =_B'_B
	
	size(_A, 1) == size(_A,2) && # Square
	issymmetric(_A) && # Symmetric
	all(trunc.(eigen(_A).values, digits=3) .>= 0.0) && # PSD
	println("Still square, symmetric, and PSD")
end

# ╔═╡ 644e3970-4b24-4392-822b-0d11e5cbeb89
md"

#### f) Eigenvalues

Find the eigenvalues and vectors of

```math
A = 
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}

```

Workout by hand!

Please define your eigenvalues in 😀 and your eigenvectors in 🥤

"

# ╔═╡ 0cd18898-359f-4d27-a4b9-63dc0b416981
md" **Lets check our answer:**"

# ╔═╡ eac62fea-954e-11ea-2768-39ce6f4059ab
begin

# Our Eigenvalues 
😀 = missing

# Our Eigenvectors

🥤= missing

end

# ╔═╡ 01bc0cf1-8759-49c1-b69c-edc690e23f8c
begin
	if ismissing(😀) || ismissing(🥤)
		still_missing()
	else
		checked_asnwer = eigen([ 3 1; 1 3])
		if issetequal(checked_asnwer.values, 😀) && issetequal(eachcol(round.(checked_asnwer.vectors; digits=2)), eachcol(round.(🥤; digits=2)))
			correct()
		else
			keep_working(md"😔")
		end
	end
end

# ╔═╡ f27f90c2-954f-11ea-3f93-17acb2ce4280
md"
#### f) Invertible matrix Eigenvalues

Let $A$ be an invertible matrix with eigenvectors $V$ and eigenvalues $\Lambda$
"

# ╔═╡ 0914879e-af78-4e1a-baf6-4f8fcad4d5ce
question_box(md"What are the eigenvectors (V₋₁) and eigenvalues (Λ₋₁) of $A^{−1}$?
			 
Write them as a function of $V$ and $\Lambda$")

# ╔═╡ 4eaf1b5e-7114-4b1a-bd13-6991f0c596bb
begin
	Random.seed!(8803)
	A = randn(n, n) + I * 10
	U, S, V = svd(A)
	S_new = Diagonal(max.(S, 1e-6)) # Ensure all singular values are positive
	A = U * S_new * V'
	@info A
end

# ╔═╡ 17a2fa3a-4dfc-45df-955a-068bbd1c225c
begin
	# Add your answer here
	V₋₁ = missing
	Λ₋₁ = missing
end

# ╔═╡ 1fec2e0a-c108-43a1-a4f3-557af9e215ab
let
	global V₋₁, Λ₋₁, A
	if ismissing(V₋₁) || ismissing(Λ₋₁)
		still_missing()
	else
		checked_asnwer = eigen(inv(A))
		if issetequal(checked_asnwer.values, Λ₋₁) && issetequal(eachcol(round.(checked_asnwer.vectors; digits=2)), eachcol(round.(V₋₁; digits=2)))
			correct()
		else
			keep_working(md"😔")
		end
	end
end

# ╔═╡ 4634c856-9553-11ea-008d-3539195970ea
md"## Final notes"

# ╔═╡ 4d0ebb46-9553-11ea-3431-2d203f594815
md"If anything about this introduction left you confused, something doesn't work, or you have a cool new idea -- don't hesitate to contact me!"

# ╔═╡ d736e096-9553-11ea-3ba5-277afde1afe7
html"""<p>We're almost done! Scroll to the top of the notebook, and click on <img src="https://cdn.jsdelivr.net/gh/ionic-team/ionicons@5.5.1/src/svg/share-outline.svg" style="width: 1em; height: 1em; margin-bottom: -.2em;"> to see the export options - or you can always share this notebook's save file. (The file is pure Julia, by the way, and it's runnable! and git friendly)</p>"""

# ╔═╡ 1deaaf36-9554-11ea-3dae-85851f73dbc6
md"
Check out the amzing packages we used here:

 - [Pluto.jl](https://github.com/fonsp/Pluto.jl) : Iteractive and light notebooks;
 - [Distributions.jl](https://github.com/JuliaStats/Distributions.jl): Easy handling of many distributions with nice type system;
 - [Plots.jl](https://github.com/JuliaPlots/Plots.jl): Julia base Plotting package that accepts many backagens (even from other programming languages);
 - [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl) : Symmbolic Algebra;
 - [Zygote.jl](https://github.com/FluxML/Zygote.jl): AD

And others that are useful for Machine Learning:
 - [JuMP.jl](https://github.com/jump-dev/JuMP.jl): Amazing Optimization package that interfaces with most solvers;
 - [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl): A Machine Learning Framework for Julia.
"

# ╔═╡ Cell order:
# ╟─b129ba7c-953a-11ea-3379-17adae34924c
# ╟─22c65fc5-08f9-4675-89c5-20ff685c35b3
# ╟─9fffca86-aec9-4c26-bced-3edaa1af9a22
# ╟─a481b9a8-306a-4434-b5d0-e8fc7bba8482
# ╠═00cc1e5e-f10e-4559-b9d9-8aba44eb493e
# ╟─4d88b926-9543-11ea-293a-1379b1b5ae64
# ╟─c9941b84-575c-4b98-94d2-72b59327a6a8
# ╟─ac439448-a766-4cb2-b76f-b4733cd14195
# ╟─23ff3286-be75-42b8-8327-46ebb7d8b538
# ╠═a33c9e20-0139-4e72-a6a1-d7e9641195cb
# ╟─c9afb329-2aa8-48e2-8238-8f6f3977b988
# ╟─529c2464-4611-405b-8ff7-b9de8158aa2d
# ╠═a6bd5fd0-f856-4f1e-8e06-4e16639d436c
# ╠═ebd908a2-98c5-4abb-84cf-9a286779a736
# ╟─570f87c7-f808-442e-b22c-d1aaec61922d
# ╟─aeb3a6bc-9540-11ea-0b8f-6d37412bfe68
# ╠═611c28fa-9542-11ea-1751-fbdedcfb7690
# ╟─da942073-32d5-4de5-82c0-271a0cb0e903
# ╠═44c9e09f-4fb6-42b2-a265-3adedfa9b2d2
# ╠═9972495f-79f4-4612-99db-d97c3e9d57b4
# ╟─e6948424-ce04-4bed-b9d1-ab6c5a513ffa
# ╟─6f7eecec-9543-11ea-1284-dd52fce3ecca
# ╟─397ffb27-b0c4-408a-8c89-b022e6924ee0
# ╟─f1346c3e-727a-488d-bfc8-54ffacd26987
# ╟─e0642f42-9545-11ea-14ee-fde52cb54ccc
# ╟─19ff8d36-9547-11ea-0e08-e5cdd8338673
# ╟─ee268cb3-c7b4-4696-a8d2-54efe9ab56a7
# ╟─9967004a-cd54-4b57-a813-c70634dece88
# ╠═b8700fc7-3e81-44ac-abf2-d62bf96dcece
# ╟─644e3970-4b24-4392-822b-0d11e5cbeb89
# ╟─0cd18898-359f-4d27-a4b9-63dc0b416981
# ╠═eac62fea-954e-11ea-2768-39ce6f4059ab
# ╟─01bc0cf1-8759-49c1-b69c-edc690e23f8c
# ╟─f27f90c2-954f-11ea-3f93-17acb2ce4280
# ╟─0914879e-af78-4e1a-baf6-4f8fcad4d5ce
# ╠═4eaf1b5e-7114-4b1a-bd13-6991f0c596bb
# ╠═17a2fa3a-4dfc-45df-955a-068bbd1c225c
# ╟─1fec2e0a-c108-43a1-a4f3-557af9e215ab
# ╟─4634c856-9553-11ea-008d-3539195970ea
# ╟─4d0ebb46-9553-11ea-3431-2d203f594815
# ╟─d736e096-9553-11ea-3ba5-277afde1afe7
# ╟─1deaaf36-9554-11ea-3dae-85851f73dbc6
