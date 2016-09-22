# SimpleGradientDescent

[![Build Status](https://travis-ci.org/rdeits/SimpleGradientDescent.jl.svg?branch=master)](https://travis-ci.org/rdeits/SimpleGradientDescent.jl)

[![Coverage Status](https://coveralls.io/repos/rdeits/SimpleGradientDescent.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/rdeits/SimpleGradientDescent.jl?branch=master)

[![codecov.io](http://codecov.io/github/rdeits/SimpleGradientDescent.jl/coverage.svg?branch=master)](http://codecov.io/github/rdeits/SimpleGradientDescent.jl?branch=master)

This package provides some helpful boilerplate code to convert general unconstrained gradient descent of a cost function into a [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) nonlinear model. This model can then be solved by any solver which supports the MathProgBase interface, such as [Ipopt](https://github.com/JuliaOpt/Ipopt.jl) and [NLopt](https://github.com/JuliaOpt/NLopt.jl).

# Usage

See `test/runtests.jl` for up-to-date examples.

```julia
num_vars = 10

# Let's define a very simple cost function:
xstar = convert(Vector{Float64}, 1:num_vars)
cost(x) = sum((x .- xstar) .^ 4)

# getmodel() returns a MathProgBase model which will allow us to perform
# the gradient descent. It will use MathProgBase's default NLP solver by
# default:
model = getmodel(cost, num_vars)

# Set the initial point from which the solver will start:
setwarmstart!(model, [0.0 for i in 1:num_vars])

# Optimize and extract the solution:
# (note that setwarmstart!, optimize!, status, and getsolution are all
# just re-exported from MathProgBase for convenience)
optimize!(model)
@test status(model) == :Optimal
x = getsolution(model)
@show x xstar
@test all(isapprox.(x, xstar, atol=1e-2))
```
