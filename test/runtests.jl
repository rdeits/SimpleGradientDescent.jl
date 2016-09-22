using SimpleGradientDescent
using Base.Test
using Ipopt

let
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

    # We can also specify a particular solver if we want to:
    model = getmodel(cost, num_vars, IpoptSolver())

    # And we can specify lower and upper bounds on the decision variables:
    model = getmodel(cost, num_vars, IpoptSolver(),
                    [-1.0 for i in 1:num_vars],
                    [1.0 for i in 1:num_vars])
    setwarmstart!(model, [0.0 for i in 1:num_vars])
    optimize!(model)
    x = getsolution(model)
    @show x xstar
    # Since we've constrained x to lie in [0, 1], it won't reach xstar
    @test all(isapprox.(x, [1.0], atol=1e-2))
end
