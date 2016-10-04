using SimpleGradientDescent
using Base.Test
using NLopt

@testset "nlopt solver" begin
    num_vars = 10

    # Let's define a very simple cost function:
    xstar = [1., 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cost(x) = sum((x .- xstar) .^ 4)

    # getmodel() returns a MathProgBase model which will allow us to perform
    # the gradient descent. We can specify a particular solver if we want to,
    # or it will try to use MathProgBase's default NLP solver if we don't.
    model = getmodel(cost, num_vars, NLoptSolver(algorithm=:LD_MMA))

    # Set the initial point from which the solver will start:
    setwarmstart!(model, [0.0 for i in 1:num_vars])

    # Optimize and extract the solution:
    # (note that setwarmstart!, optimize!, status, and getsolution are all
    # just re-exported from MathProgBase for convenience)
    optimize!(model)
    @test status(model) == :Optimal
    x = getsolution(model)
    @show x xstar
    @test all(isapprox(x, xstar, atol=1e-3))

    # We can specify lower and upper bounds on the decision variables:
    model = getmodel(cost, num_vars, NLoptSolver(algorithm=:LD_MMA),
                    [-1.0 for i in 1:num_vars],
                    [1.0 for i in 1:num_vars])
    setwarmstart!(model, [0.0 for i in 1:num_vars])
    optimize!(model)
    x = getsolution(model)
    @show x xstar
    # Since we've constrained x to lie in [0, 1], it won't reach xstar
    @test all(isapprox(x, [1.0 for i in 1:num_vars], atol=1e-3))
end

@testset "naive solver" begin
    num_vars = 10

    # Let's define a very simple cost function:
    xstar = [1., 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cost(x) = sum((x .- xstar) .^ 4)

    # getmodel() returns a MathProgBase model which will allow us to perform
    # the gradient descent. We can specify a particular solver if we want to,
    # or it will try to use MathProgBase's default NLP solver if we don't.
    model = getmodel(cost, num_vars, NaiveSolver(num_vars))

    # Set the initial point from which the solver will start:
    setwarmstart!(model, [0.0 for i in 1:num_vars])

    # Optimize and extract the solution:
    # (note that setwarmstart!, optimize!, status, and getsolution are all
    # just re-exported from MathProgBase for convenience)
    optimize!(model)
    @test status(model) == :Optimal
    x = getsolution(model)
    @show x xstar
    @test all(isapprox(x, xstar, atol=1e-3))

    # We can specify lower and upper bounds on the decision variables:
    model = getmodel(cost, num_vars, NaiveSolver(num_vars),
                    [-1.0 for i in 1:num_vars],
                    [1.0 for i in 1:num_vars])
    setwarmstart!(model, [0.0 for i in 1:num_vars])
    optimize!(model)
    x = getsolution(model)
    @show x xstar
    # Since we've constrained x to lie in [0, 1], it won't reach xstar
    @test all(isapprox(x, [1.0 for i in 1:num_vars], atol=1e-3))
end

@testset "naive preconditioned" begin
    num_vars = 10
    xstar = [1., 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cost(x) = sum((x .- xstar) .^ 4)
    unconditioned_model = getmodel(cost, num_vars, NaiveSolver(num_vars; precondition_divisors=ones(num_vars),
        iteration_limit=2))
    setwarmstart!(unconditioned_model, [0.0 for i in 1:num_vars])
    optimize!(unconditioned_model)
    @test status(unconditioned_model) == :UserLimit

    conditioned_model = getmodel(cost, num_vars, NaiveSolver(num_vars; precondition_divisors=collect(1:num_vars),
        iteration_limit=2))
    setwarmstart!(conditioned_model, [0.0 for i in 1:num_vars])
    optimize!(conditioned_model)
    @test status(conditioned_model) == :Optimal
    x = getsolution(conditioned_model)
    @test all(isapprox(x, xstar, atol=1e-3))

end
