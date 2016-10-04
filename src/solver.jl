immutable NaiveSolver{N, T} <: MathProgBase.AbstractMathProgSolver
    rate::T
    max_step::T
    precondition_divisors::SVector{N, T}
    iteration_limit::Int
    gradient_convergence_tolerance::T
end

NaiveSolver(num_vars;
    rate=1.0,
    max_step=1.0,
    precondition_divisors::AbstractVector=ones(SVector{num_vars, Float64}),
    iteration_limit=100,
    gradient_convergence_tolerance=1e-3) =
NaiveSolver{num_vars, Float64}(rate,
    max_step,
    convert(SVector{num_vars, Float64}, precondition_divisors),
    iteration_limit,
    gradient_convergence_tolerance)

type NaiveModel{N, T} <: MathProgBase.AbstractNonlinearModel
    solver::NaiveSolver{N, T}
    x::MVector{N, T}
    x_lb::SVector{N, T}
    x_ub::SVector{N, T}
    g::MVector{N, T}
    objval::T
    problem::MathProgBase.AbstractNLPEvaluator
    status::Symbol

    NaiveModel(solver::NaiveSolver{N, T}) = new(solver)
end

NaiveModel{N, T}(solver::NaiveSolver{N, T}) = NaiveModel{N, T}(solver)

MathProgBase.NonlinearModel{N, T}(solver::NaiveSolver{N, T}) = NaiveModel{N, T}(solver)

function MathProgBase.loadproblem!{N, T}(model::NaiveModel{N, T},
    num_vars::Integer, num_constraints::Integer,
    x_lb::AbstractVector, x_ub::AbstractVector,
    g_lb::AbstractVector, g_ub::AbstractVector,
    sense::Symbol,
    problem::MathProgBase.AbstractNLPEvaluator)

    @assert sense == :Min
    @assert num_vars == N
    @assert num_constraints == 0

    model.x_lb = convert(SVector{N, T}, x_lb)
    model.x_ub = convert(SVector{N, T}, x_ub)
    model.x = zeros(MVector{N, T})
    model.g = zeros(MVector{N, T})
    model.objval = zero(T)
    model.problem = problem
end

MathProgBase.setwarmstart!(model::NaiveModel, x) = copy!(model.x, x)

function MathProgBase.optimize!{N, T}(model::NaiveModel{N, T})
    model.status = :UserLimit
    conditioned_max_step = model.solver.max_step .* model.solver.precondition_divisors
    conditioned_rate = model.solver.rate .* model.solver.precondition_divisors
    for i in 1:model.solver.iteration_limit
        MathProgBase.eval_grad_f(model.problem, model.g, model.x)

        if maximum(abs.(model.g)) < model.solver.gradient_convergence_tolerance
            model.status = :Optimal
            break
        end

        step = max.(min.(conditioned_max_step, conditioned_rate .* model.g), -conditioned_max_step)
        model.x .= max.(min.(model.x .- step, model.x_ub), model.x_lb)

    end
end

MathProgBase.status(model::NaiveModel) = model.status
MathProgBase.getsolution(model::NaiveModel) = model.x
