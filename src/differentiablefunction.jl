immutable DifferentiableFunction{F <: Function, G <: Function} <: MathProgBase.AbstractNLPEvaluator
    func::F
    gradient!::G
end

DifferentiableFunction(f::Function) = DifferentiableFunction(f, (result, x) -> ForwardDiff.gradient!(result, f, x))

function MathProgBase.initialize(gd::DifferentiableFunction, requested_features::Vector{Symbol})
end

MathProgBase.features_available(gd::DifferentiableFunction) = [:Grad]

MathProgBase.jac_structure(gd::DifferentiableFunction) = (tuple(), tuple())

MathProgBase.eval_grad_f(gd::DifferentiableFunction, g, x) = gd.gradient!(g, x)


MathProgBase.eval_f(gd::DifferentiableFunction, x) = gd.func(x)

function getmodel(problem::DifferentiableFunction,
                num_vars::Integer,
                solver=MathProgBase.defaultNLPsolver,
                lower_bound=[-Inf for i in 1:num_vars],
                upper_bound=[Inf for i in 1:num_vars])
    model = MathProgBase.NonlinearModel(solver)
    MathProgBase.loadproblem!(model, num_vars, 0, lower_bound, upper_bound, Float64[], Float64[], :Min, problem)
    model
end

getmodel(f::Function, args...) = getmodel(DifferentiableFunction(f), args...)
