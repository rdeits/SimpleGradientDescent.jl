__precompile__()

module SimpleGradientDescent

import MathProgBase
import MathProgBase: setwarmstart!, optimize!, status, getsolution
import ForwardDiff

export GradientDescent,
       getmodel,
       setwarmstart!,
       optimize!,
       getsolution,
       status

immutable GradientDescent{F <: Function} <: MathProgBase.AbstractNLPEvaluator
    func::F
end

function MathProgBase.initialize(gd::GradientDescent, requested_features::Vector{Symbol})
end

MathProgBase.features_available(gd::GradientDescent) = [:Grad]

MathProgBase.jac_structure(gd::GradientDescent) = (tuple(), tuple())

MathProgBase.eval_grad_f(gd::GradientDescent, g, x) = ForwardDiff.gradient!(g, gd.func, x)

MathProgBase.eval_f(gd::GradientDescent, x) = gd.func(x)

function getmodel(problem::GradientDescent,
                num_vars::Integer,
                solver=MathProgBase.defaultNLPsolver,
                lower_bound=[-Inf for i in 1:num_vars],
                upper_bound=[Inf for i in 1:num_vars])
    model = MathProgBase.NonlinearModel(solver)
    MathProgBase.loadproblem!(model, num_vars, 0, lower_bound, upper_bound, Float64[], Float64[], :Min, problem)
    model
end

getmodel(f::Function, args...) = getmodel(GradientDescent(f), args...)

end # module
