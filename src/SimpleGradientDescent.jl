__precompile__()

module SimpleGradientDescent

import MathProgBase
import MathProgBase: setwarmstart!, optimize!, status, getsolution
import ForwardDiff
import StaticArrays: SVector, MVector

export DifferentiableFunction,
       NaiveSolver,
       getmodel,
       setwarmstart!,
       optimize!,
       getsolution,
       status

include("differentiablefunction.jl")
include("solver.jl")

end # module
