module AMPLMathProgInterface

using AmplNLReader

import MathProgBase
import MathProgBase.SolverInterface

type AmplNLPEvaluator <: SolverInterface.AbstractNLPEvaluator
  nlp::AmplModel
end


# ASL has methods to speed up initialization if certain functionalities aren't
# needed. Currently we always request everything.
MathProgBase.initialize(::AmplNLPEvaluator, requested_features) = nothing
MathProgBase.features_available(::AmplNLPEvaluator) = [:Grad, :Jac, :HessVec, :Hess]

MathProgBase.eval_f(d::AmplNLPEvaluator, x) = obj(d.nlp, x)

MathProgBase.eval_g(d::AmplNLPEvaluator, g, x) = copy!(g, cons(d.nlp, x))

MathProgBase.eval_grad_f(d::AmplNLPEvaluator, g, x) = copy!(g, grad(d.nlp, x))

function MathProgBase.jac_structure(d::AmplNLPEvaluator)
  rows, cols, vals = jac_coord(d.nlp, d.nlp.meta.x0)
  return rows, cols
end

function MathProgBase.hesslag_structure(d::AmplNLPEvaluator)
  rows, cols, vals = hess_coord(d.nlp, d.nlp.meta.x0, y=ones(d.nlp.meta.ncon))
  return rows, cols
end


function MathProgBase.eval_jac_g(d::AmplNLPEvaluator, J, x)
  rows, cols, vals = jac_coord(d.nlp, x)
  copy!(J, vals)
end

# Are there specialized methods for Jac-vec products?
# MathProgBase.eval_jac_prod(d::AmplNLPEvaluator, J, x)
# MathProgBase.eval_jac_prod_t(d::AmplNLPEvaluator, J, x)


function MathProgBase.eval_hesslag_prod(d::AmplNLPEvaluator, h, x, v, σ, μ)
  obj(d.nlp, x) # force hessian evaluation at this point
  result = hprod(d.nlp, x, v, y = -μ, obj_weight = σ)
  copy!(h, result)
end

function MathProgBase.eval_hesslag(d::AmplNLPEvaluator, H, x, σ, μ)
  obj(d.nlp, x) # force hessian evaluation at this point
  rows, cols, vals = hess_coord(d.nlp, x, y = -μ, obj_weight = σ)
  copy!(H, vals)
end

# How do we extract this?
MathProgBase.isobjlinear(d::AmplNLPEvaluator) = false
#MathProgBase.isobjquadratic(d::AmplNLPEvaluator)

MathProgBase.isconstrlinear(d::AmplNLPEvaluator,i::Int) = (i in d.nlp.meta.lin)

function loadamplproblem!(m::MathProgBase.AbstractMathProgModel, nlp::AmplModel)
  sense = nlp.meta.minimize ? :Min : :Max
  MathProgBase.loadnonlinearproblem!(m, nlp.meta.nvar, nlp.meta.ncon, nlp.meta.lvar,
    nlp.meta.uvar, nlp.meta.lcon, nlp.meta.ucon, sense, AmplNLPEvaluator(nlp))
  MathProgBase.setwarmstart!(m, nlp.meta.x0)
end

end # module
