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

  # AMPL obfuscation ordering:
  nnlvar = max(nlp.meta.nlvc, nlp.meta.nlvo)
  narcvar = nlp.meta.nwv
  nlinvar = nlp.meta.nvar - (nnlvar + narcvar + nlp.meta.nbv + nlp.meta.niv)
  nbinvar = nlp.meta.nbv
  nintvar = nlp.meta.niv

  v = fill(:Cont, nlp.meta.nvar)
  # First populate Table 4
  varidx = 1
  for i = 1:(nlp.meta.nlvb - nlp.meta.nlvbi)
    varidx += 1
  end
  for i = 1:nlp.meta.nlvbi
    v[varidx] = :Int
    varidx += 1
  end
  for i = 1:(nlp.meta.nlvc - (nlp.meta.nlvb + nlp.meta.nlvci))
    varidx += 1
  end
  for i = 1:nlp.meta.nlvci
    v[varidx] = :Int
    varidx += 1
  end
  for i = 1:(nlp.meta.nlvo - (nlp.meta.nlvc + nlp.meta.nlvoi))
    varidx += 1
  end
  for i = 1:nlp.meta.nlvoi
    v[varidx] = :Int
    varidx += 1
  end
  # Now populate Table 3
  varidx += narcvar + nlinvar
  for i = 1:nbinvar
    v[varidx] = :Bin
    varidx += 1
  end
  for i = 1:nintvar
    v[varidx] = :Int
    varidx += 1
  end
  @assert varidx == nlp.meta.nvar + 1
  # Set variable types
  if any(vtype -> vtype == :Int || vtype == :Bin, v)
      MathProgBase.setvartype!(m, v)
  end
end

end # module
