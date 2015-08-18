# Example script to solve AMPL model with IPOPT.
# Needs some restructuring to be callable from AMPL

using Ipopt
import MathProgBase
import AmplNLReader
import AMPLMathProgInterface

function solve_with_ipopt(nlfile::ASCIIString)
  nlp = AmplNLReader.AmplModel(nlfile)
  # options can be set here
  m = MathProgBase.model(IpoptSolver())
  AMPLMathProgInterface.loadamplproblem!(m, nlp)
  MathProgBase.optimize!(m)

  objval = MathProgBase.getobjval(m)
  x = MathProgBase.getsolution(m)

  println("Optimal value: $objval")
  println("Solution: $x")
  
end

if length(ARGS) != 1
  error("Usage: julia ampl_ipopt.jl nlfile")
end

solve_with_ipopt(ascii(ARGS[1]))
