include("unified_module.jl")
using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve


## Eigenvalue analysis 
## using ForwardDiff for eigenvalue analysis, need to reassign model for just u
function model(u, Par, t)
    du = similar(u)
    model!(du, u, Par, t)
    return du
end

## calculating the jacobian 
function jac(u, model, p)
    ForwardDiff.jacobian(u -> model(u, p, NaN), u)
end

## getting max real eigenvalue 
λ_stability(M) = maximum(real.(eigvals(M)))

function n_maxeig_data()
    Nvals = 0.185:0.005:1.66
    max_eig = zeros(length(Nvals))
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
  

   for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        equilibrium = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero     
        generalist_jac = jac(equilibrium, model, p)
        max_eig[Ni] = λ_stability(generalist_jac)
    end
 return hcat(collect(Nvals), max_eig)
end

let
    data = n_maxeig_data()
    maxeigen_plot = figure()
    plot(data[:,1], data[:,2], color = "black")
    hlines(0.0, 0.0, 3.0, colors="grey")
    ylabel("Re(λₘₐₓ)", fontsize = 14, fontweight=:bold)
    xlabel("Nutrient Concentration", fontsize = 14, fontweight=:bold)
    ylim(-0.05, 0.025)
    xlim(0.185, 1.66)
    return maxeigen_plot
end
