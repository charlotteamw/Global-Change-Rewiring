include("unified_module.jl")
using Parameters
using LinearAlgebra
using ForwardDiff
using PyPlot
using DifferentialEquations
using NLsolve


## omnivory & coupling data (Structural EWS)
function structural_data(p)
    @unpack h_PC_litt, h_PC_pel, h_PR_litt, h_PR_pel, e_PC_litt, e_PC_pel, e_PR_litt, e_PR_pel, a_PR_litt, a_PR_pel, a_PC_litt,  a_PC_pel = p 
    Nvals = 0.73:0.005:2.16
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
    omnivory = zeros(length(Nvals))
    coupling = zeros(length(Nvals))


    for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        eq = nlsolve((du, u) -> model!(du, u, p, 0.0), grid.u[end]).zero  
        RlittFR = ((e_PR_litt * a_PR_litt * eq[1] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        RpelFR = ((e_PR_pel * a_PR_pel * eq[2] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        ClittFR = ((e_PC_litt * a_PC_litt * eq[3] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        CpelFR = ((e_PC_pel * a_PC_pel * eq[4] * eq[5])/(1 + a_PR_litt * h_PR_litt * eq[1] + a_PR_pel * h_PR_pel * eq[2] + a_PC_litt * h_PC_litt * eq[3] + a_PC_pel * h_PC_pel * eq[4])) 
        omnivory[Ni] = (RlittFR + RpelFR)/ (RlittFR + RpelFR + ClittFR + CpelFR)
        coupling[Ni] = (RlittFR + ClittFR)/ (RlittFR + RpelFR + ClittFR + CpelFR)
    end
    return hcat(collect(Nvals), omnivory, coupling)
end

let
    data = structural_data(Par())
    structural_plot = figure()
    plot(data[:,1], data[:,2], color = "blue")
    plot(data[:,1], data[:,3], color = "red")
    ylabel("Structural Change", fontsize = 15)
    xlim(0.73, 2.16)
    ylim(0.0, 1.0)
    xlabel("Global Change", fontsize = 15)
    return structural_plot
end
