include("unified_module.jl")
include("unified_eigs.jl")
using Parameters
using LinearAlgebra
using ForwardDiff
using DifferentialEquations
using NLsolve
using StatsBase
using Statistics 
using Distributed


## omnivory & coupling data (Structural EWS)
function structural_data(p)
    @unpack h_PC_litt, h_PC_pel, h_PR_litt, h_PR_pel, e_PC_litt, e_PC_pel, e_PR_litt, e_PR_pel, a_PR_litt, a_PR_pel, a_PC_litt,  a_PC_pel = p 
    Nvals = 1.0:0.005:2.165
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(9000, 10000, length = 1000)
    omnivory = zeros(length(Nvals))
    coupling = zeros(length(Nvals))
    tp = zeros(length(Nvals))

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
        tp[Ni] = 2 + (ClittFR + CpelFR)/ (RlittFR + RpelFR + ClittFR + CpelFR)

    end
    return hcat(collect(Nvals), omnivory, coupling, tp)
end



function dynamical_data()
    Nvals = 1.0:0.005:2.165
    u0 = [0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(0.0, 10000.0, length = 10000)
    mean_vals = zeros(length(Nvals)) 
    sd_vals = zeros(length(Nvals))
    cv_vals = zeros(length(Nvals))

    for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        mean_vals[Ni]=mean(grid[5,5000:10000])
        sd_vals[Ni]=std(grid[5,5000:10000])
       
    end
    return hcat(collect(Nvals), mean_vals, sd_vals)
end

function local_min(ts)
    lmins = Float64[]
    for i in 2:(length(ts) - 3)
        if ts[i - 1] > ts[i] && ts[i] < ts[i + 1]
            push!(lmins, ts[i])
        end
    end
    return lmins
end

function local_max(ts)
    lmaxs = Float64[]
    for i in 2:(length(ts) - 3)
        if ts[i - 1] < ts[i] && ts[i] > ts[i + 1]
            push!(lmaxs, ts[i])
        end
    end
    return lmaxs
end

function minmax_func()
    Nvals = 1.0:0.005:2.165
    u0 = [ 0.5, 0.5, 0.3, 0.3, 0.3]
    tspan = (0.0, 10000.0)
    ts = range(5000, 10000, length = 1000)
    lmin_vals= zeros(length(Nvals))
    lmax_vals = zeros(length(Nvals))


    for (Ni, Nval) in enumerate(Nvals)
        p = Par(n = Nval)
        prob = ODEProblem(model!, u0, tspan, p)
        sol = OrdinaryDiffEq.solve(prob)
        grid = sol(ts)
        
        lmin_vals[Ni] = minimum(local_min(grid[5, 1:end]))
        lmax_vals[Ni] = maximum(local_max(grid[5, 1:end]))

    end
    return hcat(collect(Nvals), lmin_vals, lmax_vals)
end


##structure vs dynamics

data1 = n_maxeig_data()
data2 = structural_data(Par())
data3 = dynamical_data()
data4 = minmax_func()
eigs = data1[:,2]
omnivory = data2[:,2]
coupling = data2[:,3]
tposition = data2[:,4]
meanden = data3[:,2]
sd = data3[:,3]
cv = sd./meanden
minden = data4[:,2]
maxden = data4[:,3]

using Plots

p1 = Plots.plot(tposition, cv, legend = false,  xlabel = "TP", ylabel = "CV", lw = 2.0, colour = "black")
p2 = Plots.plot(tposition, meanden, legend = false,  xlabel = "TP", ylabel = "Mean P Density", lw = 2.0, colour = "black")
p3 = Plots.plot(tposition, eigs, legend = false,  xlabel = "TP", ylabel = "Max Re(λ)", lw = 2.0, colour = "black")
p4 = Plots.plot(tposition, minden, legend = false,  xlabel = "TP", ylabel = "Min P Density", lw = 2.0, colour = "black")
p5 = Plots.plot(coupling, cv, legend = false,  xlabel = "% Fast Chain Derived Carbon", ylabel = "CV", lw = 2.0, colour = "black")
p6 = Plots.plot(coupling, meanden, legend = false,  xlabel = "% Fast Chain Derived Carbon", ylabel = "Mean P Density", lw = 2.0, colour = "black")
p7 = Plots.plot(coupling, eigs, legend = false,  xlabel = "% Fast Chain Derived Carbon", ylabel = "Max Re(λ)", lw = 2.0, colour = "black")
p8 = Plots.plot(coupling, minden, legend = false,  xlabel = "% Fast Chain Derived Carbon", ylabel = "Min P Density", lw = 2.0, colour = "black")

Plots.plot(p1, p2, p3, p4, layout = (2, 2), legend = false)
Plots.plot(p5, p6, p7, p8, layout = (2, 2), legend = false)