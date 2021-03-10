module HolisticRegression

export compute_dual, compute_primal, get_t_α, get_σ_X

using Random, Distributions
using LinearAlgebra
using Gurobi, JuMP
using StatsBase
using Optim

gurobi_env = Gurobi.Env()


#---------- Utils ----------#

function get_support(s)
    supp = similar(s, Int)
    count_supp = 1
    
    supp_c = similar(s, Int)
    count_supp_c = 1
    
    @inbounds for i in eachindex(s)
        supp[count_supp] = i
        supp_c[count_supp_c] = i
        is_zero = s[i] < 0.5
        count_supp += !is_zero
        count_supp_c += is_zero
    end
    return resize!(supp, count_supp-1), resize!(supp_c, count_supp_c-1)
end

function get_t_α(n, p, α)
    return quantile(TDist(n-p), 1 - α/2)
end

function get_σ_X(X, y, γ)
    n, p = size(X)
    
    # Estimator σ
    M_inv = inv(I/γ + X'X)
    σ_tilde = sqrt((y'*(I - X*M_inv*X')*y)/(n-p))
    σ_X = σ_tilde * sqrt.(diag(M_inv))
    
    return σ_X
end

function get_R2(y_pred, y_true, y_train)
    SS_res = norm(y_true .- y_pred)
    SS_tot = norm(y_true .- mean(y_train))
    return 1 - (SS_res/SS_tot)^2
end

function create_gurobi_model(; TimeLimit=-1, LogFile=nothing)
    model = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env)));
    if TimeLimit >= 0
        println("Set Gurobi TimeLimit.")
        set_optimizer_attribute(model, "TimeLimit", TimeLimit)
    end
    if LogFile != nothing
        println("LogFile: $(LogFile).")
        set_optimizer_attribute(model, "LogFile", LogFile)
    else
        set_optimizer_attribute(model, "OutputFlag", 0)
    end
    set_optimizer_attribute(model, "NumericFocus", 3)
    return model
end;


#---------- Inner Problem ----------#

function g_s(D_s, b_s, σ_X_s; GD=true)
    
    # Get length of support of s
    l = length(b_s)
    
    # Case s == 0
    if l==0
        return zeros(0), 0.0
    end
    
    # Initial solution
    λ_s0 = zeros(l) .+ 1.0

    # Compute objective and gradient at the same time
    function fg!(F, G, λ_s)
        
        μ_s = λ_s .+ b_s
        β_s = D_s*μ_s
        
        if G != nothing
            G .= β_s .- σ_X_s
        end
        
        if F != nothing
            return -λ_s'σ_X_s + 0.5*μ_s'β_s
        end
    end
    
    # Lagrangian multipliers constraint
    lower = zeros(l)
    upper = [Inf for _ in 1:l]

    res = Optim.optimize(Optim.only_fg!(fg!), lower, upper, λ_s0, 
        Fminbox(GD ? GradientDescent() : LBFGS()))

    return Optim.minimizer(res), - Optim.minimum(res)
    
end

function ∇g_s(supp, supp_c, b, M, λ_s, D_s, σ_X_s, γ)
    
    β_s = D_s*(b[supp] .+ λ_s)
  
    grad = zeros(length(b))
    grad[supp] = λ_s .* σ_X_s - (β_s .^ 2)/(2γ)
    grad[supp_c] = - 0.5*γ*(b[supp_c] - M[supp_c, supp]*β_s).^2
    
    return grad
    
end


#---------- Outer Problem ----------#

function compute_warm_start_primal(M_p, b_p, k, γ, σ_X_p, time_limit; LogFile=nothing)
    
    p = length(b_p)
    
    model = create_gurobi_model(;TimeLimit=time_limit, LogFile=LogFile)

    # TODO: change big-M values
    M1 = 1000
    M2 = 1000

    @variable(model, β[i=1:p])
    @variable(model, s[i=1:p], Bin)
    @variable(model, b[i=1:p], Bin)

    @constraint(model, sum(s) <= k)
    
    @constraint(model, [i=1:p], β[i] <= M1*s[i])
    @constraint(model, [i=1:p], β[i] >= -M1*s[i])

    @constraint(model, [i=1:p], β[i]/σ_X_p[i] + M2*b[i] >= s[i])
    @constraint(model, [i=1:p], -β[i]/σ_X_p[i] + M2*(1-b[i]) >= s[i])
    
    @objective(model, Min, 0.5*(- 2*b_p'β + β'*M_p*β + (1/γ)*sum(β[j]^2 for j=1:p)))
    
    optimize!(model)
    
    s_val = Int.(value.(s))
    b_val = Int.(value.(b))
    
    return vcat(s_val .* (b_val .== 0), s_val .* (b_val .== 1))
end


"""
WarmStart ∈ { :None, :RidgeStart, :PrimalStart }
"""
function compute_dual(X_p, y, k, γ, σ_X_p; LogFile=nothing, WarmStart=:None, TimeLimit=-1)
    
    # Get dimensions
    n, p = size(X_p)
    
    # Constant
    C = 0.5*y'y
 
    # Compute data in p dimensions
    M_p = X_p'X_p
    b_p = X_p'y

    # Compute data in 2p dimensions
    M = [M_p -M_p; -M_p  M_p]
    b = [b_p; -b_p];
    σ_X = [σ_X_p; σ_X_p]
    
    # Outer problem
    miop = create_gurobi_model(;LogFile=LogFile, TimeLimit=TimeLimit)
    @variable(miop, s[1:2p], Bin)
    @variable(miop, t)
    
    t0 = -C
    s0 = zeros(2p)
    
    # Initial solution
    if (WarmStart == :RidgeStart)
        β_ridge = inv(I/γ + M_p)*b_p
        s0[findall(x -> x>0, β_ridge)] .= 1.0
        s0[findall(x -> x<0, β_ridge) .+ p] .= 1.0
    elseif (WarmStart == :PrimalStart)
        s0 .= compute_warm_start_primal(M_p, b_p, k, γ, σ_X_p, 20; LogFile=LogFile) # TODO: change time limit
    else 
        s0[1:k] .= 1
    end

    # Initial cut
    supp, supp_c = get_support(s0)
    D_s, b_s, σ_X_s = inv(I/γ + M[supp, supp]), b[supp], σ_X[supp]
    λ_s0, g_s0 = g_s(D_s, b_s, σ_X_s; GD=true)
    ∇g_s0 = ∇g_s(supp, supp_c, b, M, λ_s0, D_s, σ_X_s, γ)
    offset = g_s0 - sum(∇g_s0[j]*s0[j] for j=1:2p)

    
    # Cutting planes    
    function outer_approximation(cb_data)
        
        # Get feasible solution
        s_val = [callback_value(cb_data, s[i]) for i=1:2p]
        
        # Generate cut
        supp, supp_c = get_support(s_val)
        D_s, b_s, σ_X_s = inv(I/γ + M[supp, supp]), b[supp], σ_X[supp]
        λ_s_val, g_s_val = g_s(D_s, b_s, σ_X_s; GD=true)
        ∇g_s_val = ∇g_s(supp, supp_c, b, M, λ_s_val, D_s, σ_X_s, γ)
        offset = g_s_val - sum(∇g_s_val[j]*s_val[j] for j=1:2p)
        
        con = @build_constraint(t >= sum(∇g_s_val[j]*s[j] for j=1:2p) + offset)
        MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
        
    end
    
    MOI.set(miop, MOI.LazyConstraintCallback(), outer_approximation)

    @constraint(miop, t >= -C)
    @constraint(miop, sum(s) <= k)
    @constraint(miop, [i=1:p], s[i] + s[p+i] <= 1)
    
    @constraint(miop, t >= sum(∇g_s0[j]*s[j] for j=1:2p) + offset)
    
    set_start_value.(miop[:s], s0)
    set_start_value(miop[:t], g_s0)

    @objective(miop, Min, t + C)
    
    optimize!(miop)
    
    s_opt = value.(miop[:s])
    
    supp, supp_c = get_support(s_opt)
    D_s, b_s, σ_X_s = inv(I/γ + M[supp, supp]), b[supp], σ_X[supp]
    λ_s_opt, g_s_opt = g_s(D_s, b_s, σ_X_s; GD=true)
    β = zeros(2p)
    β[supp] = D_s*(λ_s_opt + b_s)
    
    return objective_value(miop), β[1:p] - β[p+1:end]
end

function compute_primal(X, y, k, γ, σ_X; TimeLimit=-1, LogFile=nothing)
    
    n, p = size(X)
    
    model = create_gurobi_model(;LogFile=LogFile, TimeLimit=TimeLimit)

    # TODO: change big-M values
    M1 = 1000
    M2 = 1000

    @variable(model, β[i=1:p])
    @variable(model, s[i=1:p], Bin)
    @variable(model, b[i=1:p], Bin)

    @constraint(model, sum(s) <= k)
    
    @constraint(model, [i=1:p], β[i] <= M1*s[i])
    @constraint(model, [i=1:p], β[i] >= -M1*s[i])

    @constraint(model, [i=1:p], β[i]/σ_X[i] + M2*b[i] >= s[i])
    @constraint(model, [i=1:p], -β[i]/σ_X[i] + M2*(1-b[i]) >= s[i])

    yty = y'y
    XtX = X'X
    ytX = y'X
    
    @objective(model, Min, 0.5*(yty - 2*ytX*β + β'*XtX*β + (1/γ)*sum(β[j]^2 for j=1:p)))
        
    optimize!(model)
    
    return objective_value(model), value.(β)
end

end # Module