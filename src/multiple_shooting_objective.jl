export multiple_shooting_objective

function generate_loss_func(loss, t, i)
    new_loss = nothing
    if typeof(loss) <: L2Loss && i == 1
        new_loss = L2Loss(t, loss.data[:, i:(i + length(t))])
    elseif typeof(loss) <: L2Loss
        i = (i - 1) * length(t)
        if i + length(t) < size(loss.data)[2]
            new_loss = L2Loss(t, loss.data[:, i:(i + length(t))])
        else
            new_loss = L2Loss(t, loss.data[:, i:size(loss.data)[2]])
        end
    end
    new_loss
end

struct Merged_Solution{T1, T2, T3}
    u::T1
    t::T2
    sol::T3
end

function multiple_shooting_objective(prob::DiffEqBase.DEProblem, alg, loss,
                                     regularization = nothing; priors = nothing,
                                     discontinuity_weight = 1.0,
                                     verbose_opt = false, verbose_steps = 100,
                                     prob_generator = STANDARD_MS_PROB_GENERATOR,
                                     kwargs...)
    if verbose_opt
        count = 0 # keep track of # function evaluations
    end

    cost_function = function (p)
        t0, tf = prob.tspan
        P, N = length(prob.p), length(prob.u0)
        K = Int((length(p) - P) / N)
        τ = range(t0, tf, length = K + 1)
        sol = []
        loss_val = 0
        for k in 1:K
            tmp_prob = prob_generator(prob, p, k)
            if typeof(loss) <: Union{L2Loss, LogLikeLoss}
                time_save = loss.t[findall(t -> τ[k] <= t <= τ[k + 1], loss.t)]
                push!(sol,
                      solve(tmp_prob, alg; saveat = time_save,
                            save_everystep = false, dense = false, kwargs...))
            else
                push!(sol, solve(tmp_prob, alg; kwargs...))
            end
        end
        if any((!SciMLBase.successful_retcode(s.retcode) for s in sol))
            return Inf
        end
        u = [uc for k in 1:K for uc in sol[k].u[1:(end - 1)]]
        t = [tc for k in 1:K for tc in sol[k].t[1:(end - 1)]]
        push!(u, sol[K].u[end])
        push!(t, sol[K].t[end])
        sol_loss = Merged_Solution(u, t, sol)
        sol_new = DiffEqBase.build_solution(prob, alg, sol_loss.t, sol_loss.u,
                                            retcode = ReturnCode.Success)
        loss_val = loss(sol_new)
        if priors !== nothing
            loss_val += prior_loss(priors, p[(end - length(priors)):end])
        end
        if regularization !== nothing
            loss_val += regularization(p)
        end
        for k in 2:K
            if typeof(discontinuity_weight) <: Real
                loss_val += discontinuity_weight *
                            sum((sol[k][1] - sol[k - 1][end]) .^ 2)
            else
                loss_val += sum(discontinuity_weight .*
                                (sol[k][1] - sol[k - 1][end]) .^ 2)
            end
        end
        if verbose_opt
            count::Int += 1
            if mod(count, verbose_steps) == 0
                println("Iteration: $count")
                println("Current Cost: $loss_val")
                println("Parameters: $p")
            end
        end
        loss_val
    end

    DiffEqObjective(cost_function, nothing)
end
