export build_loss_objective

function build_loss_objective(
        prob::SciMLBase.AbstractSciMLProblem, alg, loss,
        adtype = SciMLBase.NoAD(),
        regularization = nothing, args...;
        priors = nothing,
        prob_generator = STANDARD_PROB_GENERATOR,
        kwargs...
    )
    cost_function = function (p, _ = nothing)
        tmp_prob = prob_generator(prob, p)
        if loss isa Union{L2Loss, LogLikeLoss}
            sol = solve(
                tmp_prob, alg, args...; saveat = loss.t, save_everystep = false,
                dense = false, kwargs...
            )
        else
            sol = solve(tmp_prob, alg, args...; kwargs...)
        end

        loss_val = loss(sol)

        if priors !== nothing
            loss_val += prior_loss(priors, p)
        end
        if regularization !== nothing
            loss_val += regularization(p)
        end
        return loss_val
    end

    return OptimizationFunction(cost_function, adtype)
end
