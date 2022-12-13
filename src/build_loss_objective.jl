export build_loss_objective

function build_loss_objective(prob::SciMLBase.AbstractSciMLProblem, alg, loss,
                              regularization = nothing, args...;
                              priors = nothing,
                              prob_generator = STANDARD_PROB_GENERATOR,
                              kwargs...)

    cost_function = function (p, nothing)
        tmp_prob = prob_generator(prob, p)
        if typeof(loss) <: Union{L2Loss, LogLikeLoss}
            sol = solve(tmp_prob, alg, args...; saveat = loss.t, save_everystep = false,
                        dense = false, kwargs...)
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

        loss_val
    end

    OptimizationFunction(cost_function)
end
