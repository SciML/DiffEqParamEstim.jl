@setup_workload begin
    precompile_rhs = function (du, u, p, t)
        du[1] = p[1] * u[1]
        return du[2] = -p[1] * u[2]
    end
    precompile_problem = SciMLBase.ODEProblem(
        precompile_rhs, [1.0, 1.0], (0.0, 1.0), [0.5]
    )
    precompile_t = collect(range(0.0, 1.0; length = 5))
    precompile_data = reshape(collect(range(1.0, 2.0; length = 10)), 2, :)
    precompile_loss = L2Loss(precompile_t, precompile_data)

    @compile_workload begin
        Regularization(0.1)([1.0, 2.0])
        calckernel(EpanechnikovKernel(), 0.0)
        colloc_grad(precompile_t, precompile_data)
        two_stage_objective(
            precompile_problem, precompile_t, precompile_data
        )
        build_loss_objective(precompile_problem, nothing, precompile_loss)
    end
end
