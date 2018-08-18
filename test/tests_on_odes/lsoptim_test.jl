using LeastSquaresOptim

@test_broken begin
println("Use LeastSquaresOptim to fit the parameter")
cost_function = build_lsoptim_objective(prob1,t,data,Tsit5(),verbose=false)
x = [1.0]
res = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(x = x,
                f! = cost_function,
                output_length = length(t)*length(prob1.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR())
@test result.minimizer[1] ≈ 1.5 atol=3e-1
cost_function = build_lsoptim_objective(prob2,t,data,Tsit5(),verbose=false)
x = [1.3,2.7]
res = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(x = x,
                f! = cost_function,
                output_length = length(t)*length(prob2.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR())
@test res.minimizer ≈ [1.5;3.0] atol=3e-1
cost_function = build_lsoptim_objective(prob3,t,data,Tsit5(),verbose=false)
x = [1.3,0.8,2.8,1.2]
res = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(
                x = x, f! = cost_function,
                output_length = length(t)*length(prob3.u0)),
                LeastSquaresOptim.Dogleg(),LeastSquaresOptim.LSMR())
@test res.minimizer ≈ [1.5;1.0;3.0;1.0] atol=3e-1
end

