using Distributions, RecursiveArrayTools
srand(123)
original_solution = [(sol(t[i])) for i in 1:length(t)]
original_solution_matrix_form = vecvec_to_mat(original_solution)

variance = zeros(size(original_solution_matrix_form)[1],size(original_solution_matrix_form)[2])
error = zeros(size(original_solution_matrix_form)[1],size(original_solution_matrix_form)[2])

for i in 1:size(original_solution_matrix_form)[1]
  for j in 1:size(original_solution_matrix_form)[2]
    tmp = rand()
    variance[i,j] = tmp*tmp
    d = Normal(0,tmp)
    error[i,j] = rand(d)
  end
end
#variance
maximum_likelihood_data = original_solution_matrix_form + error
#maximum_likelihood_function
maximum_likelihood_cost_function = build_loss_objective(prob1,Tsit5(),MaximumLikelihood(t,maximum_likelihood_data,variance),maxiters=10000)

using NLopt
opt = Opt(:GN_ESCH, 1)
min_objective!(opt, maximum_likelihood_cost_function.cost_function2)
lower_bounds!(opt,[0.0])
upper_bounds!(opt,[5.0])
xtol_rel!(opt,1e-3)
maxeval!(opt, 10000)
(minf,minx,ret) = NLopt.optimize(opt,[1.3])
@test minx[1] ≈ 1.3 atol=1e-1
