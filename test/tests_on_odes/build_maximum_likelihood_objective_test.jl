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
maximum_likelihood_cost_function = build_loss_objective(prob1,Tsit5(),MaximumLikelihood(t,maximum_likelihood_data,variance),regularization = Regularization(0,penalty),maxiters=10000)

using Optim
result2 = optimize(maximum_likelihood_cost_function, [1.42], BFGS())
