export StanModel, bayesian_inference

struct StanModel{R,C}
  return_code::R
  chain_results::C
end
function bayesian_inference(prob::DEProblem,t,data;alg="integrate_ode_rk45",num_samples=1, num_warmup=1,kwargs...)
  const parameter_estimation_model = "
  functions {
    real[] sho(real t,real[] u,real[] theta,real[] x_r,int[] x_i) {
      real du[2];  // 2 = length(prob.u0)
      du[1] = theta[1] * u[1] - 1.0*u[1] * u[2]; //string(prob.f.pfuncs[1])
      du[2] = -3.0*u[1] + 1.0*u[1] * u[2];
      return du;
      }
    }
  data {
    real u0[2]; // 2 = length(prob.u0)
    int<lower=1> T;
    real u[T,2]; // 2 = length(prob.u0)
    real t0;
    real ts[T];
  }
  transformed data {
    real x_r[0];
    int x_i[0];
  }
  parameters {
    vector<lower=0>[2] sigma;   // 2 = length(prob.u0)
    real theta[1];   // // 1=length(prob.f.params)
  }
  model{
    real u_hat[T,2]; // 2 = length(prob.u0)
    sigma ~ inv_gamma(2, 3);
    // Define a placeholder here
    theta[1] ~ normal(1.5, 1); //1=length(prob.f.params), 1.5=prob.f.a
    u_hat = integrate_ode_rk45(sho, u0, t0, ts, theta, x_r, x_i);
    for (t in 1:T){
      u[t] ~ normal(u_hat[t], sigma);
      }
  }
  "

  stanmodel = Stanmodel(num_samples=num_samples, num_warmup=num_warmup, name="parameter_estimation_model", model=parameter_estimation_model);
  const parameter_estimation_data = Dict("y0"=>prob.u0, "T" => size(t)[1], "y" => data', "t0" => prob.tspan[1], "ts"=>t)
  return_code, chain_results = stan(stanmodel, [parameter_estimation_data]; CmdStanDir=CMDSTAN_HOME)
  return StanModel(return_code,chain_results)
end
