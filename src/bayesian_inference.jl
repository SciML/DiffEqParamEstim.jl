export StanModel, bayesian_inference

struct StanModel{R,C}
  return_code::R
  chain_results::C
end
function bayesian_inference(prob::DEProblem,t,data;alg="integrate_ode_rk45",num_samples=1, num_warmup=1,kwargs...)
  const parameter_estimation_model = "
  functions {
    real[] sho(real t,real[] y,real[] theta,real[] x_r,int[] x_i) {
      real dydt[2];
      dydt[1] = theta[1] * y[1] - 1.0*y[1] * y[2];
      dydt[2] = -3.0*y[1] + 1.0*y[1] * y[2];
      return dydt;
      }
    }
  data {
    real y0[2];
    int<lower=1> T;
    real y[T,2];
    real t0;
    real ts[T];
  }
  transformed data {
    real x_r[0];
    int x_i[0];
  }
  parameters {
    vector<lower=0>[2] sigma;
    real theta[1];
  }
  model{
    real y_hat[T,2];
    sigma ~ inv_gamma(2, 3);
    theta[1] ~ normal(1.5, 1);
    y_hat = integrate_ode_rk45(sho, y0, t0, ts, theta, x_r, x_i);
    for (t in 1:T){
      y[t] ~ normal(y_hat[t], sigma);
      }
  }
  "

  stanmodel = Stanmodel(num_samples=num_samples, num_warmup=num_warmup, name="parameter_estimation_model", model=parameter_estimation_model);
  const parameter_estimation_data = Dict("y0"=>prob.u0, "T" => size(t)[1], "y" => data', "t0" => prob.tspan[1], "ts"=>t)
  return_code, chain_results = stan(stanmodel, [parameter_estimation_data]; CmdStanDir=CMDSTAN_HOME)
  return StanModel(return_code,chain_results)
end
