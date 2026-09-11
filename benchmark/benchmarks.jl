using DiffEqParamEstim, OrdinaryDiffEq, BenchmarkTools
using StableRNGs, RecursiveArrayTools

const SUITE = BenchmarkGroup()
const rng = StableRNG(123)

# Lotka-Volterra with parameters to estimate
function lotka!(du, u, p, t)
    du[1] = p[1] * u[1] - u[1] * u[2]
    du[2] = -p[2] * u[2] + u[1] * u[2]
    return nothing
end
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
p_true = [1.5, 3.0]
prob = ODEProblem(lotka!, u0, tspan, p_true)

# Synthetic data from the true parameters
data_t = collect(0.0:0.5:10.0)
data_sol = solve(prob, Tsit5(); saveat = data_t)
data = Array(data_sol)

# =============================================================================
# Loss objective construction and evaluation
# =============================================================================

SUITE["loss"] = BenchmarkGroup()

l2 = L2Loss(data_t, data)
cost_function = build_loss_objective(prob, Tsit5(), l2)

SUITE["loss"]["build"] = @benchmarkable build_loss_objective($prob, Tsit5(), $l2)
SUITE["loss"]["evaluate"] = @benchmarkable $cost_function($p_true)
SUITE["loss"]["evaluate_perturbed"] = @benchmarkable $cost_function([1.6, 2.8])

# =============================================================================
# Regularized loss
# =============================================================================

SUITE["regularized"] = BenchmarkGroup()

reg = L2Loss(data_t, data; differ_weight = 0.5, colloc_grad = nothing)
SUITE["regularized"]["build"] = @benchmarkable build_loss_objective(
    $prob, Tsit5(), $reg
)
