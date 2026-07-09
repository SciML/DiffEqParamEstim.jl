# API Reference

This page documents the public API of DiffEqParamEstim.jl.

## Objective Builders

```@docs
build_loss_objective
multiple_shooting_objective
two_stage_objective
```

## Cost Functions

```@docs
L2Loss
LogLikeLoss
Regularization
TwoStageCost
```

## Helper Functions

```@docs
prior_loss
colloc_grad
l2lossgradient!
```
