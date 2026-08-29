"""Regression test for the analytic optimal control u*.

At the terminal time the Feynman-Kac / Doob boundary identity forces

    u*(1, x) = -sigma(1) * grad_g(x),     grad_g = -x/nu_1 - grad_r(x)

for *every* control, independent of the mixture.  `GaussianMixtureTarget`
(src/adjoint_sampling/bimodal_target.py) builds u* from the h-transform mixture;
completing the square leaves a mode-dependent constant
kappa_i = d * lambda_i * mu_i**2 / (2 nu_1 lambda_i*) in log A_i.  Omitting it
(the original bug) silently breaks u* whenever the modes are asymmetric
(mu_1**2 != mu_2**2 or lambda_1 != lambda_2) while leaving symmetric setups
correct — so this test exercises asymmetric mixtures specifically.
"""
import pytest
import torch

from adjoint_sampling import GaussianMixtureTarget, utils
from adjoint_sampling.utils import sigma_int_from_nu


@pytest.mark.parametrize("schedule,sched_args", [
    ("constant", (2.0, 0.0)),
    ("linear", (2.0, 0.25)),
])
@pytest.mark.parametrize("w1,l1,m1,l2,m2,label", [
    (0.5, 1.0, -3.0, 1.0, 0.0, "asymmetric means"),
    (0.5, 1.0, -3.0, 1.0, 3.0, "symmetric means"),
    (0.25, 1.0, -3.0, 1.0, 3.0, "asymmetric weights"),
    (0.5, 1.0, -3.0, 2.0, 3.0, "asymmetric precisions"),
    (0.5, 1.0, 3.0, 1.0, 3.0, "coincident modes (gaussian)"),
])
def test_u_star_terminal_boundary_identity(schedule, sched_args, w1, l1, m1, l2, m2, label):
    sigma_fn, nu_fn, nu_1 = utils.make_noise_schedule(schedule, *sched_args)
    sigma_int_fn = sigma_int_from_nu(nu_fn, nu_1)
    target = GaussianMixtureTarget(w1=w1, lambda1=l1, mu1=m1,
                                   w2=1.0 - w1, lambda2=l2, mu2=m2)

    x = torch.linspace(-8.0, 8.0, 96).unsqueeze(-1)
    t1 = torch.ones(x.shape[0])

    u_star = target.optimal_control(x, t1, sigma_fn, sigma_int_fn, nu_1, d=1)
    u_boundary = -sigma_fn(t1).unsqueeze(-1) * target.grad_g(x, nu_1)

    max_err = (u_star - u_boundary).abs().max().item()
    assert max_err < 1e-4, f"{schedule}/{label}: |u*(1,x) + sigma(1) grad_g| = {max_err:.3e}"
