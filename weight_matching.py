from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
from jax import random
from scipy.optimize import linear_sum_assignment


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    """We assume that one permutation cannot appear in two axes of the same weight array."""
    assert num_hidden_layers >= 1
    return permutation_spec_from_axes_to_perm({
        "Dense_0/kernel": (None, "P_0"),
        **{f"Dense_{i}/kernel": (f"P_{i-1}", f"P_{i}")
           for i in range(1, num_hidden_layers)},
        **{f"Dense_{i}/bias": (f"P_{i}", )
           for i in range(num_hidden_layers)},
        f"Dense_{num_hidden_layers}/kernel": (f"P_{num_hidden_layers-1}", None),
        f"Dense_{num_hidden_layers}/bias": (None, ),
    })


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = jnp.take(w, perm[p], axis=axis)

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(rng, ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]]
                  for p, axes in ps.perm_to_axes.items()}

    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()
            } if init_perm is None else init_perm
    perm_names = list(perm.keys())

    for iteration in range(max_iter):
        progress = False
        for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(
                    ps, perm, wk, params_b, except_axis=axis)
                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (ri == jnp.arange(len(ri))).all()

            oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
            newL = jnp.vdot(A, jnp.eye(n)[ci, :])
            print(f"{iteration}/{p}: {newL - oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = jnp.array(ci)

        if not progress:
            break

    return perm
