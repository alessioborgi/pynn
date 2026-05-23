# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Gabriele Onorato, Luke Braithwaite,
#   Mario Severino, Emanuele Mule, Dario Loi,
#   Francesco Restuccia, Fabrizio Silvestri, Pietro Liò

import math
import random

import numpy as np
import torch


def setup_torch(precision: str = "high", seed: int = 42) -> None:
    """Sets precision for float32 matrix multiplications and random seeds.

    Configures PyTorch and NumPy for reproducibility and performance.
    """
    torch.set_float32_matmul_precision(precision)
    torch.manual_seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = True  # Enable TF32 for cuDNN operations
        torch.backends.cuda.matmul.allow_tf32 = (
            True  # Enable TF32 for matrix multiplications
        )
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        _ver = torch.version  # ty: ignore[possibly-missing-submodule]
        cuda_ver = getattr(_ver, "cuda", "N/A")
        print(f"CUDA version: {cuda_ver}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            msg = (
                "Using Apple Silicon GPU (MPS), "
                "falling back to CPU for reproducibility."
            )
            print(msg)
            setattr(torch.backends.mps, "is_available", lambda: False)  # noqa: B010
        else:
            print("Using CPU")

    print(f"Float32 matmul precision set to: {precision}")
    print(f"Random seed set to: {seed}", flush=True)
    return None


def cayley(params: torch.Tensor, d: int, clamp_val: float = 10.0) -> torch.Tensor:
    """Cayley transform from skew-symmetric matrix entries to orthogonal W.

    Args:
        params: Tensor of shape ``[N, d*(d-1)//2]`` containing the independent upper
            triangular entries of the skew-symmetric matrix.
        d: The dimension of the resulting orthogonal matrix.
        clamp_val: Maximum absolute value for clamping parameters before constructing
            the skew-symmetric matrix to avoid ill-conditioned solves.

    Returns:
        Orthogonal restriction map of shape ``[N, d, d]``.
    """
    params = torch.clamp(params, -clamp_val, clamp_val)
    A = torch.zeros(params.size(0), d, d, device=params.device, dtype=params.dtype)
    indices = torch.triu_indices(d, d, offset=1)

    # Fill the strict upper triangle, then mirror it with a sign flip.
    A[:, indices[0], indices[1]] = params
    A = A - A.transpose(-2, -1)

    # Solve is more stable than forming inverse(I - A) explicitly.
    eye = torch.eye(d, device=params.device, dtype=params.dtype).unsqueeze(0)
    return torch.linalg.solve(eye - A, eye + A)


def attention_cayley(
    raw: torch.Tensor, d: int, device=None, dtype=None
) -> torch.Tensor:
    """Cayley transform via skew-symmetric part of I - softmax(raw).

    Args:
        raw: Raw un-normalized attention parameters of shape ``[N, d * d]``.
        d: The dimension of the resulting orthogonal matrix.
        device: Target tensor device.
        dtype: Target tensor dtype.

    Returns:
        Orthogonal restriction map of shape ``[N, d, d]``.
    """
    if device is None:
        device = raw.device
    if dtype is None:
        dtype = raw.dtype
    eye = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
    M = eye - torch.softmax(raw.view(-1, d, d), dim=-1)
    A = (M - M.transpose(-2, -1)) / 2  # skew-symmetric part; ensures orthogonality
    return torch.linalg.solve(eye - A, eye + A)


def householder(
    params: torch.Tensor, d: int, stop_recursion: int | None = None
) -> torch.Tensor:
    """Batched, autograd-safe FastH implementation.
    from: github.com/alexandermath/fasth
    What if Neural Networks had SVDs? NeurIPS 2020.
    Converts [E, d*d] params to orthogonal [E, d, d] via parallel WY Decomposition.

    Args:
        params: Raw Householder parameters of shape ``[E, d * d]``.
        d: The dimension of the resulting orthogonal matrix.
        stop_recursion: If provided, stop the parallel reduction at this step.

    Returns:
        Orthogonal restriction map of shape ``[E, d, d]``.
    """
    E, _ = params.shape

    # 1. Reshape and normalize to get unit Householder vectors
    V = params.view(E, d, d)
    norms = V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    V = V / norms

    # 2. FastH requires d to be a power of 2. Pad securely if necessary.
    D = 2 ** math.ceil(math.log2(d))

    if D > d:
        # Pad with standard basis vectors e_k.
        # I - 2e_k e_k^T leaves the top-left d x d block strictly orthogonal.
        V_padded = torch.zeros(E, D, D, device=params.device, dtype=params.dtype)
        V_padded[:, :d, :d] = V
        for i in range(d, D):
            V_padded[:, i, i] = 1.0
        V = V_padded

    # V now has shape [E, D, D], where rows are Householder vectors.
    Y_ = V
    W_ = -2 * Y_.clone()

    k = 1
    num_iters = int(np.log2(D))

    # 3. Parallel Reduction (Step 1 of FastH)
    for c in range(num_iters):
        k_2 = k
        k *= 2

        Y_v = Y_.view(E, D // k_2, k_2, D)
        W_v = W_.view(E, D // k_2, k_2, D)

        # Compute cross terms
        m1_ = torch.matmul(Y_v[:, 0::2], W_v[:, 1::2].transpose(-1, -2))
        m2_ = torch.matmul(W_v[:, 0::2].transpose(-1, -2), m1_)

        # Autograd-safe update (Avoids inplace W_[1::2] += ... operations)
        W_v_0 = W_v[:, 0::2]
        W_v_1 = W_v[:, 1::2] + m2_.transpose(-1, -2)

        # Interleave the blocks cleanly using stack and view
        W_ = torch.stack((W_v_0, W_v_1), dim=2).view(E, D, D)

        if stop_recursion is not None and c == stop_recursion:
            break

    # 4. Apply the resulting WY representation to an Identity matrix
    # (Step 2 of FastH)
    X = torch.eye(D, dtype=V.dtype, device=V.device).unsqueeze(0).expand(E, -1, -1)

    if stop_recursion is None:
        W_out = X + torch.matmul(W_.transpose(-1, -2), torch.matmul(Y_, X))
    else:
        for i in range(D // k - 1, -1, -1):
            W_slice = W_[:, i * k : (i + 1) * k]
            Y_slice = Y_[:, i * k : (i + 1) * k]
            X = X + torch.matmul(W_slice.transpose(-1, -2), torch.matmul(Y_slice, X))
        W_out = X

    # Return the exact d x d unpadded orthogonal block
    return W_out[:, :d, :d]
