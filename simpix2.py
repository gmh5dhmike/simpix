import sys
import math
import time
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Optional speedup: numba
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore


# ----------------------------
# Core math / cost functions
# ----------------------------

def total_cost(src_flat: np.ndarray, tgt_flat: np.ndarray, perm: np.ndarray) -> float:
    """
    Vectorised total cost for a permutation:
    sum over pixels of || src[perm[i]] - tgt[i] ||^2

    IMPORTANT: use int32/int64 to avoid overflow when squaring.
    """
    diff = src_flat[perm].astype(np.int32) - tgt_flat.astype(np.int32)
    return float(np.sum(diff * diff, dtype=np.int64))


if NUMBA_AVAILABLE:
    @njit
    def pixel_cost_nb(src_flat: np.ndarray, tgt_flat: np.ndarray, si: int, ti: int) -> int:
        dr = int(src_flat[si, 0]) - int(tgt_flat[ti, 0])
        dg = int(src_flat[si, 1]) - int(tgt_flat[ti, 1])
        db = int(src_flat[si, 2]) - int(tgt_flat[ti, 2])
        return dr * dr + dg * dg + db * db

    @njit
    def anneal_nb(
        src_flat: np.ndarray,
        tgt_flat: np.ndarray,
        steps_per_T: int,
        T0: float,
        alpha: float,
        Tmin: float,
        seed_int: int,
    ) -> Tuple[np.ndarray, float, int, float]:
        """
        Numba-accelerated simulated annealing.
        Returns (best_perm, best_cost, n_temp_steps, last_accept_ratio).
        """
        n = src_flat.shape[0]

        # Numba supports seeding + np.random.* inside njit
        np.random.seed(seed_int)

        # initial perm (Fisher-Yates shuffle)
        perm = np.arange(n, dtype=np.int32)
        for k in range(n - 1, 0, -1):
            j = np.random.randint(0, k + 1)
            tmp = perm[k]
            perm[k] = perm[j]
            perm[j] = tmp

        # initial cost (int64 accumulator)
        cost = 0
        for i in range(n):
            cost += pixel_cost_nb(src_flat, tgt_flat, perm[i], i)

        best_cost = cost
        best_perm = perm.copy()

        T = T0
        n_temp = 0
        last_acc_ratio = 0.0

        while T > Tmin:
            accepted = 0

            for _ in range(steps_per_T):
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                if i == j:
                    continue

                pi = perm[i]
                pj = perm[j]

                c_i_old = pixel_cost_nb(src_flat, tgt_flat, pi, i)
                c_j_old = pixel_cost_nb(src_flat, tgt_flat, pj, j)

                c_i_new = pixel_cost_nb(src_flat, tgt_flat, pj, i)
                c_j_new = pixel_cost_nb(src_flat, tgt_flat, pi, j)

                dE = (c_i_new + c_j_new) - (c_i_old + c_j_old)

                if dE <= 0:
                    perm[i] = pj
                    perm[j] = pi
                    cost += dE
                    accepted += 1
                else:
                    if np.random.random() < math.exp(-dE / T):
                        perm[i] = pj
                        perm[j] = pi
                        cost += dE
                        accepted += 1

                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm.copy()

            n_temp += 1
            last_acc_ratio = accepted / float(steps_per_T)
            T *= alpha

        return best_perm, float(best_cost), n_temp, last_acc_ratio


def simulated_annealing(
    src_flat: np.ndarray,
    tgt_flat: np.ndarray,
    steps_per_T: int = 50000,
    T0: float = 1000.0,
    alpha: float = 0.95,
    Tmin: float = 1e-2,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, int]:
    """
    Simulated annealing for pixel permutation.

    Default seed is None => random each run (per professor note).

    If numba is available, uses a numba-accelerated kernel for speed.
    Returns (best_perm, best_cost, seed_used).
    """
    # If seed is None, generate a fresh seed (still print it so you can reproduce)
    if seed is None:
        seed_used = int(time.time() * 1e6) & 0xFFFFFFFF
    else:
        seed_used = int(seed) & 0xFFFFFFFF

    n = src_flat.shape[0]

    if NUMBA_AVAILABLE:
        if verbose:
            print("Numba: enabled (fast path).")
            print(f"Seed used: {seed_used}")
            print(
                f"Starting SA: N={n}, T0={T0}, alpha={alpha}, "
                f"steps_per_T={steps_per_T}, Tmin={Tmin}"
            )

        best_perm, best_cost, n_temp, last_acc = anneal_nb(
            src_flat, tgt_flat, steps_per_T, T0, alpha, Tmin, seed_used
        )

        if verbose:
            print(
                f"Finished SA: best_cost={best_cost:.3e}, "
                f"temp_steps={n_temp}, last_accept={last_acc:.3f}"
            )

        return best_perm, best_cost, seed_used

    # ----------------------------
    # Pure-Python fallback (slower)
    # ----------------------------
    import random

    rng = random.Random(seed_used)
    perm = np.arange(n, dtype=np.int32)
    rng.shuffle(perm)

    cost = total_cost(src_flat, tgt_flat, perm)
    best_perm = perm.copy()
    best_cost = cost

    if verbose:
        print("Numba: not available (slow path).")
        print(f"Seed used: {seed_used}")
        print(
            f"Starting SA: N={n}, initial cost={cost:.3e}, "
            f"T0={T0}, alpha={alpha}, steps_per_T={steps_per_T}, Tmin={Tmin}"
        )

    T = T0
    it = 0
    while T > Tmin:
        accepted = 0
        for _ in range(steps_per_T):
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue

            pi = perm[i]
            pj = perm[j]

            # local costs (use Python int math)
            dr = int(src_flat[pi, 0]) - int(tgt_flat[i, 0])
            dg = int(src_flat[pi, 1]) - int(tgt_flat[i, 1])
            db = int(src_flat[pi, 2]) - int(tgt_flat[i, 2])
            c_i_old = dr * dr + dg * dg + db * db

            dr = int(src_flat[pj, 0]) - int(tgt_flat[j, 0])
            dg = int(src_flat[pj, 1]) - int(tgt_flat[j, 1])
            db = int(src_flat[pj, 2]) - int(tgt_flat[j, 2])
            c_j_old = dr * dr + dg * dg + db * db

            dr = int(src_flat[pj, 0]) - int(tgt_flat[i, 0])
            dg = int(src_flat[pj, 1]) - int(tgt_flat[i, 1])
            db = int(src_flat[pj, 2]) - int(tgt_flat[i, 2])
            c_i_new = dr * dr + dg * dg + db * db

            dr = int(src_flat[pi, 0]) - int(tgt_flat[j, 0])
            dg = int(src_flat[pi, 1]) - int(tgt_flat[j, 1])
            db = int(src_flat[pi, 2]) - int(tgt_flat[j, 2])
            c_j_new = dr * dr + dg * dg + db * db

            dE = (c_i_new + c_j_new) - (c_i_old + c_j_old)

            if dE <= 0 or rng.random() < math.exp(-dE / T):
                perm[i], perm[j] = perm[j], perm[i]
                cost += dE
                accepted += 1
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm.copy()

        it += 1
        if verbose:
            acc_ratio = accepted / float(steps_per_T)
            print(
                f"Iter {it:3d}: T={T:8.3f}, cost={cost:.3e}, "
                f"best={best_cost:.3e}, accepted={acc_ratio:.3f}"
            )

        T *= alpha

    if verbose:
        print(f"Finished SA: best_cost={best_cost:.3e}")

    return best_perm, best_cost, seed_used


# ----------------------------
# Output / visualization
# ----------------------------

def build_collage(
    src_img: Image.Image,
    tgt_img: Image.Image,
    out_img: Image.Image,
    err_img: Image.Image,
    fname: str,
) -> None:
    """Create a 2x2 collage: source, target, swapped, absolute error."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].imshow(src_img)
    axes[0, 0].set_title("Source image A")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(tgt_img)
    axes[0, 1].set_title("Target image B")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(out_img)
    axes[1, 0].set_title("Pixel-swapped A → B")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(err_img)
    axes[1, 1].set_title("|out - target|")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved collage to {fname}")


# ----------------------------
# Main program
# ----------------------------

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python simpix2.py imageA imageB [out_image=out.png] "
            "[steps_per_T] [T0] [alpha] [Tmin] [seed]\n"
            "Notes:\n"
            " - If seed is omitted, a random seed is used each run.\n"
            " - If numba is installed, the annealing loop is much faster."
        )
        return

    fsrc = sys.argv[1]
    ftgt = sys.argv[2]
    fout = sys.argv[3] if len(sys.argv) > 3 else "out.png"

    # Optional annealing parameters from the command line
    steps_per_T = int(sys.argv[4]) if len(sys.argv) > 4 else 50000
    T0 = float(sys.argv[5]) if len(sys.argv) > 5 else 1000.0
    alpha = float(sys.argv[6]) if len(sys.argv) > 6 else 0.95
    Tmin = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-2
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else None  # default None (random each run)

    print(f"Reading images: source={fsrc}, target={ftgt}")
    src_img = Image.open(fsrc).convert("RGB")
    tgt_img = Image.open(ftgt).convert("RGB")

    if src_img.size != tgt_img.size:
        raise ValueError(
            f"Images must have the same size, got {src_img.size} vs {tgt_img.size}"
        )

    w, h = src_img.size
    n = w * h
    print(f"Image size: {w} x {h} ({n} pixels)")

    src_arr = np.array(src_img, dtype=np.uint8)
    tgt_arr = np.array(tgt_img, dtype=np.uint8)

    # flatten to (N, 3)
    src_flat = src_arr.reshape((-1, 3))
    tgt_flat = tgt_arr.reshape((-1, 3))

    t0 = time.time()
    perm, best_cost, seed_used = simulated_annealing(
        src_flat,
        tgt_flat,
        steps_per_T=steps_per_T,
        T0=T0,
        alpha=alpha,
        Tmin=Tmin,
        seed=seed,
        verbose=True,
    )
    t1 = time.time()
    elapsed = t1 - t0
    print(f"Execution time: {elapsed:.2f} s (seed={seed_used})")

    out_flat = src_flat[perm]
    out_arr = out_flat.reshape((h, w, 3)).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode="RGB")

    # Save pixel-swapped image
    out_img.save(fout)
    print(f"Saved pixel-swapped image to {fout}")

    # Absolute error image: |out - target|
    err = np.abs(out_arr.astype(np.int16) - tgt_arr.astype(np.int16)).astype(np.uint8)
    err_img = Image.fromarray(err, mode="RGB")

    # Unique collage name per run (avoids overwriting)
    base_src = fsrc.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    base_tgt = ftgt.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    collage_name = f"collage_{base_src}_to_{base_tgt}.png"
    build_collage(src_img, tgt_img, out_img, err_img, fname=collage_name)

    # PDF version of the swapped image (for submission) — unique per run
    pdf_name = f"pixel_swapped_{base_src}_to_{base_tgt}.pdf"
    out_img.save(pdf_name, "PDF", resolution=300.0)
    print(f"Saved pixel-swapped image as PDF to {pdf_name}")

    plt.show()


if __name__ == "__main__":
    main()

