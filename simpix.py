import sys
import math
import random
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def pixel_cost(src_pix: np.ndarray, tgt_pix: np.ndarray) -> int:
    """
    Squared L2 distance between two RGB pixels, using Python ints
    to avoid uint8 overflow.
    """
    dr = int(src_pix[0]) - int(tgt_pix[0])
    dg = int(src_pix[1]) - int(tgt_pix[1])
    db = int(src_pix[2]) - int(tgt_pix[2])
    return dr * dr + dg * dg + db * db


def initial_permutation(n: int, rng: random.Random) -> np.ndarray:
    """Return a random permutation of 0..n-1."""
    perm = np.arange(n, dtype=np.int32)
    rng.shuffle(perm)
    return perm


def total_cost(src_flat: np.ndarray, tgt_flat: np.ndarray, perm: np.ndarray) -> float:
    """
    Vectorised total cost for a permutation:
    sum over pixels of || src[perm[i]] - tgt[i] ||^2
    """
    diff = src_flat[perm].astype(np.int16) - tgt_flat.astype(np.int16)
    return float(np.sum(diff * diff))


def simulated_annealing(
    src_flat: np.ndarray,
    tgt_flat: np.ndarray,
    steps_per_T: int = 50000,
    T0: float = 1000.0,
    alpha: float = 0.95,
    Tmin: float = 1e-2,
    seed: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    Simulated annealing for the pixel permutation:

    - State: a permutation perm[i] giving which source pixel goes to target pixel i.
    - Energy: sum_i || src[perm[i]] - tgt[i] ||^2.
    - Move: pick two target indices i, j and swap perm[i], perm[j].
    - Accept always if dE <= 0, or with prob exp(-dE/T) if dE > 0.
    """
    rng = random.Random(seed)
    n = src_flat.shape[0]

    perm = initial_permutation(n, rng)
    cost = total_cost(src_flat, tgt_flat, perm)
    best_perm = perm.copy()
    best_cost = cost

    T = T0
    it = 0

    print(
        f"Starting SA: N={n}, initial cost={cost:.3e}, "
        f"T0={T0}, alpha={alpha}, steps_per_T={steps_per_T}, Tmin={Tmin}"
    )

    while T > Tmin:
        accepted = 0
        for _ in range(steps_per_T):
            # pick two distinct target indices to swap their source assignments
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue

            pi = perm[i]
            pj = perm[j]

            # old local costs
            c_i_old = pixel_cost(src_flat[pi], tgt_flat[i])
            c_j_old = pixel_cost(src_flat[pj], tgt_flat[j])

            # new local costs if we swapped assignments
            c_i_new = pixel_cost(src_flat[pj], tgt_flat[i])
            c_j_new = pixel_cost(src_flat[pi], tgt_flat[j])

            dE = (c_i_new + c_j_new) - (c_i_old + c_j_old)

            if dE <= 0 or rng.random() < math.exp(-dE / T):
                # accept
                perm[i], perm[j] = perm[j], perm[i]
                cost += dE
                accepted += 1
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm.copy()

        it += 1
        acc_ratio = accepted / float(steps_per_T)
        print(
            f"Iter {it:3d}: T={T:8.3f}, cost={cost:.3e}, "
            f"best={best_cost:.3e}, accepted={accepted} ({acc_ratio:.3f})"
        )
        T *= alpha

    print(f"Finished SA: best_cost={best_cost:.3e}")
    return best_perm, best_cost


def build_collage(
    src_img: Image.Image,
    tgt_img: Image.Image,
    out_img: Image.Image,
    fname: str = "collage.png",
) -> None:
    """Create a 2x2 collage similar to the starter code."""
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

    # Same pixel-swapped result again in bottom-right
    axes[1, 1].imshow(out_img)
    axes[1, 1].set_title("Pixel-swapped (duplicate)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(fname)
    print(f"Saved collage to {fname}")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python simpix.py imageA imageB [out_image=out.png] "
            "[steps_per_T] [T0] [alpha] [Tmin] [seed]"
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
    seed = int(sys.argv[8]) if len(sys.argv) > 8 else 1

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

    perm, best_cost = simulated_annealing(
        src_flat,
        tgt_flat,
        steps_per_T=steps_per_T,
        T0=T0,
        alpha=alpha,
        Tmin=Tmin,
        seed=seed,
    )

    out_flat = src_flat[perm]
    out_arr = out_flat.reshape((h, w, 3)).astype(np.uint8)
    out_img = Image.fromarray(out_arr, mode="RGB")

    # Save pixel-swapped image
    out_img.save(fout)
    print(f"Saved pixel-swapped image to {fout}")

    # Collage of originals + result
    build_collage(src_img, tgt_img, out_img, fname="collage.png")

    # PDF version of the swapped image (for submission)
    pdf_name = "pixel_swapped.pdf"
    out_img.save(pdf_name, "PDF", resolution=300.0)
    print(f"Saved pixel-swapped image as PDF to {pdf_name}")

    plt.show()


if __name__ == "__main__":
    main()
