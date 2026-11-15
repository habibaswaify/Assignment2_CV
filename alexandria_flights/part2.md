# Part 2: From Matches to Motion

All code for this part is in `code/fitting.py` (2.1a, 2.1b, 2.1c and 2.2).

## Q2.1: Fitting Transforms

**`fit_similarity` (2.1a).** A similarity is `x' = a*x - b*y + tx`, `y' = b*x + a*y + ty`. It is not
linear in `(x, y)`, but it is linear in the four unknowns `(a, b, tx, ty)`. Every match therefore gives two
rows of a linear system `A u = rhs`:

| equation | row of `A` for `(a, b, tx, ty)` | right-hand side |
|---|---|---|
| x' | `[x, -y, 1, 0]` | `x'` |
| y' | `[y,  x, 0, 1]` | `y'` |

The system has 2N rows and 4 unknowns and is solved by least squares (`np.linalg.lstsq`).

**Hartley normalization (2.1b).** Each point set is moved to zero mean and scaled so that the average
distance from the origin is sqrt(2). `p` and `q` each get their own 3x3 matrix (`Tp`, `Tq`). The given code
undoes it afterwards with `H = inv(Tq) @ Hn @ Tp`, because `Tq q ~ Hn Tp p` implies `q ~ inv(Tq) Hn Tp p`.

**DLT (2.1c).** The homography equation `u = (h1.p) / (h3.p)` is not linear, but cross-multiplying gives
`h1.p - u (h3.p) = 0`, which is linear in the nine entries of `h`. Each match adds two rows to the
2N x 9 matrix `A`:

```
[-x, -y, -1,  0,  0,  0,  u*x, u*y, u]
[ 0,  0,  0, -x, -y, -1,  v*x, v*y, v]
```

We want `A h = 0`. With noisy data this cannot hold exactly, so we minimize `|A h|` subject to `|h| = 1`
(the constraint excludes `h = 0` and reflects that the scale of a homography does not matter, so it has 8
degrees of freedom). The minimizer is the right singular vector of `A` for the smallest singular value, which
is the last row of `Vt` from `np.linalg.svd`. With exactly 4 matches `A` is 8 x 9, and the full `Vt`
(NumPy's default) still contains the null-space vector.

**Normalization:** Pixel coordinates are in the hundreds, so entries such as `u*x` are around
10^5 while other entries are 1. The matrix is badly conditioned, and the SVD's algebraic error is dominated
by the large columns. After normalization all entries are of order 1.

### Results

Error of each fit on five pairs of consecutive corniche frames, fitted only to the correct matches. The error
is the mean distance in pixels between where the fit and the ground truth put the frame's four corners and
center.

| frames | correct matches | similarity (px) | homography (px) | homography, not normalized (px) |
|---|---|---|---|---|
| 10 -> 11 | 377 | 0.51 | 0.14 | 0.31 |
| 20 -> 21 | 337 | 0.33 | 0.08 | 0.30 |
| 30 -> 31 | 312 | 0.42 | 0.42 | 0.15 |
| 40 -> 41 | 278 | 0.80 | 0.23 | 0.97 |
| 50 -> 51 | 348 | 0.59 | 0.12 | 0.43 |
| mean |  | 0.53 | 0.20 | 0.43 |


### Takeaways:

- The homography (0.20 px mean) beats the similarity (0.53 px) because it can absorb the aircraft’s tilt. 
- Normalization improves the homography on average (0.20 px vs 0.43 px) by keeping the DLT matrix well conditioned. 
- The gain isn’t uniform: on pair 30 → 31 the unnormalized fit happened to be better, which is within the noise with 300+ clean matches. 

## Q2.2: RANSAC

For every hypothesis `T` (the transform fitted to one random sample) that passes `plausible()`:

1. **Score it.** The error of a match is `|T(p_i) - q_i|`. A match is an inlier if this is below 3 px.
2. **Keep best.** If the hypothesis has more inliers than the best so far (and at least `sample` of them),
   store its inlier mask.
3. **Update number of iterations** with the formula of Lecture 8. Let `w = k / n` be the inlier ratio
   of the new best hypothesis and `s` the sample size (2 or 4). One sample is all inliers with probability
   `w^s`, so after `N` samples the probability that every one was contaminated is `(1 - w^s)^N`. Requiring
   this to be at most `1 - p` with `p = 0.99` gives

   `N = log(1 - p) / log(1 - w^s)`

   It is recomputed only when a better hypothesis is found, because only then does the inlier-ratio estimate
   improve. A guard returns `N = 0` when `w^s >= 1` to avoid `log(0)`.


### Results

RANSAC was given all the matches, wrong ones included. Means over 20 pairs of consecutive corniche frames:

| model | error (px) | iterations | inliers | outliers (% of matches) |
|---|---|---|---|---|
| similarity | 0.66 | 2.25 | 365 | 2 |
| homography | 0.38 | 4.20 | 360 | 3 |


### Takeaways:

- The homography needs more iterations than the similarity (4.20 vs 2.25) because its sample has 4 points: a sample is all inliers with probability w⁴ rather than w², so more draws are needed for the same confidence.  
- Both counts are small because only 2–3% of the matches are wrong, since the ratio test already removes most outliers.

