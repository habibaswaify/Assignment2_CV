import sys
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils, gt, viz
from features import features, features_sift, match
from utils import apply
from fitting import ransac, fit_homography
from odometry import chain

S0 = 600.0  #given
FRAME = (640, 480)
C = np.array(FRAME) / 2.0

def distort(p, kappa, c=C): #simulate cam distortion
    d = p - c
    return c + d * (1 + kappa* (d ** 2).sum(-1, keepdims=True) / S0 ** 2)   #mathematical distortion model 


def undistort_points(pd, kappa, c=C, iters=12): #reverse distort
    d = pd - c
    rd = np.linalg.norm(d, axis=1)  #observed distance from center
    r = rd.copy()
    for _ in range(iters):  #approx - newtons method
        r = r - (r + kappa * r ** 3 / S0 ** 2 - rd) / (1 + 3 * kappa * r ** 2 / S0 ** 2)
    return c + d * (r / np.maximum(rd, 1e-9))[:, None]


def undistort_image(img, kappa):    #entire image correction
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    m = distort(np.stack([xx, yy], -1).reshape(-1, 2), kappa, np.array([w, h]) / 2.0).reshape(h, w, 2)
    return cv2.remap(img, m[..., 0].astype(np.float32), m[..., 1].astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def collect_matches(F, step=4, gap=3):  #k-estim
    out = []
    for i in range(0, len(F) - gap, step):
        (k1, d1), (k2, d2) = F[i], F[i + gap]
        mm = match(d1, d2)
        if len(mm) >= 30:
            out.append((k1[mm[:, 0]], k2[mm[:, 1]]))
    return out


def cost(kappa, M, tau=2.0):    #eval candidate k
    tot, n = 0.0, 0
    for p, q in M:
        pu, qu = undistort_points(p, kappa), undistort_points(q, kappa)
        T, _, _ = ransac(pu, qu, fit_homography, 4, 1.5)
        e2 = np.full(len(p), tau ** 2) if T is None else \
            np.minimum(((apply(T, pu) - qu) ** 2).sum(1), tau ** 2)
        tot, n = tot + e2.sum(), n + len(p)
    return tot / n


def estimate_kappa(M):  #best parameter
    coarse = np.arange(-0.30, 0.3001, 0.02)
    c1 = [cost(k, M) for k in coarse]
    k0 = coarse[int(np.argmin(c1))]
    fine = np.arange(k0 - 0.02, k0 + 0.0201, 0.004)
    c2 = [cost(k, M) for k in fine]
    return float(fine[int(np.argmin(c2))]), (coarse, c1, fine, c2)

def run_flight(name, feat):
    fd = utils.flight(name)
    imgs = utils.load_frames(fd)
    F_raw = [feat(im) for im in imgs]
    kappa, curves = estimate_kappa(collect_matches(F_raw))
    print(f'{name}: kappa = {kappa:+.3f}', flush=True)
    und = [undistort_image(im, kappa) for im in imgs]
    F_und = [feat(im) for im in und]
    res = {'kappa': kappa, 'curves': curves, 'imgs': imgs, 'und': und, 'F_raw': F_raw, 'F_und': F_und}
    for tag, F in (('raw', F_raw), ('undistorted', F_und)):
        for model in ('similarity', 'homography'):
            Ts, _, lost, _ = chain(F, model)
            e = gt.path_errors(fd, Ts)
            res[tag, model] = (Ts, float(e.mean()), float(e[-1]), len(lost))
    return res


def main():
    out = utils.results_dir('bonus')
    feat = features_sift if '--sift' in sys.argv else features
    flights = ['corniche', 'loop', 'lost_at_sea']
    R = {fl: run_flight(fl, feat) for fl in flights}

    #curves
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for a, fl in zip(ax, flights):
        coarse, s1, fine, s2 = R[fl]['curves']
        a.plot(coarse, s1, 'o-', ms=3, label='coarse'); a.plot(fine, s2, 's-', ms=3, label='fine')
        a.axvline(R[fl]['kappa'], color='k', ls='--'); a.set_title(f'{fl}: kappa = {R[fl]["kappa"]:+.3f}')
        a.set_xlabel('kappa'); a.set_ylabel('mean truncated error$^2$ (px$^2$)'); a.grid(alpha=.3)
    ax[0].legend(); fig.tight_layout(); fig.savefig(f'{out}/bonus_kappa.png', dpi=130); plt.close(fig)

    #before/after of one frame
    fig, ax = plt.subplots(2, 3, figsize=(14, 7))
    for j, fl in enumerate(flights):
        ax[0, j].imshow(R[fl]['imgs'][20], cmap='gray'); ax[0, j].set_title(f'{fl}: original')
        ax[1, j].imshow(R[fl]['und'][20], cmap='gray'); ax[1, j].set_title(f'undistorted, kappa = {R[fl]["kappa"]:+.3f}')
        ax[0, j].axis('off'); ax[1, j].axis('off')
    fig.tight_layout(); fig.savefig(f'{out}/bonus_undistorted.png', dpi=130); plt.close(fig)

    # Q3.2 again comp
    rows = []
    for fl in ('corniche', 'loop', 'lost_at_sea'):
        for model in ('similarity', 'homography'):
            _, m0, f0, _ = R[fl]['raw', model]
            _, m1, f1, l1 = R[fl]['undistorted', model]
            rows.append([fl, f'{R[fl]["kappa"]:+.3f}', model, m0, f0, m1, f1])
    print('\nBonus  Q3.2 again, on the undistorted frames')
    utils.table(f'{out}/bonus_models.md',
                ['flight', 'kappa', 'model', 'mean before (m)', 'final before (m)', 'mean after (m)', 'final after (m)'], rows)
    viz.plot_paths(utils.flight('loop'),
                   {'similarity, original': R['loop']['raw', 'similarity'][0],
                    'homography, original': R['loop']['raw', 'homography'][0],
                    'similarity, undistorted': R['loop']['undistorted', 'similarity'][0],
                    'homography, undistorted': R['loop']['undistorted', 'homography'][0]},
                   f'{out}/bonus_paths_loop.png')

    # match error against radius, before and after
    for tag in ('raw', 'undistorted'):
        key = 'F_raw' if tag == 'raw' else 'F_und'
        pairs = []
        for fl, i, j in (('corniche', 20, 21), ('loop', 20, 21), ('loop', 3, 132)):
            (k1, d1), (k2, d2) = R[fl][key][i], R[fl][key][j]
            mm = match(d1, d2)
            pairs.append((f'{fl}, frames {i} and {j}', k1[mm[:, 0]], k2[mm[:, 1]], gt.gt_H(utils.flight(fl), i, j)))
        viz.plot_error_vs_radius(pairs, f'{out}/bonus_error_vs_radius_{tag}.png')


if __name__ == '__main__':
    utils.run(main)