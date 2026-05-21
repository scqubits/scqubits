"""Numerical validation of the Transmon convergence diagnostics.

Not part of the published package. Establishes that the shipped error estimates
and verdicts are numerically trustworthy by comparing them against a high-cutoff
ground truth, across a parameter grid spanning fast and slow charge-basis
convergence and the charge-degeneracy point.

Methodology note (instability watch): at very large ``ncut`` the diagonalizer's
own floating-point noise can exceed the truncation error we are trying to
measure, which would corrupt the ground truth. The script therefore measures the
ground-truth noise floor (|E(ncut_a) - E(ncut_b)| at two high cutoffs) for every
parameter set, flags any instability, and only assesses soundness for errors
clearly above that floor.

Run:  python tools/convergence_validation.py
"""

from __future__ import annotations

import numpy as np

import scqubits as scq
import scqubits.core.units as units
import scqubits.settings as settings
import scqubits.utils.convergence_utils as cutils

settings.T1_DEFAULT_WARNING = False

NCUT_GT = 200  # ground-truth cutoff
NCUT_GT_CHECK = 160  # second high cutoff, to gauge the diagonalizer noise floor
SAFETY = settings.CONVERGENCE_SAFETY_FACTOR
RATE_FLOOR = settings.CONVERGENCE_RATE_FLOOR_HZ

# (label, EJ, EC, ng): span fast (low EJ/EC) to slow (high EJ/EC) charge-basis
# convergence, and ng = 0 / 0.25 / 0.5 (the last is the charge-degeneracy point
# with near-degenerate doublets, exercising the cluster path).
PARAM_GRID = [
    ("EJ/EC=67  ng=0   ", 20.0, 0.3, 0.0),
    ("EJ/EC=67  ng=0.25", 20.0, 0.3, 0.25),
    ("EJ/EC=67  ng=0.5 ", 20.0, 0.3, 0.5),
    ("EJ/EC=250 ng=0   ", 50.0, 0.2, 0.0),
    ("EJ/EC=20  ng=0   ", 10.0, 0.5, 0.0),
    ("EJ/EC=20  ng=0.5 ", 10.0, 0.5, 0.5),
    ("EJ/EC=5   ng=0.5 ", 2.5, 0.5, 0.5),
]
TEST_NCUTS = [4, 6, 8, 12, 16, 24, 40]
N_LEVELS = 6


def low_evals(EJ, EC, ng, ncut, n):
    return np.sort(
        scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut).eigenvals(evals_count=n)
    )


def ground_truth(EJ, EC, ng, n):
    """Return (E_exact, noise_floor): the high-cutoff spectrum and the empirical
    diagonalizer noise floor (max disagreement of low levels at two high cutoffs)."""
    e_gt = low_evals(EJ, EC, ng, NCUT_GT, n)
    e_chk = low_evals(EJ, EC, ng, NCUT_GT_CHECK, n)
    noise = float(np.max(np.abs(e_gt - e_chk)))
    return e_gt, noise


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def validate_stability_and_energies():
    section("GROUND-TRUTH STABILITY (instability watch) + ENERGY SOUNDNESS")
    worst_ratio = 0.0
    worst_where = ""
    violations = 0
    assessed = 0
    any_unstable = False
    for label, EJ, EC, ng in PARAM_GRID:
        e_gt, noise = ground_truth(EJ, EC, ng, N_LEVELS)
        unstable = noise > 1e-6
        any_unstable = any_unstable or unstable
        flag = "  <-- UNSTABLE GROUND TRUTH" if unstable else ""
        print(
            f"\n[{label}] noise floor |E{NCUT_GT_CHECK}-E{NCUT_GT}| = {noise:.2e}{flag}"
        )
        meaningful = max(10.0 * noise, 1e-9)
        for ncut in TEST_NCUTS:
            tmon = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
            rep = tmon.estimate_convergence(n_levels=N_LEVELS, mode="verify")
            e_now = low_evals(EJ, EC, ng, ncut, N_LEVELS)
            worst_r = 0.0
            worst_k = -1
            for k, v in enumerate(rep.per_level):
                true_err = abs(float(e_now[k]) - float(e_gt[k]))
                if true_err <= meaningful:
                    continue
                assessed += 1
                est = v.abs_err_est_GHz or 0.0
                r = true_err / max(est, 1e-300)
                if r > worst_r:
                    worst_r, worst_k = r, k
                if r > 1.0:
                    violations += 1
            if worst_r > worst_ratio:
                worst_ratio, worst_where = (
                    worst_r,
                    f"{label} ncut={ncut} level={worst_k}",
                )
            tag = "ok" if worst_r <= 1.0 else "UNDERESTIMATE"
            print(
                f"   ncut={ncut:3d}: worst true_err/est = {worst_r:6.3f} (level {worst_k})  [{tag}]"
            )
    print("\n" + "-" * 78)
    print(
        f"Energy soundness: {violations} underestimate(s) of {assessed} assessed levels"
    )
    print(f"Worst true_err/abs_err_est = {worst_ratio:.3f}  at  {worst_where}")
    print(f"(safety factor S = {SAFETY}; want worst <= 1.0)")
    return violations == 0 and not any_unstable


def validate_verdicts():
    section("VERDICT SOUNDNESS (no dangerous false 'converged')")
    false_converged = 0
    checks = 0
    for label, EJ, EC, ng in PARAM_GRID:
        e_gt, noise = ground_truth(EJ, EC, ng, N_LEVELS)
        tol = 10.0 * noise
        for ncut in TEST_NCUTS:
            for target in (1e-2, 1e-4, 1e-6):
                tmon = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
                rep = tmon.estimate_convergence(
                    n_levels=N_LEVELS, mode="verify", target_abs_GHz=target
                )
                e_now = low_evals(EJ, EC, ng, ncut, N_LEVELS)
                for k, v in enumerate(rep.per_level):
                    if v.status != "converged":
                        continue
                    checks += 1
                    true_err = abs(float(e_now[k]) - float(e_gt[k]))
                    if true_err > target + tol:
                        false_converged += 1
                        print(
                            f"  FALSE CONVERGED: {label} ncut={ncut} level={k} "
                            f"target={target:.0e} true_err={true_err:.2e}"
                        )
    print(
        f"\nFalse 'converged' verdicts: {false_converged} of {checks} converged-level checks"
    )
    return false_converged == 0


def validate_strict_ratio_test():
    section("STRICT-MODE RATIO TEST (asymptoticity detection)")
    # A moderately-converged transmon is in the geometric regime for the low
    # levels. A strict 'converged' requires a ratio-tested 'verified_empirical'
    # result, so strict mode must reach 'converged' backed by 'verified_empirical'
    # via the ratio test.
    tmon = scq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=16)
    rep = tmon.estimate_convergence(n_levels=4, mode="strict", target_abs_GHz=1e-4)
    evid = [v.evidence for v in rep.per_level]
    methods = [v.estimator_method for v in rep.per_level]
    all_verified = all(e == "verified_empirical" for e in evid)
    ran_ratio_test = any("ratio_test" in m for m in methods)
    print(f"  aggregate: {rep.aggregate_status}")
    print(f"  evidence : {evid}")
    print(f"  methods  : {methods}")
    print(f"  all verified_empirical={all_verified}  ran_ratio_test={ran_ratio_test}")
    return rep.aggregate_status == "converged" and all_verified and ran_ratio_test


def validate_derived():
    section("DERIVED CHANNELS vs GROUND TRUTH (ng=0, non-degenerate)")
    EJ, EC, ng, n = 20.0, 0.3, 0.0, 5
    gt = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=NCUT_GT)
    e_gt, evec_gt = gt.eigensys(evals_count=n)
    to_hz = units.units_scale_factor()
    all_sound = True
    for ncut in [6, 10, 16, 24]:
        tmon = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut)
        rep = tmon.estimate_convergence(
            n_levels=n,
            mode="verify",
            include_derived=True,
            derived_quantities=["wavefunctions", "matrix_elements", "coherence"],
        )
        e_now, evec_now = tmon.eigensys(evals_count=n)

        true_overlap = cutils.wavefunction_overlap(
            evec_now[:, :n], evec_gt[:, :n], ncut, NCUT_GT
        )
        true_wf = 1.0 - np.minimum(1.0, true_overlap)
        wf_r = _worst_ratio(true_wf, rep.derived["wavefunctions"])

        true_me = _true_matelem_change(tmon, gt, evec_now, evec_gt, n)
        me_r = _worst_ratio(true_me, rep.derived["matrix_elements"])

        co_r = 0.0
        for v in rep.derived["coherence"].per_level:
            if "noise_floor" in v.warnings:
                continue  # rate at the noise floor: relative change is meaningless
            ch = v.estimator_method[:-5]  # strip "_rate"
            try:
                r0 = (
                    abs(float(getattr(tmon, ch)(get_rate=True, esys=(e_now, evec_now))))
                    * to_hz
                )
                r1 = (
                    abs(float(getattr(gt, ch)(get_rate=True, esys=(e_gt, evec_gt))))
                    * to_hz
                )
            except Exception:
                continue
            true_rate = abs(r1 - r0) / max(r1, RATE_FLOOR)
            if true_rate > 1e-9:
                co_r = max(co_r, true_rate / max(v.eps_gap_est or 0.0, 1e-300))

        sound = wf_r <= 1.0 and me_r <= 1.0 and co_r <= 1.0
        all_sound = all_sound and sound
        print(
            f"  ncut={ncut:3d}: worst true/reported  wf={wf_r:6.3f}  "
            f"matrix_elem={me_r:6.3f}  coherence={co_r:6.3f}  "
            f"[{'ok' if sound else 'UNDERESTIMATE'}]"
        )
    return all_sound


def _worst_ratio(true_per_level, sub_report):
    worst = 0.0
    for k, v in enumerate(sub_report.per_level):
        if k >= len(true_per_level):
            break
        if true_per_level[k] <= 1e-9:
            continue
        worst = max(worst, float(true_per_level[k]) / max(v.eps_gap_est or 0.0, 1e-300))
    return worst


def _true_matelem_change(tmon, gt, evec_now, evec_gt, n):
    movement = np.zeros(n)
    for op in tmon.get_operator_names():
        try:
            m0 = np.asarray(tmon.matrixelement_table(op, evecs=evec_now[:, :n]))
            m1 = np.asarray(gt.matrixelement_table(op, evecs=evec_gt[:, :n]))
        except Exception:
            continue
        if m0.shape != (n, n) or m1.shape != (n, n):
            continue
        # Compare matrix-element magnitudes: each eigenvector's global phase is a
        # gauge choice that can flip between cutoffs, matching the framework's
        # phase-invariant comparison in _matrix_element_movement.
        d = np.abs(np.abs(m1) - np.abs(m0))
        for k in range(n):
            row_ref = max(float(np.linalg.norm(m1[k, :])), 1e-12)
            col_ref = max(float(np.linalg.norm(m1[:, k])), 1e-12)
            movement[k] = max(
                movement[k],
                max(
                    float(np.linalg.norm(d[k, :])) / row_ref,
                    float(np.linalg.norm(d[:, k])) / col_ref,
                ),
            )
    return movement


def validate_high_cutoff_check():
    section("CHECK STABILITY AT HIGH CUTOFF (refinement does not blow up)")
    ok = True
    for ncut in [60, 100, 150]:
        tmon = scq.Transmon(EJ=20.0, EC=0.3, ng=0.0, ncut=ncut)
        rep = tmon.estimate_convergence(n_levels=4, mode="verify", target_abs_GHz=1e-6)
        worst = max((v.abs_err_est_GHz or 0.0) for v in rep.per_level)
        status = rep.aggregate_status
        good = worst < 1e-4 and status == "converged"
        ok = ok and good
        print(
            f"  ncut={ncut:3d}: aggregate={status:11s} worst abs_err_est={worst:.2e}  "
            f"[{'ok' if good else 'SUSPECT'}]"
        )
    return ok


def validate_zeropi_fd_groundtruth():
    section("ZEROPI FINITE-DIFFERENCE GROUND TRUTH (no false converged, FD strict)")
    # Trusted reference: a much larger box, a much finer grid, and a higher charge
    # cutoff than any user calc below, so |E_user - E_ref| is a faithful proxy for
    # the user's true FD + charge truncation error. A second slightly-coarser
    # reference gauges the reference's own noise floor.
    EJ, EL, ECJ, EC, ng, flux = 10.0, 0.04, 20.0, 0.04, 0.1, 0.23
    n = 4

    def evals(window, pts, ncut):
        grid = scq.Grid1d(-window, window, pts)
        zp = scq.ZeroPi(
            grid=grid,
            EJ=EJ,
            EL=EL,
            ECJ=ECJ,
            EC=EC,
            ng=ng,
            flux=flux,
            ncut=ncut,
            truncated_dim=n + 2,
        )
        return np.sort(zp.eigenvals(evals_count=n))

    saved_stencil = settings.STENCIL
    try:
        settings.STENCIL = 7
        e_ref = evals(8.0 * np.pi, 500, 30)
        noise = float(np.max(np.abs(e_ref - evals(8.0 * np.pi, 440, 26))))

        ok = True
        target = 1e-2
        for stencil in (5, 7):
            settings.STENCIL = stencil
            grid = scq.Grid1d(-6.0 * np.pi, 6.0 * np.pi, 120)
            zp = scq.ZeroPi(
                grid=grid,
                EJ=EJ,
                EL=EL,
                ECJ=ECJ,
                EC=EC,
                ng=ng,
                flux=flux,
                ncut=20,
                truncated_dim=n + 2,
            )
            e_user = np.sort(zp.eigenvals(evals_count=n))
            rep = zp.estimate_convergence(
                n_levels=n, mode="strict", target_abs_GHz=target
            )
            worst_ratio = 0.0
            false_converged = 0
            for k, v in enumerate(rep.per_level):
                true_err = abs(float(e_user[k] - e_ref[k]))
                est = v.abs_err_est_GHz or 0.0
                if true_err > noise:  # only assess above the reference noise floor
                    worst_ratio = max(worst_ratio, true_err / max(est, 1e-30))
                if v.status == "converged" and true_err >= target:
                    false_converged += 1
            good = worst_ratio <= 1.0 and false_converged == 0
            ok = ok and good
            print(
                f"  STENCIL={stencil} (p={stencil - 1}): agg={rep.aggregate_status:13s} "
                f"worst true/est={worst_ratio:.2f}  false_converged={false_converged}  "
                f"[{'ok' if good else 'SUSPECT'}]"
            )
        print(f"  (reference noise floor = {noise:.2e} GHz; target = {target:.0e})")
    finally:
        settings.STENCIL = saved_stencil
    return ok


def validate_fluxonium_groundtruth():
    section("FLUXONIUM GROUND TRUTH (no false converged, HO-basis, regime grid)")
    # Fluxonium HO-basis convergence is regime-dependent: deep wells (small EL)
    # near half flux have a broad wavefunction and need a high cutoff. A
    # high-cutoff reference gives the true error of each moderate "user" cutoff;
    # the verify-mode estimate must bound it and never falsely report converged.
    n = 4
    target = 1e-4
    cutoff_gt, cutoff_gt2 = 400, 360
    # (label, EJ, EC, EL, flux): standard, half flux, deep well, and a stiffer well
    grid = [
        ("EL=0.5 flux=0   ", 8.9, 2.5, 0.5, 0.0),
        ("EL=0.5 flux=0.5 ", 8.9, 2.5, 0.5, 0.5),
        ("EL=0.2 flux=0.5 ", 8.9, 2.5, 0.2, 0.5),
        ("EL=1.0 flux=0.5 ", 5.0, 1.0, 1.0, 0.5),
    ]
    test_cutoffs = [20, 30, 50, 80]

    def evals(EJ, EC, EL, flux, cutoff):
        flx = scq.Fluxonium(
            EJ=EJ, EC=EC, EL=EL, flux=flux, cutoff=cutoff, truncated_dim=n + 2
        )
        return np.sort(flx.eigenvals(evals_count=n))

    ok = True
    for label, EJ, EC, EL, flux in grid:
        e_ref = evals(EJ, EC, EL, flux, cutoff_gt)
        noise = float(np.max(np.abs(e_ref - evals(EJ, EC, EL, flux, cutoff_gt2))))
        worst_ratio = 0.0
        false_converged = 0
        for cutoff in test_cutoffs:
            flx = scq.Fluxonium(
                EJ=EJ, EC=EC, EL=EL, flux=flux, cutoff=cutoff, truncated_dim=n + 2
            )
            e_user = np.sort(flx.eigenvals(evals_count=n))
            rep = flx.estimate_convergence(n_levels=n, target_abs_GHz=target)
            for k, v in enumerate(rep.per_level):
                true_err = abs(float(e_user[k] - e_ref[k]))
                est = v.abs_err_est_GHz or 0.0
                if true_err > noise:
                    worst_ratio = max(worst_ratio, true_err / max(est, 1e-30))
                if v.status == "converged" and true_err >= target:
                    false_converged += 1
        good = worst_ratio <= 1.0 and false_converged == 0
        ok = ok and good
        print(
            f"  {label}: worst true/est={worst_ratio:.2f}  "
            f"false_converged={false_converged}  (noise {noise:.1e})  "
            f"[{'ok' if good else 'SUSPECT'}]"
        )
    return ok


def main():
    results = {
        "energy soundness + stability": validate_stability_and_energies(),
        "verdict soundness": validate_verdicts(),
        "strict ratio test": validate_strict_ratio_test(),
        "derived channels": validate_derived(),
        "high-cutoff check stability": validate_high_cutoff_check(),
        "zeropi FD ground truth": validate_zeropi_fd_groundtruth(),
        "fluxonium ground truth": validate_fluxonium_groundtruth(),
    }
    section("SUMMARY")
    for name, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}]  {name}")
    print()
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
