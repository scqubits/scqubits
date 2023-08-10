import argparse
from itertools import product

import numpy as np
import h5py

import scqubits as scq

E_C = 11
ECm = 50.0
E_L1 = 3.52
E_L2 = 3.52
E_La = 0.271
E_Lb = 0.266
E_J = 4.246
E_Ja = 5.837
E_Jb = 4.930
E_Ca = 0.892
E_Cb = 0.8655

flux_c, flux_s = 0.2662, 0.01768

FTC_grounded = scq.FluxoniumTunableCouplerGrounded(EJa=E_Ja, EJb=E_Jb, EC_twoqubit=np.inf,
                                               ECq1=E_Ca, ECq2=E_Cb, ELa=E_La, ELb=E_Lb,
                                               flux_a=0.5, flux_b=0.5, flux_c=0.30,
                                               fluxonium_cutoff=110, fluxonium_truncated_dim=7,
                                               ECc=E_C, ECm=ECm, EL1=E_L1, EL2=E_L2, EJC=E_J,
                                               fluxonium_minus_truncated_dim=7, h_o_truncated_dim=7)


def run_ftc_zz(
    idx,
    num_pts,
    num_cpus=4,
    highest_exc_q=2,
    highest_exc_m=3,
    highest_exc_p=3,
    flux_q_max=0.06,
    flux_c_max=0.5,
):
    flux_q_linspace = np.linspace(0.0, flux_q_max, num_pts)
    flux_c_linspace = np.linspace(0.0, flux_c_max, num_pts)
    flux_prod = list(product(flux_q_linspace, flux_c_linspace))
    flux_q, flux_c = flux_prod[idx]
    FTC_grounded.flux_c = flux_c
    (E00, E00_4, E_00_2) = FTC_grounded.fourth_order_energy_shift(
        0, 0, flux_a=0.5 + flux_q, flux_b=0.5 - flux_q, highest_exc_q=highest_exc_q, highest_exc_m=highest_exc_m,
        highest_exc_p=highest_exc_p, num_cpus=num_cpus)
    (E01, E01_4, E_01_2) = FTC_grounded.fourth_order_energy_shift(
        0, 1, flux_a=0.5 + flux_q, flux_b=0.5 - flux_q, highest_exc_q=highest_exc_q, highest_exc_m=highest_exc_m,
        highest_exc_p=highest_exc_p, num_cpus=num_cpus)
    (E10, E10_4, E_10_2) = FTC_grounded.fourth_order_energy_shift(
        1, 0, flux_a=0.5 + flux_q, flux_b=0.5 - flux_q, highest_exc_q=highest_exc_q, highest_exc_m=highest_exc_m,
        highest_exc_p=highest_exc_p, num_cpus=num_cpus)
    (E11, E11_4, E_11_2) = FTC_grounded.fourth_order_energy_shift(
        1, 1, flux_a=0.5 + flux_q, flux_b=0.5 - flux_q, highest_exc_q=highest_exc_q, highest_exc_m=highest_exc_m,
        highest_exc_p=highest_exc_p, num_cpus=num_cpus)
    data_dict = {"eta": E11 - E10 - E01 + E00}
    param_dict = {"highest_exc_q": highest_exc_q, "highest_exc_m": highest_exc_m, "highest_exc_p": highest_exc_p,
                  "flux_q_max": flux_q_max, "flux_c_max": flux_c_max, "idx": idx, "num_pts": num_pts}
    filepath = str(idx).zfill(5)+"_ZZ_shift_vs_flux.h5py"
    write_to_h5(filepath, data_dict, param_dict)


def write_to_h5(filepath, data_dict, param_dict, loud=True):
    if loud:
        print(f"writing data to {filepath}")
    with h5py.File(filepath, "w") as f:
        for key, val in data_dict.items():
            written_data = f.create_dataset(key, data=val)
        for kwarg in param_dict.keys():
            try:
                f.attrs[kwarg] = param_dict[kwarg]
            except TypeError:
                f.attrs[kwarg] = str(param_dict[kwarg])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ftc zz")
    parser.add_argument("--idx", default=0, type=int, help="index that is unraveled")
    parser.add_argument(
        "--num_pts", default=5, type=int, help="number of points in flux lists"
    )
    parser.add_argument("--num_cpus", default=8, type=int, help="num cpus")
    args = parser.parse_args()
    run_ftc_zz(args.idx, args.num_pts, num_cpus=args.num_cpus)
