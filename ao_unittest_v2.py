#!/usr/bin/env python3
"""
Script that tests multiple first-row transition metals (Cr, Mn2+, Ni, Fe2+)
in a modular way and in multiple basis sets (sto-3g, 3-21g):
  1) Defines systems in a dictionary with {atom, spin, charge}.
  2) For each system (Cr, Mn, Ni, Fe), runs an ROHF calc, does meta-Löwdin
     analysis, builds threshold-based orbital labels in both sto-3g and 3-21g.
  3) Generates cube files for orbitals with frac_3d >= 0.75.
  4) Optionally calls GPT to produce labels and compares them with threshold
     labels in a professional unittest using pass/fail counts.

Requires:
  - pyscf
  - openai>=1.0.0
  - python-dotenv (if using .env for OPENAI_API_KEY)
"""

import os
import re
import unittest
import numpy as np
from functools import reduce

from pyscf import gto, scf, lo
from pyscf.tools import cubegen  # to generate .cube files

try:
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()  # loads .env if present
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    USE_GPT = True if OPENAI_API_KEY else False
except ImportError:
    USE_GPT = False


############################################
# 1) Systems dictionary, plus a list of basis sets
############################################
SYSTEMS = {
    # Chromium, high spin => d^5, 4s^1 => spin=5
    "Cr_highspin": {
        "atom": "Cr",
        "spin": 6,
        "charge": 0,
    },
    # Mn2+ high spin => d^5 => spin=5
    "Mn2_highspin": {
        "atom": "Mn",
        "spin": 5,
        "charge": 2,
    },
    # Ni closed-shell => d^8, 4s^2 => spin=0 but here spin=2 for "high spin"
    "Ni2_highspin": {
        "atom": "Ni",
        "spin": 2,
        "charge": 2,
    },
    # Fe2+ high spin => typically d^6 => spin=4
    "Fe2_highspin": {
        "atom": "Fe",
        "spin": 4,
        "charge": 2,
    },
}

BASIS_SETS = ["sto-3g", "3-21g"]


############################################
# 2) Common logic: run calculation & label
############################################
def run_calc(system_key, basis, threshold_3d=0.75):
    """
    1) Build + run an ROHF for the specified system in SYSTEMS with the given basis.
    2) Perform meta-Löwdin, compute fraction of 3d, pick largest AO or 3d shape.
    3) Return:
       mo_list (list of dicts with threshold labels),
       mo_coeff_meta,
       ao_labels,
       mol,
       mo_coeff
    """
    params = SYSTEMS[system_key]

    # 1) Build the molecule
    mol = gto.Mole()
    mol.atom = [[params["atom"], (0., 0., 0.)]]
    mol.spin = params["spin"]
    mol.charge = params["charge"]
    mol.basis = basis  # use the basis passed in
    mol.build()

    # 2) SCF
    mf = scf.ROHF(mol)
    mf.verbose = 4
    mf.max_cycle = 200
    mf.kernel()

    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ
    mo_coeff = mf.mo_coeff
    nmo = len(mo_energy)

    # 3) Meta-Löwdin
    S = mf.get_ovlp()
    U_meta = lo.orth_ao(mol, 'meta_lowdin')
    mo_coeff_meta = U_meta.T @ mo_coeff

    # 4) Identify "X 3d" AOs, e.g. "Fe 3d", "Cr 3d", etc.
    ao_labels = mol.ao_labels()
    d_indices = []
    d_map = {}
    shape_order = ["3dxy", "3dyz", "3dz^2", "3dxz", "3dx2-y2"]

    for i, lbl in enumerate(ao_labels):
        if params["atom"] in lbl and "3d" in lbl:
            d_indices.append(i)
            # if the label has "3dxy", "3dxz", etc.
            if "3dxy" in lbl:
                d_map["3dxy"] = i
            elif "3dyz" in lbl:
                d_map["3dyz"] = i
            elif "3dz^2" in lbl or "3dz2" in lbl:
                d_map["3dz^2"] = i
            elif "3dxz" in lbl:
                d_map["3dxz"] = i
            elif "3dx2-y2" in lbl or "3dx2y2" in lbl:
                d_map["3dx2-y2"] = i

    mo_list = []
    for j in range(nmo):
        frac_3d = np.sum(mo_coeff_meta[d_indices, j]**2)

        # threshold logic
        if frac_3d >= threshold_3d:
            # pick largest single 3d shape
            largest_val = 0.0
            largest_shape = "3d_mixed"
            for shp in shape_order:
                if shp in d_map:
                    idx = d_map[shp]
                    c_val = abs(mo_coeff_meta[idx, j])
                    if c_val > largest_val:
                        largest_val = c_val
                        largest_shape = shp
            shape_label = largest_shape
        else:
            # pick largest AO overall
            largest_val = 0.0
            largest_lbl = None
            for i_ao, ao_lbl in enumerate(ao_labels):
                c_val = abs(mo_coeff_meta[i_ao, j])
                if c_val > largest_val:
                    largest_val = c_val
                    largest_lbl = ao_lbl
            # parse out a short label from e.g. "0 Fe 4s" => "4s"
            shape_label = largest_lbl.split()[-1]

        # build final label
        e_str = f"{mo_energy[j]:.3f}"
        frac_str = f"{100.0 * frac_3d:.1f}"
        occ_str = f"{mo_occ[j]:.1f}"
        label_str = f"{shape_label}, E={e_str}, {frac_str}, Occ={occ_str}"

        mo_list.append({
            "index": j,
            "energy": mo_energy[j],
            "occ": mo_occ[j],
            "frac_3d": frac_3d,
            "threshold_label": label_str
        })

    return mo_list, mo_coeff_meta, ao_labels, mol, mo_coeff


############################################
# 3) Generate cube files for 3d orbitals
############################################
def generate_3d_cubes(system_key, mo_list, mo_coeff, mol, threshold_3d=0.75):
    """
    For each MO in mo_list with frac_3d >= threshold_3d, generate a .cube file.
    Name them e.g. {system_key}_mo_{j}.cube
    """
    if not os.path.exists("cube_outputs"):
        os.mkdir("cube_outputs")

    for mo in mo_list:
        if mo["frac_3d"] >= threshold_3d:
            j = mo["index"]
            cube_name = f"{system_key}_mo_{j}.cube"
            out_path = os.path.join("cube_outputs", cube_name)
            print(f"Generating cube for MO #{j}: {out_path}")
            cubegen.orbital(mol, out_path, mo_coeff[:, j], nx=60, ny=60, nz=60)


############################################
# 4) GPT-based label
############################################
def build_full_ao_vector_string(mo_index, mo_coeff_meta, ao_labels):
    """
    Return multiline string of entire AO vector in the format:
       "0 Fe 1s:  0.001
        0 Fe 2s:  0.999
        ..."
    """
    ao_vec = mo_coeff_meta[:, mo_index]
    lines = []
    for i, lbl in enumerate(ao_labels):
        val = ao_vec[i]
        lines.append(f"{lbl}: {val:.3f}")
    return "\n".join(lines)

def gpt_label_mo(mo_info, mo_coeff_meta, ao_labels, client, system_key):
    """
    Produce a label in the format "<shape>, E=X.XXX, YYY.Y, Occ=Z.Z".
    We'll pass the entire AO vector, plus the threshold shape in 'Shape='.
    """
    system_prompt = (
        "You are a quantum chemistry expert labeling orbitals based on the "
        "molecular orbital energies and their AO vectors in the Meta-Lowdin basis. "
        "Output exactly one line in the format:\n"
        "<shape>, E=X.XXX, YYY.Y, Occ=Z.Z\n"
        "No extra lines or commentary."
    )

    ao_vec_str = build_full_ao_vector_string(mo_info["index"], mo_coeff_meta, ao_labels)
    user_msg = (
        f"System = {system_key}\n"
        f"Energy = {mo_info['energy']:.3f}\n"
        f"Occ = {mo_info['occ']:.1f}\n"
        f"FractionOf3d = {mo_info['frac_3d']*100:.1f}\n"
        f"Shape = {mo_info['threshold_label'].split(',')[0]}\n"
        "Output single line: <shape>, E=X.XXX, YYY.Y, Occ=Z.Z, where YYY.Y is the "
        "fraction of 3d orbitals in the MO FractionOf3d.\n"
    )

    resp = client.chat.completions.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0.0,
        max_tokens=100
    )
    return resp.choices[0].message.content.strip()


############################################
# 5) Parsing & Comparison
############################################
def parse_label_fields(label_str):
    """
    Expected format: "<shape>, E=<energy>, <frac>, Occ=<occ>"
    e.g. "3dxz, E=-0.431, 100.0, Occ=2.0"
    Returns a dict: { "shape":str, "energy":float, "frac":float, "occ":float }
    """
    pattern = r"^\s*([^,]+),\s*E=([-\d.]+),\s*([-\d.]+),\s*Occ=([-\d.]+)\s*$"
    m = re.match(pattern, label_str)
    if not m:
        raise ValueError(f"Label '{label_str}' doesn't match format <shape>, E=X.XXX, YYY.Y, Occ=Z.Z")
    out = {}
    out["shape"]  = m.group(1).strip()
    out["energy"] = float(m.group(2))
    out["frac"]   = float(m.group(3))
    out["occ"]    = float(m.group(4))
    return out

def compare_label_data(tdata, gdata, energy_tol=0.01, frac_tol=1.0, occ_tol=0.1):
    """
    Compare shape strings exactly. Compare numeric fields with tolerances.
    Return True if "similar enough," else False.
    """
    # shape must match exactly
    if tdata["shape"] != gdata["shape"]:
        return False
    # numeric comparisons
    if abs(tdata["energy"] - gdata["energy"]) > energy_tol:
        return False
    if abs(tdata["frac"] - gdata["frac"]) > frac_tol:
        return False
    if abs(tdata["occ"] - gdata["occ"]) > occ_tol:
        return False
    return True


############################################
# 6) The unittest class
############################################
class TestFirstRowTransitionMetals(unittest.TestCase):

    def _test_system(self, system_key, basis):
        """
        1) Run the calculation & labeling for 'system_key' with the given basis.
        2) Generate .cube for orbitals with frac_3d>=0.75.
        3) If GPT is available, compare threshold vs GPT labels.
        """
        print(f"\n=== Testing system: {system_key}, basis={basis} ===")

        # 1) Run the calc
        mo_list, mo_coeff_meta, ao_labels, mol, mo_coeff = run_calc(system_key, basis)

        # 2) Generate cube files for orbitals with fraction_3d >= 0.75
        generate_3d_cubes(system_key, mo_list, mo_coeff, mol, threshold_3d=0.75)

        # 3) GPT labeling comparison
        if not USE_GPT:
            print("GPT not available, skipping label similarity test.")
            return

        client = OpenAI(api_key=OPENAI_API_KEY)
        pass_count = 0
        fail_count = 0
        total = len(mo_list)

        for mo in mo_list:
            thr_label = mo["threshold_label"]
            try:
                tdata = parse_label_fields(thr_label)
            except ValueError as e:
                # If threshold label is invalid, fail the entire test
                self.fail(f"Threshold label parse error on {system_key} MO#{mo['index']}: {e}")

            # GPT label
            gpt_label = gpt_label_mo(mo, mo_coeff_meta, ao_labels, client, system_key)
            try:
                gdata = parse_label_fields(gpt_label)
            except ValueError as e:
                self.fail(f"GPT label parse error on {system_key} MO#{mo['index']}: label='{gpt_label}', error={e}")

            print(f"  MO #{mo['index']:2d}: threshold='{thr_label}', gpt='{gpt_label}'")

            if compare_label_data(tdata, gdata):
                pass_count += 1
            else:
                fail_count += 1
                print(f"*** Mismatch for {system_key} MO #{mo['index']}: thr='{thr_label}', gpt='{gpt_label}'")

        print(f"Test results for {system_key} with {basis}: {pass_count} passed, {fail_count} failed, total={total}.")
        # If any fail => overall test fails
        self.assertEqual(fail_count, 0, f"{fail_count} mismatch(es) in {system_key} [{basis}]")

    def test_cr_highspin(self):
        for basis in BASIS_SETS:
            with self.subTest(basis=basis):
                self._test_system("Cr_highspin", basis)

    def test_mn2_highspin(self):
        for basis in BASIS_SETS:
            with self.subTest(basis=basis):
                self._test_system("Mn2_highspin", basis)

    def test_ni_highspin(self):
        for basis in BASIS_SETS:
            with self.subTest(basis=basis):
                self._test_system("Ni2_highspin", basis)

    def test_fe2_highspin(self):
        for basis in BASIS_SETS:
            with self.subTest(basis=basis):
                self._test_system("Fe2_highspin", basis)


if __name__ == "__main__":
    unittest.main()
