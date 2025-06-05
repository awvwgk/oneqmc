import json
import os
import urllib.request

import numpy as np

from oneqmc.molecule import Molecule

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), os.pardir, "data")
    reference_energy_dir = os.path.join(
        os.path.dirname(__file__), os.pardir, "experiment_results", "03_tinymol", "references"
    )

    print("Download data from https://raw.githubusercontent.com/mdsunivie/deeperwin.")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/mdsunivie/deeperwin/refs/heads/master/datasets/db/geometries.json",
        os.path.join(data_dir, "geometries.json"),
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/mdsunivie/deeperwin/refs/heads/master/datasets/db/datasets.json",
        os.path.join(data_dir, "datasets.json"),
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/mdsunivie/deeperwin/refs/heads/master/datasets/db/energies.csv",
        os.path.join(reference_energy_dir, "tinymol_deeperwin.csv"),
    )
    print("Downloaded.")
    print(f"DeepErwin baseline results were stored in {reference_energy_dir}/tinymol_deeperwin.csv")
    with open(os.path.join(data_dir, "datasets.json")) as f:
        datasets = json.load(f)
    with open(os.path.join(data_dir, "geometries.json")) as f:
        geometries = json.load(f)

    keys = [
        "TinyMol_CNO_rot_dist_test_in_distribution_30geoms",
        "TinyMol_CNO_rot_dist_test_out_of_distribution_40geoms",
        "TinyMol_CNO_rot_dist_train_18compounds_360geoms",
    ]

    for key in keys:
        os.makedirs(os.path.join(data_dir, key), exist_ok=True)
        for subkey in datasets[key]["datasets"]:
            collected = {}
            for geom in datasets[subkey]["geometries"]:
                geo_hash, geo_name = geom.split("__")
                geometry_data = geometries[geo_hash]
                mol = Molecule.make(
                    coords=np.asarray(geometry_data["R"]),
                    charges=np.asarray(geometry_data["Z"]),
                    charge=0,
                    spin=0,
                    unit="bohr",
                )
                dict_repr = mol.to_qcelemental().dict(encoding="json")
                collected[geo_name] = dict_repr
            f_out_name = os.path.join(data_dir, key, f"{subkey}.json")
            print(f"Writing {f_out_name} with {len(collected)} structures.")
            with open(f_out_name, "w") as f_out:
                json.dump(collected, f_out)

    print("Removing cached files.")
    os.remove(os.path.join(data_dir, "geometries.json"))
    os.remove(os.path.join(data_dir, "datasets.json"))
    print("Done. The following datasets are ready:", ", ".join(keys))
