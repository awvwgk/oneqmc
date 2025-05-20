import os
import subprocess
from pathlib import Path

import pytest


class OneQMCProcessError(Exception):
    pass


@pytest.fixture(scope="class")
def project_root():
    return Path(os.path.realpath(os.path.dirname(__file__))) / "../.."


@pytest.fixture(scope="class")
def train_workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("train_workdir")


@pytest.fixture(scope="class")
def run_transferable_script(project_root):
    fix_args = [
        "-d",
        "integration_test-B",
        "-a",
        "envnet",
        "--electron-batch-size",
        "2",
        "--mol-batch-size",
        "1",
        "--pretrain-steps",
        "1",
        "--max-eq-steps",
        "200",
        "--train-steps",
        "2",
        "--chkpts-fast-interval",
        "1",
        "--mcmc-n-block",
        "2",
    ]

    def runner(extra_args):
        result = subprocess.run(
            [
                "python",
                "scripts/transferable.py",
                *fix_args,
                *extra_args,
            ],
            cwd=project_root,
            capture_output=True,
        )
        if result.returncode != 0:
            raise OneQMCProcessError(result.stderr.decode())
        return result

    return runner


@pytest.fixture(scope="class")
def run_density_script(project_root):
    fix_args = [
        "-d",
        "integration_test-B",
        "-a",
        "envnet",
        "--electron-batch-size",
        "2",
        "--max-eq-steps",
        "200",
        "--decorr-steps",
        "20",
        "--train-steps",
        "2",
        "--chkpts-fast-interval",
        "1",
        "--save-grid-levels",
    ]

    def runner(extra_args):
        result = subprocess.run(
            [
                "python",
                "scripts/density.py",
                *fix_args,
                *extra_args,
            ],
            cwd=project_root,
            capture_output=True,
        )
        if result.returncode != 0:
            raise OneQMCProcessError(result.stderr.decode())
        return result

    return runner


class TestScripts:
    @pytest.mark.parametrize(
        "extra_args",
        [
            ["--multi-system-sampler", "double-langevin", "--no-autoresume"],
            ["--multi-system-sampler", "stack", "--no-autoresume"],
        ],
    )
    def test_train(self, run_transferable_script, train_workdir, extra_args):
        result = run_transferable_script(["--workdir", train_workdir] + extra_args)
        files = os.listdir(train_workdir)
        assert "oneqmc_train.log" in files
        assert "training" in files
        train_files = os.listdir(train_workdir / "training")
        assert "result.h5" in train_files
        assert "chkpt--1.pt" in train_files
        assert any(f.startswith("events.out.tfevents.") for f in train_files)
        assert "Supervised pretraining completed" in result.stdout.decode()
        assert "Initialising and preparing sampler" in result.stdout.decode()
        assert "Start training" in result.stdout.decode()
        assert "The training has been completed!" in result.stdout.decode()

    def test_restart(self, run_transferable_script, train_workdir, tmpdir):
        result = run_transferable_script(
            [
                "--workdir",
                tmpdir,
                "--chkpt",
                train_workdir / "training/chkpt--3.pt",
                "--no-autoresume",  # otherwise it will autoload most recent
            ]
        )
        files = os.listdir(tmpdir)
        assert "oneqmc_train.log" in files
        assert "training" in files
        train_files = os.listdir(tmpdir / "training")
        assert "result.h5" in train_files
        assert "chkpt--1.pt" in train_files
        assert any(f.startswith("events.out.tfevents.") for f in train_files)
        assert "Restart training from step 1" in result.stdout.decode()
        assert "The training has been completed!" in result.stdout.decode()

    def test_finetune(self, run_transferable_script, train_workdir, tmpdir):
        result = run_transferable_script(
            [
                "--workdir",
                tmpdir,
                "--chkpt",
                train_workdir / "training/chkpt--1.pt",
                "--orb-parameter-mode",
                "fine-tune",
                "--discard-sampler-state",
                "--no-autoresume",  # otherwise it will autoload most recent
            ]
        )
        files = os.listdir(tmpdir)
        assert "oneqmc_train.log" in files
        assert "training" in files
        train_files = os.listdir(tmpdir / "training")
        assert "result.h5" in train_files
        assert any(f.startswith("events.out.tfevents.") for f in train_files)
        assert (
            "Restart training from step 0" in result.stdout.decode()
        )  # due to --discard-sampler-state
        assert "Initialising fine-tune mode." in result.stdout.decode()
        assert "The training has been completed!" in result.stdout.decode()

    def test_density(self, run_density_script, train_workdir, tmpdir):
        result = run_density_script(
            [
                "--workdir",
                tmpdir,
                "--chkpt",
                train_workdir / "training/chkpt--1.pt",
            ]
        )
        files = os.listdir(tmpdir)
        assert "density" in files
        density_files = os.listdir(tmpdir / "density")
        assert "chkpt--1.pt" in density_files
        assert "Completed equilibration" in result.stdout.decode()
