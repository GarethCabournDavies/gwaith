"""Unit tests for the hwilas.data_processing.truth module."""

import argparse

import h5py
import numpy as np
import pytest

from hwilas.data_processing import truth


def test_add_truth_arguments():
    """Test that truth overlay options are added to the parser."""
    parser = argparse.ArgumentParser()
    truth.add_truth_arguments(parser)

    # Defaults: no truth file, no overlay.
    args = parser.parse_args([])
    assert args.truth_file is None
    assert args.plot_truth is False

    # Explicit values.
    args = parser.parse_args(["--truth-file", "truth.h5", "--plot-truth"])
    assert args.truth_file == "truth.h5"
    assert args.plot_truth is True


def test_load_truth_file_roundtrip(tmp_path):
    """Test that load_truth_file recovers a standardised truth file."""
    catalog = np.zeros(
        2,
        dtype=[("source_group", "S8"), ("m1", "<f8"), ("CoalescenceTime", "<f8")],
    )
    catalog["source_group"] = ["mbhb", "mbhb"]
    catalog["m1"] = [1.0, 3.0]
    catalog["CoalescenceTime"] = [2.0, 4.0]
    timeseries = {
        "LISA_A": np.array([0.0, 1.0, 2.0]),
        "LISA_E": np.array([3.0, 4.0, 5.0]),
    }
    original = truth.TruthData(
        source_group="mbhb",
        catalog=catalog,
        timeseries=timeseries,
        epoch=123.0,
        delta_t=5.0,
    )
    path = str(tmp_path / "truth.h5")
    original.to_hdf5(path)

    loaded = truth.load_truth_file(path)

    assert loaded.source_group == "mbhb"
    assert loaded.n_sources == 2
    assert np.allclose(loaded.coalescence_times(), [2.0, 4.0])
    assert set(loaded.timeseries.keys()) == {"LISA_A", "LISA_E"}
    assert np.allclose(loaded.timeseries["LISA_A"], timeseries["LISA_A"])
    assert loaded.epoch == 123.0
    assert loaded.delta_t == 5.0


def test_load_truth_file_invalid(tmp_path):
    """Test that load_truth_file rejects a file that is not standardised."""
    path = str(tmp_path / "not_truth.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("foo", data=[1, 2, 3])

    with pytest.raises(ValueError, match="does not look like a standardised truth file"):
        truth.load_truth_file(path)


def test_detect_truth_file_type_sangria(tmp_path):
    """A plain sky/<group> layout with no Mojito markers is detected as Sangria."""
    path = str(tmp_path / "sangria.h5")
    with h5py.File(path, "w") as f:
        sky = f.create_group("sky")
        mbhb = sky.create_group("mbhb")
        mbhb.create_dataset("cat", data=np.array([1.0], dtype="<f8"))

    assert truth.detect_truth_file_type(path) == "sangria"


def test_detect_truth_file_type_mojito(tmp_path):
    """A file carrying a Mojito-style metadata marker is detected as Mojito."""
    path = str(tmp_path / "mojito.h5")
    with h5py.File(path, "w") as f:
        f.create_group("meta")
        sky = f.create_group("sky")
        mbhb = sky.create_group("mbhb")
        mbhb.create_dataset("cat", data=np.array([1.0], dtype="<f8"))

    assert truth.detect_truth_file_type(path) == "mojito"


def _make_mojito_catalogue(path):
    """Write a small synthetic Mojito *catalogue* file (Binaries + Metadata)."""
    n = 3
    with h5py.File(path, "w") as f:
        binaries = f.create_group("Binaries")
        binaries.create_dataset(
            "PrimaryMassSourceFrame", data=np.array([10.0, 20.0, 30.0])
        )
        binaries.create_dataset(
            "SecondaryMassSourceFrame", data=np.array([5.0, 6.0, 7.0])
        )
        binaries.create_dataset(
            "TimeCoalescencePhenomTPHMSSBFrame",
            data=np.array([300.0, 611.0, 173.0]),
        )
        meta = f.create_group("Metadata")
        meta.attrs["SourceType"] = "MBHB"
        meta.attrs["NumberBinaries"] = n


def test_load_truth_mojito_catalogue(tmp_path):
    """Test loading a Mojito catalogue file (Binaries group layout)."""
    path = str(tmp_path / "mojito_cat.h5")
    _make_mojito_catalogue(path)

    loaded = truth.load_truth_mojito(path, source_group="mbhb")

    assert loaded.source_group == "mbhb"
    assert loaded.n_sources == 3
    assert loaded.timeseries is None
    assert "PrimaryMassSourceFrame" in loaded.catalog.dtype.names
    assert np.allclose(loaded.catalog["SecondaryMassSourceFrame"], [5.0, 6.0, 7.0])
    assert "CoalescenceTime" in loaded.catalog.dtype.names
    assert np.allclose(loaded.catalog["CoalescenceTime"], [300.0, 611.0, 173.0])
    assert np.allclose(loaded.coalescence_times(), [300.0, 611.0, 173.0])


def test_standardize_truth_catalog_prefers_ssb_masses_when_present():
    """Canonical mass fields should use SSB masses when the catalog provides them."""
    catalog = np.zeros(
        2,
        dtype=[
            ("PrimaryMassSourceFrame", "<f8"),
            ("SecondaryMassSourceFrame", "<f8"),
            ("PrimaryMassSSBFrame", "<f8"),
            ("SecondaryMassSSBFrame", "<f8"),
            ("TimeCoalescencePhenomTPHMSSBFrame", "<f8"),
        ],
    )
    catalog["PrimaryMassSourceFrame"] = [10.0, 20.0]
    catalog["SecondaryMassSourceFrame"] = [4.0, 8.0]
    catalog["PrimaryMassSSBFrame"] = [100.0, 200.0]
    catalog["SecondaryMassSSBFrame"] = [40.0, 80.0]
    catalog["TimeCoalescencePhenomTPHMSSBFrame"] = [1.0, 2.0]

    standardized = truth.standardize_truth_catalog(catalog)

    assert "mass1" in standardized.dtype.names
    assert "mass2" in standardized.dtype.names
    np.testing.assert_allclose(standardized["mass1"], [100.0, 200.0])
    np.testing.assert_allclose(standardized["mass2"], [40.0, 80.0])


def test_load_truth_mojito_converts_distance_from_pc_to_mpc(tmp_path):
    """Mojito distance fields with pc units should be converted to Mpc."""
    path = str(tmp_path / "mojito_pc_distance.h5")
    with h5py.File(path, "w") as f:
        binaries = f.create_group("Binaries")
        binaries.create_dataset(
            "PrimaryMassSourceFrame",
            data=np.array([10.0, 20.0]),
        )
        binaries.create_dataset(
            "SecondaryMassSourceFrame",
            data=np.array([5.0, 6.0]),
        )
        binaries.create_dataset(
            "TimeCoalescencePhenomTPHMSSBFrame",
            data=np.array([100.0, 200.0]),
        )
        dset = binaries.create_dataset(
            "LuminosityDistance",
            data=np.array([1.0e6, 2.5e6]),
        )
        dset.attrs["unit"] = "pc"

    loaded = truth.load_truth_mojito(path, source_group="mbhb")
    assert "distance" in loaded.catalog.dtype.names
    np.testing.assert_allclose(loaded.catalog["distance"], [1.0, 2.5])


def test_load_truth_mojito_missing_binaries(tmp_path):
    """Test that a file without a Binaries group raises a clear error."""
    path = str(tmp_path / "not_catalogue.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("foo", data=[1, 2, 3])

    with pytest.raises(ValueError, match="no 'Binaries' group found"):
        truth.load_truth_mojito(path)


def test_standardize_truth_catalog_adds_canonical_spin_fields():
    """Catalogs should expose common spin aliases regardless of source layout."""
    catalog = np.zeros(
        2,
        dtype=[
            ("Mass1", "<f8"),
            ("Mass2", "<f8"),
            ("CoalescenceTime", "<f8"),
            ("EclipticLongitude", "<f8"),
            ("EclipticLatitude", "<f8"),
            ("Spin1", "<f8"),
            ("PolarAngleOfSpin1", "<f8"),
            ("Spin2", "<f8"),
            ("PolarAngleOfSpin2", "<f8"),
        ],
    )
    catalog["Mass1"] = [10.0, 20.0]
    catalog["Mass2"] = [5.0, 9.0]
    catalog["CoalescenceTime"] = [100.0, 200.0]
    catalog["EclipticLongitude"] = [0.1, 0.2]
    catalog["EclipticLatitude"] = [-0.3, 0.4]
    catalog["Spin1"] = [0.7, 0.5]
    catalog["PolarAngleOfSpin1"] = [0.0, np.pi / 3]
    catalog["Spin2"] = [0.6, 0.4]
    catalog["PolarAngleOfSpin2"] = [np.pi / 4, np.pi / 6]

    standardized = truth.standardize_truth_catalog(catalog)

    assert "tc" in standardized.dtype.names
    assert "mass1" in standardized.dtype.names
    assert "mass2" in standardized.dtype.names
    assert "chirp_mass" in standardized.dtype.names
    assert "eclipticlongitude" in standardized.dtype.names
    assert "eclipticlatitude" in standardized.dtype.names
    assert "spin1z" in standardized.dtype.names
    assert "spin2z" in standardized.dtype.names

    np.testing.assert_allclose(standardized["spin1z"], [0.7, 0.5 * np.cos(np.pi / 3)])
    np.testing.assert_allclose(standardized["spin2z"], [0.6 * np.cos(np.pi / 4), 0.4 * np.cos(np.pi / 6)])
    assert "chi1z" in standardized.dtype.names
    assert "chi2z" in standardized.dtype.names


def test_load_truth_mojito_requires_phenom_ssb_tc(tmp_path):
    """Mojito inputs must use the exact Phenom SSB coalescence-time column."""
    path = str(tmp_path / "mojito_missing_tc.h5")
    with h5py.File(path, "w") as f:
        binaries = f.create_group("Binaries")
        binaries.create_dataset(
            "TimeCoalescencePetersSourceFrame",
            data=np.array([45.0, 126.0, 49.0]),
        )
        binaries.create_dataset(
            "PrimaryMassSourceFrame",
            data=np.array([10.0, 20.0, 30.0]),
        )
        binaries.create_dataset(
            "SecondaryMassSourceFrame",
            data=np.array([5.0, 6.0, 7.0]),
        )

    with pytest.raises(ValueError, match="TimeCoalescencePhenomTPHMSSBFrame"):
        truth.load_truth_mojito(path, source_group="mbhb")


def _make_sangria_catalogue(path, *, tc_name="CoalescenceTime", mass1_name="Mass1", mass2_name="Mass2"):
    """Write a small synthetic Sangria catalogue file under sky/<group>/cat."""
    arr = np.array(
        [
            (10.0, 5.0, 100.0),
            (20.0, 6.0, 200.0),
            (30.0, 7.0, 300.0),
        ],
        dtype=[(mass1_name, "<f8"), (mass2_name, "<f8"), (tc_name, "<f8")],
    )
    with h5py.File(path, "w") as f:
        sky = f.create_group("sky")
        mbhb = sky.create_group("mbhb")
        mbhb.create_dataset("cat", data=arr)


def test_load_truth_sangria_catalogue(tmp_path):
    """Test loading a Sangria file with the exact expected fields."""
    path = str(tmp_path / "sangria_cat.h5")
    _make_sangria_catalogue(path)

    loaded = truth.load_truth_sangria(path, source_group="mbhb")

    assert loaded.source_group == "mbhb"
    assert loaded.n_sources == 3
    assert np.allclose(loaded.coalescence_times(), [100.0, 200.0, 300.0])


def test_load_truth_sangria_missing_required_fields(tmp_path):
    """Sangria inputs must declare the exact expected raw fields."""
    path = str(tmp_path / "sangria_missing_fields.h5")
    _make_sangria_catalogue(path, tc_name="TimeCoalescencePhenomTPHMSSBFrame")

    with pytest.raises(ValueError, match="CoalescenceTime"):
        truth.load_truth_sangria(path, source_group="mbhb")