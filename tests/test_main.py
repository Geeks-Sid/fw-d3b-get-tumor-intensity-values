"""Tests for fw_gear_functions.main."""
import nibabel as nib
import numpy as np
import pytest

from fw_gear_functions.main import run

SHAPE = (4, 4, 4)


def save(path, data, affine=None):
    """Write `data` as a NIfTI file and return its path as a string."""
    nib.save(nib.Nifti1Image(data, np.eye(4) if affine is None else affine), path)
    return str(path)


@pytest.fixture
def make_inputs(tmp_path):
    """Return a function writing an image + segmentation pair to tmp_path."""

    def _make(image, seg, **kwargs):
        return (
            save(tmp_path / "image.nii.gz", np.asarray(image, dtype=np.float64)),
            save(tmp_path / "ManualSeg.nii.gz", np.asarray(seg, dtype=np.uint8), **kwargs),
        )

    return _make


def test_run_works_outside_a_gear_container(make_inputs):
    """Importing the module must not need /flywheel/v0/config.json."""
    image = np.zeros(SHAPE)
    seg = np.zeros(SHAPE)
    image[0, 0, :3] = [10, 20, 60]
    seg[0, 0, :3] = 1

    info = run(*make_inputs(image, seg))["info"]["TumorSegIntensity"]

    assert info["enhancing_mean"] == pytest.approx(30)
    assert info["enhancing_median"] == pytest.approx(20)
    assert info["enhancing_stdev"] == pytest.approx(np.std([10, 20, 60]))


def test_fractional_intensities_are_not_rounded(make_inputs):
    """Statistics of e.g. normalised or ADC images must keep their decimals."""
    image = np.zeros(SHAPE)
    seg = np.zeros(SHAPE)
    image[0, 0, :3] = [0.2, 0.4, 0.6]
    seg[0, 0, :3] = 1

    info = run(*make_inputs(image, seg))["info"]["TumorSegIntensity"]

    assert info["enhancing_mean"] == pytest.approx(0.4)
    assert info["enhancing_median"] == pytest.approx(0.4)


def test_zero_intensity_voxels_inside_the_mask_are_counted(make_inputs):
    """Only the mask decides membership; a 0-valued voxel is still tumor."""
    image = np.zeros(SHAPE)
    seg = np.zeros(SHAPE)
    image[0, 0, :4] = [0, 0, 10, 20]
    seg[0, 0, :4] = 2

    info = run(*make_inputs(image, seg))["info"]["TumorSegIntensity"]

    assert info["nonenhancing_mean"] == pytest.approx(7.5)
    assert info["nonenhancing_median"] == pytest.approx(5)


def test_regions_use_their_own_labels(make_inputs):
    """1=enhancing, 2=non-enhancing, 3=cyst, 4=edema, whole tumor=1+2+3 (no edema)."""
    image = np.zeros(SHAPE)
    seg = np.zeros(SHAPE)
    image.flat[:5] = [10, 20, 30, 40, 1000]
    seg.flat[:5] = [1, 2, 3, 4, 0]

    info = run(*make_inputs(image, seg))["info"]["TumorSegIntensity"]

    assert info["enhancing_mean"] == 10
    assert info["nonenhancing_mean"] == 20
    assert info["cyst_mean"] == 30
    assert info["edema_mean"] == 40
    assert info["wt_mean"] == pytest.approx(20)
    assert info["wt_median"] == pytest.approx(20)
