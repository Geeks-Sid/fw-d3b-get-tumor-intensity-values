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
