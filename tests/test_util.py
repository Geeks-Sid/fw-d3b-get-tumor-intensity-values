"""Tests for fw_gear_functions.util."""
import pytest

from fw_gear_functions.util import find_segmentation


def test_find_segmentation_returns_the_only_nifti_manualseg():
    names = ["t1.nii.gz", "t1_ManualSeg.nii.gz", "t1_ManualSeg.json", "notes.txt"]
    assert find_segmentation(names, "t1.nii.gz") == "t1_ManualSeg.nii.gz"


def test_find_segmentation_accepts_uncompressed_nifti():
    assert find_segmentation(["a.nii", "a_ManualSeg.nii"], "a.nii") == "a_ManualSeg.nii"


def test_find_segmentation_never_returns_the_input_itself():
    """Running the gear on the segmentation must not measure it against itself."""
    with pytest.raises(ValueError, match="found 0"):
        find_segmentation(["a_ManualSeg.nii.gz"], "a_ManualSeg.nii.gz")


def test_find_segmentation_without_candidate_raises():
    with pytest.raises(ValueError, match="found 0"):
        find_segmentation(["t1.nii.gz"], "t1.nii.gz")


def test_find_segmentation_with_several_candidates_raises():
    """Previously the last match silently won, depending on listing order."""
    names = ["t1.nii.gz", "a_ManualSeg.nii.gz", "b_ManualSeg.nii.gz"]
    with pytest.raises(ValueError, match="found 2"):
        find_segmentation(names, "t1.nii.gz")
