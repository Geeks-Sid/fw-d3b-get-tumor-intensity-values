"""Main module."""
import logging

import nibabel as nib
import numpy as np

from .util import validate_file

log = logging.getLogger(__name__)

# Region name -> segmentation label values it covers.
REGIONS = {
    "enhancing": (1,),
    "nonenhancing": (2,),
    "cyst": (3,),
    "edema": (4,),
    "wt": (1, 2, 3),  # whole tumor: enhancing + non-enhancing + cyst
}
KNOWN_LABELS = {0, *(label for labels in REGIONS.values() for label in labels)}


def process(input_path, seg_filename):
    """Compute intensity statistics of the input image within each tumor region.

    Args:
        input_path (Path-like): Path to the image to measure.
        seg_filename (Path-like): Path to the segmentation (integer labels).

    Returns:
        dict: ``{"info": {"TumorSegIntensity": {...}}}`` holding
            ``<region>_mean``, ``<region>_median`` and ``<region>_stdev`` for
            every region in REGIONS. A region with no voxels gets ``None`` so a
            value stored by an earlier run is overwritten, not left stale.

    Raises:
        ValueError: If image and segmentation shapes differ, or the image has
            NaN/Inf voxels inside a segmented region.
    """
    file_dictionary = {"TumorSegIntensity": {}}
    stats = file_dictionary["TumorSegIntensity"]

    img = nib.load(input_path)
    seg = nib.load(seg_filename)
    if img.shape != seg.shape:
        raise ValueError(
            f"Image shape {img.shape} does not match segmentation shape {seg.shape}; "
            "the segmentation must be on the same voxel grid as the image."
        )
    if not np.allclose(img.affine, seg.affine, atol=1e-3):
        log.warning(
            "Image and segmentation affines differ; voxel-wise statistics may be "
            "taken from spatially misaligned voxels."
        )

    im = img.get_fdata()
    mask = np.rint(seg.get_fdata())

    unexpected = sorted(set(np.unique(mask).tolist()) - KNOWN_LABELS)
    if unexpected:
        log.warning(f"Ignoring unexpected segmentation labels: {unexpected}")

    for region, labels in REGIONS.items():
        voxels = im[np.isin(mask, labels)]
        if voxels.size == 0:
            log.info(f"No {region.upper()} voxels (labels {labels}); statistics set to None")
            stats.update({f"{region}_{s}": None for s in ("mean", "median", "stdev")})
            continue
        if not np.isfinite(voxels).all():
            raise ValueError(f"Image has NaN/Inf voxels inside the {region.upper()} region.")
        stats[f"{region}_mean"] = float(np.mean(voxels))
        stats[f"{region}_median"] = float(np.median(voxels))
        stats[f"{region}_stdev"] = float(np.std(voxels))
        log.info(f"Added {region.upper()} statistics (labels {labels})")

    return {"info": file_dictionary}


def run(input_path, seg_filename):
    """Validate both files, then compute the statistics; see `process`.

    Raises:
        ValueError: If either file is empty, or `process` rejects the data.
    """
    errors = validate_file(input_path) + validate_file(seg_filename)
    if errors:
        raise ValueError("; ".join(errors))
    return process(input_path, seg_filename)
