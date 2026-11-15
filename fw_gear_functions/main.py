"""Main module."""
import logging

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)

# Region name -> segmentation label values it covers.
REGIONS = {
    "enhancing": (1,),
    "nonenhancing": (2,),
    "cyst": (3,),
    "edema": (4,),
    "wt": (1, 2, 3),  # whole tumor: enhancing + non-enhancing + cyst
}


def process(input_path, seg_filename):
    """Compute intensity statistics of the input image within each tumor region.

    Args:
        input_path (Path-like): Path to the image to measure.
        seg_filename (Path-like): Path to the segmentation (integer labels).

    Returns:
        dict: ``{"info": {"TumorSegIntensity": {...}}}`` holding
            ``<region>_mean``, ``<region>_median`` and ``<region>_stdev`` for
            every region that contains at least one voxel.
    """
    file_dictionary = {"TumorSegIntensity": {}}
    stats = file_dictionary["TumorSegIntensity"]

    im = nib.load(input_path).get_fdata()
    mask = np.rint(nib.load(seg_filename).get_fdata())

    for region, labels in REGIONS.items():
        voxels = im[np.isin(mask, labels)]
        if voxels.size == 0:
            continue
        stats[f"{region}_mean"] = np.mean(voxels)
        stats[f"{region}_median"] = np.median(voxels)
        stats[f"{region}_stdev"] = np.std(voxels)
        log.info(f"Added {region.upper()} statistics (labels {labels})")

    return {"info": file_dictionary}


def run(input_path, seg_filename):
    """Run the gear's processing; see `process`."""
    return process(input_path, seg_filename)
