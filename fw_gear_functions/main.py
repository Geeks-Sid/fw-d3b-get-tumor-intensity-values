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
            every region in REGIONS. A region with no voxels gets ``None`` so a
            value stored by an earlier run is overwritten, not left stale.
    """
    file_dictionary = {"TumorSegIntensity": {}}
    stats = file_dictionary["TumorSegIntensity"]

    im = nib.load(input_path).get_fdata()
    mask = np.rint(nib.load(seg_filename).get_fdata())

    for region, labels in REGIONS.items():
        voxels = im[np.isin(mask, labels)]
        if voxels.size == 0:
            log.info(f"No {region.upper()} voxels (labels {labels}); statistics set to None")
            stats.update({f"{region}_{s}": None for s in ("mean", "median", "stdev")})
            continue
        stats[f"{region}_mean"] = float(np.mean(voxels))
        stats[f"{region}_median"] = float(np.median(voxels))
        stats[f"{region}_stdev"] = float(np.std(voxels))
        log.info(f"Added {region.upper()} statistics (labels {labels})")

    return {"info": file_dictionary}


def run(input_path, seg_filename):
    """Run the gear's processing; see `process`."""
    return process(input_path, seg_filename)
