"""Util module."""
import os
import typing as t
from pathlib import Path

from flywheel_gear_toolkit import GearToolkitContext

AnyPath = t.Union[str, Path]


def validate_file(filepath: AnyPath) -> t.List[str]:
    """Returns a list of validation errors if any."""
    errors = []
    errors += validate_file_size(filepath)
    return errors


def validate_file_size(filepath: AnyPath) -> t.List[str]:
    """Returns a list of validation errors related to file size."""
    errors = []
    if not os.path.getsize(filepath) > 1:
        errors.append("File is empty: {}".format(filepath))
    return errors


def find_segmentation(file_names: t.Iterable[str], input_name: str) -> str:
    """Return the name of the one NIfTI "ManualSeg" file among `file_names`.

    The input file itself is never a candidate.

    Raises:
        ValueError: If there is no candidate, or more than one (ambiguous).
    """
    candidates = [
        name
        for name in file_names
        if "ManualSeg" in name
        and name.endswith((".nii", ".nii.gz"))
        and name != input_name
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one NIfTI file with 'ManualSeg' in its name next to "
            f"{input_name}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def create_metadata(context: GearToolkitContext, fe):
    """Populate .metadata.json.

    Args:
        context (GearToolkitContext): The gear context.
        fe (dict): A dictionary containing the file attributes to update.
    """
    file_input = context.get_input("input-file")
    context.metadata.update_file(file_input, info=fe.get("info", {}))
