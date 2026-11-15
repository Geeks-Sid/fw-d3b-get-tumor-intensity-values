#!/usr/bin/env python
"""The run script."""
import logging
import sys
import tempfile
import typing as t
import os

from flywheel_gear_toolkit import GearToolkitContext

from fw_gear_functions.main import run
from fw_gear_functions.parser import parse_config
from fw_gear_functions.util import create_metadata, find_segmentation

log = logging.getLogger(__name__)


def main(context: GearToolkitContext) -> int:  # pragma: no cover
    """Parse config and run."""
    fw, file_path, file_type, config = parse_config(context)
    file_ = context.get_input("input-file")

    hierarchy = file_["hierarchy"]
    if hierarchy.get("type") != "acquisition":
        raise ValueError(
            f"input-file must be in an acquisition, not a {hierarchy.get('type')}."
        )
    acq = context.client.get_acquisition(hierarchy["id"])

    # the tumor segmentation must be in the same acquisition as the input image
    seg_filename = find_segmentation(
        [file.name for file in acq.files], file_["location"]["name"]
    )
    log.info(f"Using segmentation {seg_filename}")

    # the temp dir is removed even when processing fails
    with tempfile.TemporaryDirectory() as tmp_dir:
        seg_path = os.path.join(tmp_dir, seg_filename)
        acq.download_file(seg_filename, seg_path)
        fe = run(file_path, seg_path)

    # Tag file
    tag = config.get("tag")
    context.metadata.add_file_tags(file_, t.cast(str, tag))

    # create .metadata.json
    create_metadata(context, fe)

    return 0

if __name__ == "__main__":
    with GearToolkitContext(fail_on_validation=False) as context:
        try:
            context.init_logging()
            status = main(context)
        except Exception as exc:
            log.exception(exc)
            status = 1

    sys.exit(status)
