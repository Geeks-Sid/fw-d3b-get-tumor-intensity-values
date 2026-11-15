# SDK gear to calculate intensity value statistics for tumor regions

This gear extracts intensity value statistics from an input image, within enhancing, non-enhancing, cystic, edema, and whole-tumor (enhancing+non-enhancing+cystic) regions and injects the results as file-level metadata on Flywheel.

## Usage

Run at the file-level.

### Inputs

* input-file: NIfTI image to process. Must be in an acquisition.

### Configuration

* __debug__ (boolean, default False): Include debug statements in output.

### Segmentation

The segmentation is read from the acquisition that holds the input image. It must

* be a `.nii` or `.nii.gz` file with "ManualSeg" in its name (the input file itself is ignored);
* be the only such file in the acquisition (zero or several matches fail the job);
* have the same shape as the input image (a mismatch fails the job). Differing affines only log a warning.

Label values: 1 = enhancing, 2 = non-enhancing, 3 = cystic, 4 = edema. Whole tumor (`wt`) = labels 1+2+3. Other non-zero labels are ignored with a warning.

### Output

File-level metadata `info.TumorSegIntensity` always holds 15 values: `<region>_mean`, `<region>_median` and `<region>_stdev` for each region in `enhancing`, `nonenhancing`, `cyst`, `edema`, `wt`.

* A region with no voxels in the segmentation gets `null`, which overwrites any value stored by an earlier run.
* Statistics use every voxel labelled for the region, including voxels with intensity 0. Image intensities are used as stored (no rounding).
* `_stdev` is the population standard deviation (`ddof=0`).
* The job fails if the image has NaN/Inf voxels inside a region, or if either file is empty.

### Limitations

Current limitations of the gear are as follows:

* the segmentation must already be on the input image's voxel grid; the gear does not resample or register anything

## Development

```
pip install -r requirements.txt pytest
python -m pytest
```
