# GEE_RG_Detection

This repository contains a Google Earth Engine (GEE) script for mapping rock glaciers using AlphaEarth satellite embeddings.

## Contents

- `scripts/rock_glacier_classification.js` – end-to-end workflow for training KNN, Random Forest, Support Vector Machine, and Gradient Tree Boost classifiers on AlphaEarth embeddings and exporting the results.

## Usage

1. Open the [GEE Code Editor](https://code.earthengine.google.com/).
2. Create/import your region of interest polygon and rename it to `geometry`.
3. Draw training polygons for rock glacier (`RG`) and non–rock glacier (`NONRG`) areas and ensure they are stored as `FeatureCollection` objects with the same names.
4. Copy the contents of `scripts/rock_glacier_classification.js` into a new script in the Code Editor.
5. Adjust the `year` variable or any export descriptions as needed.
6. Run the script to generate accuracy assessments, classifier comparison charts, area estimates, and export tasks.

Exports include:

- Classified rasters for each classifier (GeoTIFF)
- Sentinel-2 RGB composite (GeoTIFF)
- AlphaEarth embedding mosaic (GeoTIFF)
- Training dataset summary (CSV)

## License

This project is released under the terms of the MIT License. See the [LICENSE](LICENSE) file for details.
