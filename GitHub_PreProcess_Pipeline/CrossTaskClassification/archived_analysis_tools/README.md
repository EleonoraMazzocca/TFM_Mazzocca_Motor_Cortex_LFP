# Archived Analysis Tools

This folder contains exploratory or one-off research scripts that are not part
of the main `CrossTaskClassification` pipeline.

They were moved here to keep the top-level folder focused on preprocessing,
splitting, classification, and leakage verification.

Archived here:
- `build_reconstruction_manifest.py`
- `inspect_separated_segments.py`
- `inspect_session_pca_health.py`
- `inspect_signal_similarity.py`
- `masked_channel_reconstruction.py`
- `plot_pca_mean_phase_grid.py`
- `plot_pca_mean_phase_planes.py`
- `plot_pca_trial_trajectories.py`

These scripts still read shared paths from the parent
`CrossTaskClassification/data_paths.py`.
