# C-MAPSS Data Source

**Dataset**: Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)
**Source**: NASA Prognostics Center of Excellence Data Repository
**URL**: https://data.nasa.gov/download/xaut-bemq/application%2Fzip
**Alternative URL**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
**Access Date**: 2026-10-04
**License**: Public Domain (US Government Work)

## Contents
- FD001: Single operating condition, single fault mode (HPC degradation)
- FD002: Six operating conditions, single fault mode
- FD003: Single operating condition, two fault modes
- FD004: Six operating conditions, two fault modes

## Files per subset
- `train_FDxxx.txt`: Run-to-failure trajectories
- `test_FDxxx.txt`: Trajectories cut before failure
- `RUL_FDxxx.txt`: True remaining useful life for test trajectories

## Columns (26 total)
1. Engine unit number
2. Time (cycles)
3-5. Operational settings (altitude, Mach, TRA)
6-26. Sensor measurements (21 sensors)

## Citation
A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling
for Aircraft Engine Run-to-Failure Simulation," in Proc. 1st Int. Conf.
Prognostics and Health Management (PHM08), Denver CO, 2008.
