# SupeRANSAC for SUFLECA

This directory vendors the SupeRANSAC C++/pybind implementation used by
SUFLECA for 3D-3D rigid transform estimation. Upstream project:
[danini/superansac](https://github.com/danini/superansac).

The Python module is named `pysuperansac` and exposes:

- `RANSACSettings`
- `SamplerType.Uniform`
- `SamplerType.PROSAC`
- `ScoringType.RANSAC`
- `ScoringType.MAGSAC`
- `LocalOptimizationType.Nothing`
- `estimateRigidTransform`

Build it from the repository root with:

```bash
pip install -e third_party/superansac
```

SupeRANSAC ([danini/superansac](https://github.com/danini/superansac)) is
authored by Daniel Barath and released under the MIT License. The license is
included in this directory as `LICENSE`.
