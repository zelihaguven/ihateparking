# ihateparking deployment bundle

This directory contains reproducible artifacts for the portable runtime image.

- `export/perception_model_numpy_mlp_v2.npz`: NumPy baseline perception checkpoint.
- `export/perception_metadata.json`: model classes, feature schema and training metrics.
- `export/planning_profiles.json`: Hybrid A* and dynamic safety parameters. Hybrid A* is algorithmic and has no learned weights.
- `export/scenario_suite.json`: deterministic perpendicular, parallel and angled parking cases.
- `export/MANIFEST.json`: SHA-256 checksums for the export files.

## Build

```bash
python3 scripts/export_pipeline.py --output-dir deployment/export --scenario-count 24 --seed 100
docker build -t ihateparking:deployment .
```

## Run

```bash
docker run --rm -p 8010:8010 ihateparking:deployment
```

The container currently exposes the transport-independent FastAPI/mock runtime. It is a deployment skeleton for real hardware integration, not a claim that camera, CAN, LiDAR or ROS2 drivers are already wired to a physical vehicle. Production deployment should replace the NumPy checkpoint with a validated ONNX/TensorRT model and connect the sensor/actuator contracts through the hardware adapter.
