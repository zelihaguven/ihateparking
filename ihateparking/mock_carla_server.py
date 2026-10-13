"""Lightweight CARLA-compatible mock server for local pipeline testing.

This is intentionally not a visual simulator. It provides a small TCP
JSON-lines boundary with deterministic kinematic motion, synthetic sensor
artifacts, GroundTruthBackend output and Hybrid A* planning. The protocol is
close enough to test client/session boundaries before the real CARLA runtime is
installed.
"""
from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import socket
import socketserver
import threading
from typing import Any

import numpy as np

from .ai_perception import NeuralPerceptionAdapter
from .geometry import Pose
from .ground_truth_benchmark import backend_for_scenario
from .planner import HybridAStar
from .scenario_generator import ScenarioConfig, ScenarioGenerator, ScenarioKind, ScenarioSpec
from .stack_contracts import Gear, SensorFrame

try:
    import cv2
except ImportError:  # pragma: no cover - optional visual artifact dependency
    cv2 = None


class MockCarlaWorld:
    """Deterministic local world with a CARLA-like sensor/control boundary."""

    def __init__(self, output_dir: str | Path = "artifacts/mock_carla"):
        self.output_dir = Path(output_dir)
        self.generator = ScenarioGenerator()
        self.scenario: ScenarioSpec | None = None
        self.pose = Pose(0.0, 0.0, 0.0)
        self.speed_mps = 0.0
        self.steering_angle_rad = 0.0
        self.gear = Gear.PARK
        self.frame = 0
        self.timestamp_s = 0.0
        self.collisions = 0
        self._perception_model: NeuralPerceptionAdapter | None = None

    def reset(self, seed: int = 100, kind: str = ScenarioKind.PERPENDICULAR_REVERSE.value) -> dict[str, Any]:
        try:
            scenario_kind = ScenarioKind(kind)
        except ValueError as exc:
            raise ValueError(f"unknown scenario kind: {kind}") from exc
        self.scenario = self.generator.generate(ScenarioConfig(seed=seed, kind=scenario_kind))
        self.pose = self.scenario.start_pose
        self.speed_mps = 0.0
        self.steering_angle_rad = 0.0
        self.gear = Gear.PARK
        self.frame = 0
        self.timestamp_s = 0.0
        self.collisions = 0
        self._perception_model = NeuralPerceptionAdapter(backend_for_scenario(self.scenario))
        return self.snapshot()

    def step(self, speed_mps: float, steering_angle_rad: float, gear: str, dt: float = 0.05) -> dict[str, Any]:
        self._require_reset()
        if dt <= 0 or dt > 1.0:
            raise ValueError("dt must be between 0 and 1 second")
        try:
            command_gear = Gear(gear)
        except ValueError as exc:
            raise ValueError(f"unsupported gear: {gear}") from exc
        max_steering = self.scenario.vehicle.max_steering
        steering = max(-max_steering, min(max_steering, float(steering_angle_rad)))
        speed = max(0.0, min(3.0, float(speed_mps)))
        signed_speed = speed if command_gear is Gear.DRIVE else -speed if command_gear is Gear.REVERSE else 0.0
        yaw_rate = signed_speed / self.scenario.vehicle.wheelbase * np.tan(steering)
        next_pose = Pose(
            self.pose.x + signed_speed * np.cos(self.pose.yaw) * dt,
            self.pose.y + signed_speed * np.sin(self.pose.yaw) * dt,
            float((self.pose.yaw + yaw_rate * dt + np.pi) % (2.0 * np.pi) - np.pi),
        )
        collision = self.scenario.to_world().is_collision(next_pose)
        if collision:
            self.collisions += 1
        else:
            self.pose = next_pose
        self.speed_mps = signed_speed
        self.steering_angle_rad = steering
        self.gear = command_gear
        self.frame += 1
        self.timestamp_s += dt
        return self.snapshot()

    def plan(self) -> dict[str, Any]:
        self._require_reset()
        result = HybridAStar(self.scenario.to_world()).plan(start=self.pose)
        return {
            "success": result.success,
            "message": result.message,
            "expanded": result.expanded,
            "cost": result.cost,
            "path_length": result.path_length,
            "poses": [asdict(pose) for pose in result.poses],
            "gears": result.gears,
            "steerings": result.steerings,
        }

    def snapshot(self) -> dict[str, Any]:
        self._require_reset()
        sensor = self._sensor_frame()
        perception = self._perception_model.infer(sensor)
        sensor_files = self._write_sensor_artifacts()
        return {
            "scenario_id": self.scenario.scenario_id,
            "frame": self.frame,
            "timestamp_s": self.timestamp_s,
            "state": {
                "pose": asdict(self.pose),
                "speed_mps": self.speed_mps,
                "steering_angle_rad": self.steering_angle_rad,
                "gear": self.gear.value,
                "collisions": self.collisions,
            },
            "sensor_frame": asdict(sensor),
            "sensor_files": sensor_files,
            "perception": self._serialize_perception(perception),
            "ground_truth_source": "GroundTruthBackend",
        }

    def spawn_plan(self) -> dict[str, Any]:
        self._require_reset()
        # Reuse the CARLA adapter's pure plan builder without importing CARLA.
        from .carla_adapter import CarlaScenarioAdapter
        return CarlaScenarioAdapter().build_spawn_plan(self.scenario).to_dict()

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = request.get("op")
        if operation == "health":
            return {"ok": True, "service": "mock_carla", "carla_compatible": False}
        if operation == "reset":
            return {"ok": True, "snapshot": self.reset(int(request.get("seed", 100)), str(request.get("kind", ScenarioKind.PERPENDICULAR_REVERSE.value)))}
        if operation == "sensor_frame":
            return {"ok": True, "snapshot": self.snapshot()}
        if operation == "plan":
            return {"ok": True, "plan": self.plan()}
        if operation == "spawn_plan":
            return {"ok": True, "spawn_plan": self.spawn_plan()}
        if operation == "step":
            return {"ok": True, "snapshot": self.step(float(request.get("speed_mps", 0.0)), float(request.get("steering_angle_rad", 0.0)), str(request.get("gear", "park")), float(request.get("dt", .05)))}
        if operation == "shutdown":
            return {"ok": True, "shutdown": True}
        raise ValueError(f"unknown operation: {operation}")

    def _sensor_frame(self) -> SensorFrame:
        files = self._artifact_paths()
        return SensorFrame(
            timestamp_s=self.timestamp_s,
            frame_id=f"{self.scenario.scenario_id}:{self.frame:06d}",
            rgb_uri=files["rgb_front"],
            depth_uri=files["depth_front"],
            lidar_uri=files["lidar"],
            odometry={"x": self.pose.x, "y": self.pose.y, "yaw": self.pose.yaw, "speed_mps": self.speed_mps},
            steering_angle_rad=self.steering_angle_rad,
        )

    def _artifact_paths(self) -> dict[str, str]:
        scenario_dir = self.output_dir / self.scenario.scenario_id
        return {
            "rgb_front": str(scenario_dir / f"rgb_front_{self.frame:06d}.npy"),
            "depth_front": str(scenario_dir / f"depth_front_{self.frame:06d}.npy"),
            "lidar": str(scenario_dir / f"lidar_{self.frame:06d}.npy"),
        }

    def _write_sensor_artifacts(self) -> dict[str, str]:
        paths = self._artifact_paths()
        scenario_dir = Path(paths["rgb_front"]).parent
        scenario_dir.mkdir(parents=True, exist_ok=True)
        rgb = np.zeros((256, 512, 3), dtype=np.uint8)
        rgb[:] = (235, 244, 241)
        depth = np.full((128, 256), 50.0, dtype=np.float32)
        lidar = self._mock_lidar_points()
        if cv2 is not None:
            cv2.rectangle(rgb, (30, 20), (480, 235), (215, 229, 226), 2)
            scale = 22.0
            origin = np.array([50.0, 205.0])
            for obstacle in self.scenario.static_obstacles:
                center = (int(origin[0] + obstacle.center[0] * scale), int(origin[1] - obstacle.center[1] * scale))
                size = (max(2, int(obstacle.length * scale / 2)), max(2, int(obstacle.width * scale / 2)))
                cv2.rectangle(rgb, (center[0] - size[0], center[1] - size[1]), (center[0] + size[0], center[1] + size[1]), (70, 85, 98), -1)
            slot = self.scenario.target_slot.center
            slot_center = (int(origin[0] + slot[0] * scale), int(origin[1] - slot[1] * scale))
            cv2.circle(rgb, slot_center, 14, (70, 160, 110), 2)
            ego = (int(origin[0] + self.pose.x * scale), int(origin[1] - self.pose.y * scale))
            cv2.circle(rgb, ego, 9, (50, 150, 195), -1)
            cv2.putText(rgb, "MOCK BEV / NOT A REAL CAMERA", (35, 250), cv2.FONT_HERSHEY_SIMPLEX, .45, (25, 55, 68), 1, cv2.LINE_AA)
        np.save(paths["rgb_front"], rgb)
        np.save(paths["depth_front"], depth)
        np.save(paths["lidar"], lidar)
        return paths

    def _mock_lidar_points(self) -> np.ndarray:
        points = []
        for obstacle in self.scenario.static_obstacles:
            dx = obstacle.center[0] - self.pose.x
            dy = obstacle.center[1] - self.pose.y
            distance = float(np.hypot(dx, dy))
            if distance > 30:
                continue
            points.append([dx, dy, 0.0, 1.0])
            points.append([dx, dy, 1.0, 1.0])
        if not points:
            points.append([0.0, 0.0, 0.0, 0.0])
        return np.asarray(points, dtype=np.float32)

    @staticmethod
    def _serialize_perception(perception: Any) -> dict[str, Any]:
        return {
            "timestamp_s": perception.timestamp_s,
            "frame_id": perception.frame_id,
            "model_name": perception.model_name,
            "drivable_space_confidence": perception.drivable_space_confidence,
            "out_of_distribution_score": perception.out_of_distribution_score,
            "slots": [asdict(item) for item in perception.slots],
            "obstacles": [asdict(item) for item in perception.obstacles],
        }

    def _require_reset(self) -> None:
        if self.scenario is None or self._perception_model is None:
            raise RuntimeError("mock world must be reset before this operation")


class _JsonLineHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        for raw_line in self.rfile:
            try:
                request = json.loads(raw_line.decode("utf-8"))
                response = self.server.state.handle(request)  # type: ignore[attr-defined]
            except Exception as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            self.wfile.write((json.dumps(response, default=str) + "\n").encode("utf-8"))
            self.wfile.flush()
            if response.get("shutdown"):
                break


class _MockTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class MockCarlaServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 2100, output_dir: str | Path = "artifacts/mock_carla"):
        self.state = MockCarlaWorld(output_dir)
        self.server = _MockTCPServer((host, port), _JsonLineHandler)
        self.server.state = self.state
        self.host, self.port = self.server.server_address

    def serve_forever(self) -> None:
        self.server.serve_forever(poll_interval=.05)

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class MockCarlaClient:
    def __init__(self, host: str, port: int, timeout_s: float = 5.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    def request(self, operation: str, **payload: Any) -> dict[str, Any]:
        request = {"op": operation, **payload}
        with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as connection:
            connection.sendall((json.dumps(request) + "\n").encode("utf-8"))
            line = connection.makefile("rb").readline()
        response = json.loads(line.decode("utf-8"))
        if not response.get("ok", True):
            raise RuntimeError(response.get("error", "mock server request failed"))
        return response


def run_smoke_test(output: str | Path = "artifacts/mock_carla_smoke_test.json", frames: int = 8) -> dict[str, Any]:
    server = MockCarlaServer(port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = MockCarlaClient(server.host, server.port)
    try:
        health = client.request("health")
        reset = client.request("reset", seed=100, kind=ScenarioKind.PERPENDICULAR_REVERSE.value)
        plan = client.request("plan")["plan"]
        snapshots = [reset["snapshot"]]
        for index in range(frames):
            snapshots.append(client.request("step", speed_mps=.4, steering_angle_rad=0.0, gear="drive", dt=.05)["snapshot"])
        spawn_plan = client.request("spawn_plan")["spawn_plan"]
        report = {
            "benchmark": "mock_carla_pipeline_smoke_test",
            "health": health,
            "scenario_id": snapshots[0]["scenario_id"],
            "plan_success": plan["success"],
            "plan_expanded": plan["expanded"],
            "plan_path_points": len(plan["poses"]),
            "sensor_snapshots": len(snapshots),
            "sensor_frame_ids": [item["sensor_frame"]["frame_id"] for item in snapshots],
            "perception_models": sorted({item["perception"]["model_name"] for item in snapshots}),
            "all_have_ground_truth": all(item["ground_truth_source"] == "GroundTruthBackend" for item in snapshots),
            "collision_count": snapshots[-1]["state"]["collisions"],
            "spawn_sensor_names": [item["name"] for item in spawn_plan["sensors"]],
            "artifact_root": str(server.state.output_dir),
        }
    finally:
        try:
            client.request("shutdown")
        except Exception:
            pass
        server.shutdown()
        thread.join(timeout=2.0)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local mock CARLA server or pipeline smoke test")
    parser.add_argument("--mode", choices=("server", "smoke-test"), default="smoke-test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2100)
    parser.add_argument("--output-dir", default="artifacts/mock_carla")
    parser.add_argument("--output", default="artifacts/mock_carla_smoke_test.json")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    if args.mode == "server":
        server = MockCarlaServer(args.host, args.port, args.output_dir)
        print(f"mock_carla_listening={server.host}:{server.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()
        return 0
    report = run_smoke_test(args.output, frames=args.frames)
    print(json.dumps({key: report[key] for key in ("scenario_id", "plan_success", "sensor_snapshots", "all_have_ground_truth", "collision_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
