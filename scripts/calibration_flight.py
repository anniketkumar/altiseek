"""Resumable B sensor-fit sweep of fixed lattice views using the live camera."""

import argparse
import json
from pathlib import Path

import numpy as np
import rclpy
from search_flight_pilot import SearchFlight

from altiseek.deploy.calibration import load_view, save_view
from altiseek.env.navigation import Navigation
from altiseek.world.spec import WorldSpec


class CalibrationFlight(SearchFlight):
    def __init__(
        self, output: Path, spec: WorldSpec, storage: Path, contract_id: str, pilot: bool
    ) -> None:
        if spec.split != "B" or spec.world_set != "sensor-fit":
            raise ValueError("Calibration flight requires sensor-fit B")
        self.storage, self.contract_id, self.pilot = storage, contract_id, pilot
        self.calibration_completed = 0
        self.calibration_expected = 0
        super().__init__(output, spec, "B5", 17)
        self.deadline_s = 5400

    def run_search(self) -> None:
        try:
            nav = Navigation(
                self.cfg.render.levels, self.cfg.render.camera.hfov_deg, self.search_config.energy
            )
            goals = []
            for li in range(len(nav.altitudes)):
                indices = [i for i, node in enumerate(nav.nodes) if node.level == li]
                rows = sorted({nav.nodes[i].iy for i in indices})
                ordered = []
                for row_number, iy in enumerate(rows):
                    ordered.extend(
                        sorted(
                            (i for i in indices if nav.nodes[i].iy == iy),
                            key=lambda i: nav.nodes[i].ix,
                            reverse=bool(row_number % 2),
                        )
                    )
                goals.extend([ordered[0], ordered[-1]] if self.pilot else ordered)
            self.calibration_expected = len(goals)
            for index in goals:
                node = nav.nodes[index]
                key = (node.level, node.ix, node.iy)
                if load_view(self.storage, key, self.contract_id) is not None:
                    self.calibration_completed += 1
                    continue
                goal = np.r_[nav.positions[index], nav.altitudes[node.level]]
                view = self.sense(goal, node.level)
                visible = view.camera.fully_visible_cells(nav.table.grid_cells, nav.table.cell_m)
                metadata = {
                    **self.records[-1],
                    "world_id": self.spec.world_id,
                    "flight_run": str(self.output),
                }
                save_view(self.storage, key, self.contract_id, view.detections, visible, metadata)
                self.calibration_completed += 1
                print(
                    f"Calibration {self.calibration_completed}/{len(goals)} saved {key}", flush=True
                )
        except Exception as error:
            self.search_error = repr(error)
        finally:
            self.stop_requested = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("world", type=Path)
    parser.add_argument("storage", type=Path)
    parser.add_argument("contract_id")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    rclpy.init()
    node = CalibrationFlight(
        args.output, WorldSpec.load(args.world), args.storage, args.contract_id, args.pilot
    )
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
        node.policy_job.result(timeout=10)
        node.inference_pool.shutdown(wait=True)
        complete = (
            node.calibration_completed == node.calibration_expected
            and node.calibration_expected > 0
        )
        report = {
            "passed": bool(node.passed and complete and not node.search_error),
            "world_id": node.spec.world_id,
            "contract_id": args.contract_id,
            "completed": node.calibration_completed,
            "expected": node.calibration_expected,
            "pilot": args.pilot,
            "error": node.search_error,
        }
        (args.output / "calibration-summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        node.finish()
        node.destroy_node()
        rclpy.shutdown()
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
