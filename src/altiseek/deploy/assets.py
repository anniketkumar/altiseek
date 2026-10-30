"""North-up tiled sea textures and the fixed nadir Gazebo camera (metres, ENU)."""

import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from altiseek.imaging.config import CameraSpec


def camera_sensor(camera: CameraSpec, topic: str, rate_hz: float = 2.0) -> str:
    """SDF camera looking along link +X; mount pitch +pi/2 for nadir."""
    r = camera.resolution_px
    return f"""<sensor name="imager" type="camera">
      <topic>{escape(topic)}</topic><always_on>true</always_on>
      <update_rate>{rate_hz}</update_rate><visualize>false</visualize>
      <camera><horizontal_fov>{math.radians(camera.hfov_deg):.16g}</horizontal_fov>
      <image><width>{r}</width><height>{r}</height><format>R8G8B8</format></image>
      <clip><near>0.1</near><far>1000</far></clip></camera></sensor>"""


def plane_mesh(texture: str, x0: float, y0: float, width: float, height: float) -> str:
    """COLLADA plane, row-zero texture edge mapped to north (+y), facing up."""
    vertices = f"{x0} {y0} 0 {x0 + width} {y0} 0 {x0 + width} {y0 + height} 0 {x0} {y0 + height} 0"
    return f"""<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit meter="1" name="meter"/><up_axis>Z_UP</up_axis></asset>
<library_images><image id="image"><init_from>{escape(texture)}</init_from></image></library_images>
<library_effects><effect id="effect"><profile_COMMON>
<newparam sid="surface"><surface type="2D"><init_from>image</init_from></surface></newparam>
<newparam sid="sampler"><sampler2D><source>surface</source></sampler2D></newparam>
<technique sid="common"><lambert><ambient><color>1 1 1 1</color></ambient>
<diffuse><texture texture="sampler" texcoord="UVMap"/>
</diffuse></lambert></technique></profile_COMMON></effect></library_effects>
<library_materials><material id="material"><instance_effect url="#effect"/></material>
</library_materials><library_geometries><geometry id="plane"><mesh>
<source id="positions"><float_array id="positions-array" count="12">{vertices}</float_array>
<technique_common><accessor source="#positions-array" count="4" stride="3">
<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
</accessor></technique_common></source>
<source id="normals"><float_array id="normals-array" count="3">0 0 1</float_array>
<technique_common><accessor source="#normals-array" count="1" stride="3">
<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
</accessor></technique_common></source>
<source id="uv"><float_array id="uv-array" count="8">0 0 1 0 1 1 0 1</float_array>
<technique_common><accessor source="#uv-array" count="4" stride="2">
<param name="S" type="float"/><param name="T" type="float"/></accessor></technique_common></source>
<vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
<triangles count="2" material="surface-material">
<input semantic="VERTEX" source="#vertices" offset="0"/>
<input semantic="TEXCOORD" source="#uv" offset="1" set="0"/>
<input semantic="NORMAL" source="#normals" offset="2"/>
<p>0 0 0 1 1 0 2 2 0 0 0 0 2 2 0 3 3 0</p></triangles></mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="scene"><node id="ground">
<instance_geometry url="#plane"><bind_material><technique_common>
<instance_material symbol="surface-material" target="#material">
<bind_vertex_input semantic="UVMap" input_semantic="TEXCOORD" input_set="0"/>
</instance_material></technique_common></bind_material></instance_geometry>
</node></visual_scene></library_visual_scenes><scene><instance_visual_scene url="#scene"/>
</scene></COLLADA>"""


def export_tiles(rgb: NDArray[np.uint8], output: Path, gsd_mpp: float, tile_px: int = 2048) -> str:
    """Export bounded PNG/mesh tiles with no resampling; return a static SDF model.

    Input row 0 is the north edge. Pixel boundaries span [0,width*gsd] and
    [0,height*gsd] in ENU. Output must be gitignored artifact storage.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("Expected an H x W x 3 uint8 texture")
    if not math.isfinite(gsd_mpp) or gsd_mpp <= 0 or tile_px < 1:
        raise ValueError("Positive GSD and tile size required")
    output.mkdir(parents=True, exist_ok=True)
    h, w = rgb.shape[:2]
    visuals = []
    for row in range(0, h, tile_px):
        for col in range(0, w, tile_px):
            tile = rgb[row : row + tile_px, col : col + tile_px]
            th, tw = tile.shape[:2]
            stem = f"tile_{row}_{col}"
            Image.fromarray(tile).save(output / f"{stem}.png")
            mesh = output / f"{stem}.dae"
            mesh.write_text(
                plane_mesh(
                    f"{stem}.png",
                    col * gsd_mpp,
                    (h - row - th) * gsd_mpp,
                    tw * gsd_mpp,
                    th * gsd_mpp,
                ),
                encoding="utf-8",
            )
            visuals.append(f"""<visual name="{stem}"><cast_shadows>false</cast_shadows>
              <geometry><mesh><uri>{escape(mesh.resolve().as_uri())}</uri></mesh></geometry>
              </visual>""")
    return (
        '<model name="sea"><static>true</static><link name="surface">'
        + "".join(visuals)
        + "</link></model>"
    )


def checker_texture(area_m: float = 300.0, gsd_mpp: float = 0.25) -> NDArray[np.uint8]:
    """Synthetic 5 m checks, with red NE and blue SW markers for orientation."""
    size = round(area_m / gsd_mpp)
    yy, xx = np.indices((size, size))
    x = (xx + 0.5) * gsd_mpp
    y = area_m - (yy + 0.5) * gsd_mpp
    shade = np.where((np.floor(x / 5) + np.floor(y / 5)) % 2 == 0, 235, 20)
    rgb = np.repeat(shade[..., None], 3, axis=2).astype(np.uint8)
    centre = area_m / 2
    rgb[(abs(x - centre - 10) < 2) & (abs(y - centre - 10) < 2)] = (255, 0, 0)
    rgb[(abs(x - centre + 10) < 2) & (abs(y - centre + 10) < 2)] = (0, 0, 255)
    return rgb


def static_camera_world(
    ground: str, camera: CameraSpec, altitudes_m: list[float], render_engine: str = "ogre2"
) -> str:
    """A static geometric probe, not a flight or transfer evaluation."""
    rigs = []
    for i, altitude in enumerate(altitudes_m):
        rigs.append(f"""<model name="camera_{i}"><static>true</static>
          <pose>150 150 {altitude} 0 {math.pi / 2} {math.pi / 2}</pose>
          <link name="optical">{camera_sensor(camera, f"/altiseek/probe/l{i}", 1)}</link>
          </model>""")
    return f"""<sdf version="1.9"><world name="camera_probe">
      <physics name="physics" type="ignored"><max_step_size>0.004</max_step_size>
      <real_time_factor>1</real_time_factor></physics>
      <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
      <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
      <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
      <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>{escape(render_engine)}</render_engine></plugin>
      <scene><ambient>1 1 1 1</ambient><background>0.1 0.1 0.1 1</background>
      <shadows>false</shadows></scene>{ground}{"".join(rigs)}</world></sdf>"""
