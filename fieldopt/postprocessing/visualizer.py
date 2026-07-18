"""
Interactive visualization of the hybrid manufacturing process.

Requires: PySide6, pyvistaqt

Launch via::

    from fieldopt.postprocessing.visualizer import launch_visualizer
    launch_visualizer(pipeline_result)

Two sliders control the playback:
  * **Layer slider** – selects the current layer in the interleaved sequence.
  * **Path-point slider** – selects how far along the path the tool has moved.

Six toggleable overlays:
  1. Material state (isosurface mesh extracted from dense occupancy grid)
  2. Tool shape at the current path point
  3. Current layer mesh
  4. Path curve (per-segment solid lines + traversed highlight)
  5. Bounding box
  6. Jump lines between segments (optional, dashed gray)
"""
import sys
import numpy as np
import torch
import pyvista as pv

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QSlider, QLabel, QCheckBox, QFrame, QPushButton, QSpinBox, QComboBox,
)
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

# Low-poly resolution for tool geometry (fast to render over remote)
_TOOL_RES = 20

# ======================================================================
# Geometry builders (lightweight for remote viewing)
# ======================================================================

def _build_am_tool_mesh(tip, cone_height, cone_half_angle, z_top,
                        axis=None):
    """Return a low-poly PyVista mesh for the AM cone + cylinder tool.

    The cone apex sits at *tip* (the deposition point) and widens along
    *axis*.  The cylinder continues from the cone base upward along *axis*.

    Args:
        tip: Tool tip position ``(3,)`` in real-world coordinates.
        cone_height: AM cone height in real-world units.
        cone_half_angle: AM cone half-angle in radians.
        z_top: World-space top Z used to cap the cylinder section.
        axis: Unit direction from tip toward the machine head, ``(3,)``.
              Defaults to ``[0, 0, 1]`` (vertical).

    Returns:
        ``pv.PolyData`` mesh of the AM tool.
    """
    if axis is None:
        axis = np.array([0.0, 0.0, 1.0])
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)

    r_base = cone_height * np.tan(cone_half_angle)
    cone_center = tip + axis * (cone_height / 2.0)
    cone = pv.Cone(
        center=cone_center, direction=-axis,
        height=cone_height, radius=r_base, resolution=_TOOL_RES,
    )
    cone_base = tip + axis * cone_height
    az = axis[2]
    if abs(az) > 1e-6:
        cyl_h = max((z_top - cone_base[2]) / az, 1e-6)
    else:
        cyl_h = cone_height * 3.0
    cyl_center = cone_base + axis * (cyl_h / 2.0)
    cyl = pv.Cylinder(
        center=cyl_center, direction=axis,
        radius=r_base, height=cyl_h, resolution=_TOOL_RES,
    )
    return cone.merge(cyl)


def _build_sm_tool_mesh(ball_center, tool_vec_np, r_tip, r_shank, h, R, H):
    """Return a low-poly PyVista mesh for the SM ball-end mill + holder.

    The holder is drawn as a flat disk (height 1) instead of a tall cylinder.

    Args:
        ball_center: Ball center ``(3,)`` in real-world coordinates.
        tool_vec_np: Tool axis vector ``(3,)``.
        r_tip: Ball-tip radius.
        r_shank: Shank radius.
        h: Shank length.
        R: Holder radius.
        H: Holder length (unused; disk height is fixed at 1).

    Returns:
        ``pv.PolyData`` mesh of the SM tool.
    """
    axis = tool_vec_np / (np.linalg.norm(tool_vec_np) + 1e-12)

    sphere = pv.Sphere(
        radius=r_tip, center=ball_center,
        theta_resolution=_TOOL_RES, phi_resolution=_TOOL_RES,
    )
    sp_pts = sphere.points - ball_center
    mask = (sp_pts @ axis) < 0
    hemi = sphere.extract_points(mask)

    shank_center = ball_center + axis * (h / 2.0)
    shank = pv.Cylinder(
        center=shank_center, direction=axis,
        radius=r_shank, height=h, resolution=_TOOL_RES,
    )
    # Top holder as a flat disk (height 1) instead of full cylinder
    disk_height = 1.0
    holder_center = ball_center + axis * (h + disk_height / 2.0)
    holder = pv.Cylinder(
        center=holder_center, direction=axis,
        radius=R, height=disk_height, resolution=_TOOL_RES,
    )
    combined = shank.merge(holder)
    if hemi.n_points > 0:
        combined = combined.merge(hemi)
    return combined


# ======================================================================
# Material state helper -- dense grid -> isosurface mesh
# ======================================================================

def _compute_material_surface(ctx, time_val, resolution=80, batch_size=16384):
    """
    Compute material existence on a dense 3-D grid and extract the boundary
    surface via marching cubes.  Returns a PyVista PolyData surface mesh
    already scaled to real-world coordinates, or *None* if empty.

    Args:
        ctx: Pipeline postprocess context.
        time_val: Process time at which to evaluate material state.
        resolution: Dense sampling resolution per axis.
        batch_size: Inference batch size for grid queries.

    Returns:
        ``pv.PolyData`` surface mesh in real-world coordinates, or ``None``.
    """
    model = ctx.model
    check_func = ctx.check_func
    device = ctx.device
    max_time = ctx.max_time
    scale = ctx.scale
    sb = ctx.spaceBox

    xs = np.linspace(float(sb[0][0]), float(sb[1][0]), resolution)
    ys = np.linspace(float(sb[0][1]), float(sb[1][1]), resolution)
    zs = np.linspace(float(sb[0][2]), float(sb[1][2]), resolution)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    pts_np = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    occupancy = []
    with torch.no_grad():
        for i in range(0, len(pts_np), batch_size):
            bp = torch.tensor(
                pts_np[i:i + batch_size], dtype=torch.float32, device=device,
            )
            isIn, _ = check_func(bp)
            f1, f2, lm1_raw, lm2_raw = model(bp, field_type='timesAndMasks')
            lm1 = torch.where(isIn == 1, 5.0, lm1_raw)
            lm2 = torch.where(isIn == 1, -5.0, lm2_raw)
            t1 = f1.squeeze(-1)
            t2 = (f1 + f2 * (max_time - f1)).squeeze(-1)
            p1 = torch.sigmoid(lm1).squeeze(-1)
            p2 = torch.sigmoid(lm2).squeeze(-1)
            deposited = (p1 >= 0.5) & (t1 <= time_val)
            removed = (p2 >= 0.5) & (t2 <= time_val)
            occ = (deposited & ~removed).float()
            occupancy.append(occ.cpu().numpy())
            del bp, isIn, f1, f2, lm1_raw, lm2_raw, lm1, lm2
    torch.cuda.empty_cache()

    occ_field = np.concatenate(occupancy).reshape(resolution, resolution, resolution)
    if occ_field.max() < 0.5:
        return None

    grid = pv.StructuredGrid()
    grid.dimensions = (resolution, resolution, resolution)
    # Coordinates already in model space
    grid.points = pts_np
    grid.point_data['occupancy'] = occ_field.ravel()

    try:
        surface = grid.contour(isosurfaces=[0.5], scalars='occupancy')
    except Exception:
        return None
    if surface is None or surface.n_points == 0:
        return None

    # Scale to real-world coordinates
    surface.points = surface.points * scale
    return surface


# ======================================================================
# Main window
# ======================================================================

class ManufacturingVisualizer(QMainWindow):
    """Interactive Qt window to inspect layer/path/tool state.

    Uses the new structured ``segments`` output from the pipeline.
    Actor-based rendering avoids full ``clear()`` calls, eliminating
    the flicker that occurred when dragging sliders.
    """

    def __init__(self, result: dict):
        super().__init__()
        self.setWindowTitle("Hybrid Manufacturing Process Viewer")
        self.setGeometry(80, 80, 1500, 900)

        self.result = result
        self.layers = result['layers']
        self.paths = result['paths']
        self.ctx = result['ctx']
        self.scale = result['scale']
        self.n_layers = len(self.layers)
        # Coordinate recentering offset applied by pipeline (0-vector when not set)
        self.xy_center_offset = np.asarray(
            result.get('xy_center_offset', [0.0, 0.0, 0.0]), dtype=float
        )

        # Cache: layer_idx -> PyVista PolyData surface (or None)
        self._material_cache: dict = {}
        self._last_static_state = None

        # Layer Review mode state
        self._lr_displayed: set = set()  # set of actor names currently in scene

        # ---- Qt layout ----
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        try:
            self.plotter.show_axes()
        except Exception:
            pass
        main_layout.addWidget(self.plotter, 1)

        self._build_controls(main_layout)

        # Auto-switch to Layer Review mode when no paths are available
        if not self.paths:
            self.mode_combo.setCurrentIndex(1)  # triggers _on_mode_changed
        else:
            self._update_scene()

    # ------------------------------------------------------------------
    # Control panel
    # ------------------------------------------------------------------
    def _build_controls(self, parent_layout):
        """Create right-side Qt control panel widgets."""
        panel = QWidget()
        panel.setFixedWidth(340)
        lay = QVBoxLayout(panel)
        lay.setAlignment(Qt.AlignmentFlag.AlignTop)

        # ---- Mode selector ----
        lay.addWidget(QLabel("<b>Display Mode</b>"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Process View", "Layer Review"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        lay.addWidget(self.mode_combo)

        lay.addWidget(self._sep())

        # ==================================================================
        # Process View controls (index 0)
        # ==================================================================
        self._pv_widget = QWidget()
        pv_lay = QVBoxLayout(self._pv_widget)
        pv_lay.setContentsMargins(0, 0, 0, 0)

        # Layer slider
        pv_lay.addWidget(QLabel("<b>Layer Control</b>"))
        row = QHBoxLayout()
        self.layer_slider = QSlider(Qt.Orientation.Horizontal)
        self.layer_slider.setRange(0, max(self.n_layers - 1, 0))
        self.layer_slider.setValue(0)
        self.layer_slider.setSingleStep(1)
        self.layer_slider.setPageStep(1)  # Wheel scroll moves one step
        self.layer_slider.valueChanged.connect(self._on_layer_changed)
        self.layer_label = QLabel(f"0 / {self.n_layers}")
        row.addWidget(self.layer_slider)
        row.addWidget(self.layer_label)
        pv_lay.addLayout(row)
        self.layer_info_label = QLabel("")
        pv_lay.addWidget(self.layer_info_label)

        pv_lay.addWidget(self._sep())

        # Path-point slider
        pv_lay.addWidget(QLabel("<b>Path-Point Control</b>"))
        row2 = QHBoxLayout()
        self.point_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_slider.setRange(0, 0)
        self.point_slider.setValue(0)
        self.point_slider.setSingleStep(1)
        self.point_slider.setPageStep(1)  # Wheel scroll moves one step
        self.point_slider.valueChanged.connect(self._on_point_changed)
        self.point_label = QLabel("0 / 0")
        row2.addWidget(self.point_slider)
        row2.addWidget(self.point_label)
        pv_lay.addLayout(row2)

        pv_lay.addWidget(self._sep())

        # Visibility toggles
        pv_lay.addWidget(QLabel("<b>Visibility</b>"))

        self.chk_material = QCheckBox("Material state")
        self.chk_material.setChecked(True)
        self.chk_material.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_material)

        self.chk_tool = QCheckBox("Tool shape")
        self.chk_tool.setChecked(True)
        self.chk_tool.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_tool)

        self.chk_layer_mesh = QCheckBox("Layer mesh")
        self.chk_layer_mesh.setChecked(True)
        self.chk_layer_mesh.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_layer_mesh)

        self.chk_path = QCheckBox("Path curve")
        self.chk_path.setChecked(True)
        self.chk_path.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_path)

        self.chk_bbox = QCheckBox("Bounding box")
        self.chk_bbox.setChecked(True)
        self.chk_bbox.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_bbox)

        self.chk_jumps = QCheckBox("Show jumps")
        self.chk_jumps.setChecked(False)
        self.chk_jumps.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_jumps)

        self.chk_removed_points = QCheckBox("Show removed points")
        self.chk_removed_points.setChecked(False)
        self.chk_removed_points.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_removed_points)

        self.chk_origin_axes = QCheckBox("Show origin axes")
        self.chk_origin_axes.setChecked(True)
        self.chk_origin_axes.toggled.connect(self._update_scene)
        pv_lay.addWidget(self.chk_origin_axes)

        pv_lay.addWidget(self._sep())

        # Material resolution
        pv_lay.addWidget(QLabel("<b>Material Resolution</b>"))
        res_row = QHBoxLayout()
        self.res_spin = QSpinBox()
        self.res_spin.setRange(20, 200)
        self.res_spin.setValue(80)
        self.res_spin.setSuffix("  voxels")
        res_row.addWidget(self.res_spin)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_material)
        res_row.addWidget(self.btn_refresh)
        pv_lay.addLayout(res_row)

        lay.addWidget(self._pv_widget)

        # ==================================================================
        # Layer Review controls (index 1) – initially hidden
        # ==================================================================
        self._lr_widget = QWidget()
        lr_lay = QVBoxLayout(self._lr_widget)
        lr_lay.setContentsMargins(0, 0, 0, 0)

        lr_lay.addWidget(QLabel("<b>Layer Type</b>"))
        self.lr_type_combo = QComboBox()
        self.lr_type_combo.addItems(["AM Layers", "SM Layers"])
        self.lr_type_combo.currentIndexChanged.connect(self._on_lr_filter_changed)
        lr_lay.addWidget(self.lr_type_combo)

        lr_lay.addWidget(self._sep())

        lr_lay.addWidget(QLabel("<b>Display Mode</b>"))
        self.lr_display_combo = QComboBox()
        self.lr_display_combo.addItems(["All Up To Index", "Current Layer Only"])
        self.lr_display_combo.currentIndexChanged.connect(self._on_lr_filter_changed)
        lr_lay.addWidget(self.lr_display_combo)

        lr_lay.addWidget(self._sep())

        lr_lay.addWidget(QLabel("<b>Layer Index</b>"))
        lr_row = QHBoxLayout()
        self.lr_slider = QSlider(Qt.Orientation.Horizontal)
        self.lr_slider.setRange(0, max(self.n_layers - 1, 0))
        self.lr_slider.setValue(max(self.n_layers - 1, 0))
        self.lr_slider.setSingleStep(1)
        self.lr_slider.setPageStep(1)  # Mouse wheel scrolls 1 step
        self.lr_slider.valueChanged.connect(self._on_lr_slider_changed)
        self.lr_label = QLabel(f"{max(self.n_layers - 1, 0)} / {max(self.n_layers - 1, 0)}")
        lr_row.addWidget(self.lr_slider)
        lr_row.addWidget(self.lr_label)
        lr_lay.addLayout(lr_row)

        self.lr_info_label = QLabel("")
        lr_lay.addWidget(self.lr_info_label)

        lr_lay.addWidget(self._sep())

        self.chk_lr_bbox = QCheckBox("Show Bounding Box")
        self.chk_lr_bbox.setChecked(True)
        self.chk_lr_bbox.toggled.connect(self._on_lr_bbox_changed)
        lr_lay.addWidget(self.chk_lr_bbox)

        self.chk_lr_origin_axes = QCheckBox("Show origin axes")
        self.chk_lr_origin_axes.setChecked(True)
        self.chk_lr_origin_axes.toggled.connect(self._on_lr_bbox_changed)
        lr_lay.addWidget(self.chk_lr_origin_axes)

        lay.addWidget(self._lr_widget)
        self._lr_widget.setVisible(False)

        lay.addStretch()
        parent_layout.addWidget(panel)

    @staticmethod
    def _sep():
        """Create a horizontal separator line widget."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_mode_changed(self, index):
        """Switch between Process View (0) and Layer Review (1) modes."""
        is_lr = (index == 1)
        self._pv_widget.setVisible(not is_lr)
        self._lr_widget.setVisible(is_lr)
        # Clear all scene actors before drawing the new mode
        self._clear_all_actors()
        self._lr_displayed.clear()
        self._last_static_state = None
        try:
            self._update_scene()
        except Exception as e:
            print(f'Mode switch scene update error: {e}')
            import traceback
            traceback.print_exc()

    def _on_lr_slider_changed(self, val):
        """Handle Layer Review index slider change."""
        n = self.lr_slider.maximum()
        self.lr_label.setText(f"{val} / {n}")
        self._update_scene()

    def _get_filtered_layer_indices(self):
        """Return list of global layer indices matching the current type filter."""
        show_am = self.lr_type_combo.currentIndex() == 0
        indices = []
        for li, layer in enumerate(self.layers):
            if layer.mesh is None or layer.mesh.n_points == 0:
                continue
            if show_am and layer.layer_type == 'AM':
                indices.append(li)
            elif not show_am and layer.layer_type == 'SM':
                indices.append(li)
        return indices

    def _on_lr_filter_changed(self, _=None):
        """Handle AM/SM type or display mode change in Layer Review mode."""
        # Rebuild slider range based on filtered layer count
        filtered = self._get_filtered_layer_indices()
        n = max(len(filtered) - 1, 0)
        self.lr_slider.blockSignals(True)
        self.lr_slider.setRange(0, n)
        self.lr_slider.setValue(n)
        self.lr_slider.setPageStep(1)  # Mouse wheel scrolls 1 step
        self.lr_label.setText(f"{n} / {n}")
        self.lr_slider.blockSignals(False)
        # Full rebuild
        self._clear_all_actors()
        self._lr_displayed.clear()
        self._update_scene()

    def _on_lr_bbox_changed(self, _=None):
        """Handle bounding box checkbox toggle in Layer Review mode."""
        self._remove_actors('bbox')
        self._update_scene()

    def _on_layer_changed(self, val):
        """Handle layer slider changes and refresh dependent controls."""
        self.layer_label.setText(f"{val} / {self.n_layers}")
        path_info = self._current_path_info()
        n_pts = self._total_segment_points(path_info)
        self.point_slider.blockSignals(True)
        self.point_slider.setRange(0, max(n_pts - 1, 0))
        self.point_slider.setPageStep(1)
        self.point_slider.setValue(min(self.point_slider.value(), max(n_pts - 1, 0)))
        self.point_slider.blockSignals(False)
        self._update_point_label()
        self._update_scene()

    def _on_point_changed(self, _val):
        """Handle path-point slider change event."""
        self._update_point_label()
        self._update_scene()

    def _update_point_label(self):
        """Update UI label showing current point index over total."""
        pi = self._current_path_info()
        n = self._total_segment_points(pi)
        cur = self.point_slider.value()
        self.point_label.setText(f"{cur} / {n}")

    def _refresh_material(self):
        """Invalidate current-layer material cache and redraw scene."""
        li = self.layer_slider.value()
        if li in self._material_cache:
            del self._material_cache[li]
        self._update_scene()

    # ------------------------------------------------------------------
    # Data helpers (segment-based)
    # ------------------------------------------------------------------
    def _current_layer(self):
        idx = self.layer_slider.value()
        if 0 <= idx < self.n_layers:
            return self.layers[idx]
        return None

    def _current_path_info(self):
        idx = self.layer_slider.value()
        if 0 <= idx < len(self.paths):
            return self.paths[idx]
        return None

    @staticmethod
    def _get_segments(path_info):
        """Return structured segments list from a path info dict."""
        if path_info is None:
            return []
        return path_info.get('segments', [])

    @staticmethod
    def _total_segment_points(path_info):
        """Return total number of points across all segments."""
        if path_info is None:
            return 0
        segments = path_info.get('segments', [])
        return sum(seg['points'].shape[0] for seg in segments)

    @staticmethod
    def _all_segment_points(path_info):
        """Concatenate all segment points into a single ``(N, 3)`` array."""
        if path_info is None:
            return np.empty((0, 3))
        segments = path_info.get('segments', [])
        if not segments:
            return np.empty((0, 3))
        return np.concatenate([seg['points'] for seg in segments], axis=0)

    @staticmethod
    def _locate_point_in_segments(segments, global_idx):
        """Find ``(seg_idx, local_idx)`` for a global point index."""
        offset = 0
        for si, seg in enumerate(segments):
            n = seg['points'].shape[0]
            if global_idx < offset + n:
                return si, global_idx - offset
            offset += n
        return None, None

    def _get_material_surface(self, layer_idx, time_val):
        """Return cached (or freshly computed) material isosurface mesh."""
        if layer_idx not in self._material_cache:
            res = self.res_spin.value()
            surf = _compute_material_surface(
                self.ctx, time_val, resolution=res,
            )
            # _compute_material_surface works in model-space → scaled to real
            # coords, but the pipeline shifted all other geometry by
            # xy_center_offset.  Apply the same shift here so material
            # aligns with layer meshes and paths.
            if surf is not None and surf.n_points > 0:
                surf.points -= self.xy_center_offset
            self._material_cache[layer_idx] = surf
        return self._material_cache[layer_idx]

    # ------------------------------------------------------------------
    # Actor management (avoids clear() -> prevents flicker)
    # ------------------------------------------------------------------
    # All Process View actor names (must be cleared when switching to Layer Review)
    _ACTOR_NAMES = ('bbox', 'material', 'layer_mesh', 'tool', 'cur_point',
                    'removed_points',
                    'origin_axis_x', 'origin_axis_y', 'origin_axis_z',
                    'origin_axis_x_label', 'origin_axis_y_label', 'origin_axis_z_label')

    def _remove_actors(self, *names):
        for name in names:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

    def _remove_path_actors(self):
        """Remove all dynamically-named path/jump/progress actors."""
        actors_to_remove = [
            k for k in list(self.plotter.renderer.actors.keys())
            if k and (k == 'paths' or k.startswith('path_seg_') or k.startswith('jump_')
                      or k.startswith('path_prog_'))
        ]
        for name in actors_to_remove:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

    def _remove_lr_actors(self, names=None):
        """Remove Layer Review mesh actors by name set (or all lr_ actors)."""
        if names is None:
            names = [
                k for k in list(self.plotter.renderer.actors.keys())
                if k and k.startswith('lr_')
            ]
        for name in names:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass

    def _clear_all_actors(self):
        """Remove every named actor in the scene (full reset on mode switch).

        Clears all Process View actors: bbox, material, layer_mesh, tool,
        cur_point, removed_points, paths, jump_*, path_prog_*.
        """
        self._remove_actors(*self._ACTOR_NAMES)
        self._remove_path_actors()
        self._remove_lr_actors()
        # Fallback: remove any remaining Process View actors by pattern
        # (handles edge cases where PyVista may use different internal names)
        for name in list(self.plotter.renderer.actors.keys() or []):
            if not name:
                continue
            if (name in self._ACTOR_NAMES or name == 'paths' or
                    name.startswith('path_seg_') or name.startswith('jump_') or
                    name.startswith('path_prog_')):
                try:
                    self.plotter.remove_actor(name)
                except Exception:
                    pass

    def _suppress_render(self):
        """Return a context manager that batches VTK renders.

        When the pyvistaqt interactor is rendered inside a
        ``suppress_rendering`` block all ``add_mesh`` / ``remove_actor``
        calls defer their implicit render callbacks; only the explicit
        ``self.plotter.render()`` at the very end of ``_update_scene``
        actually paints to screen.  This eliminates the blank intermediate
        frame caused by remove → (blank render) → add → render.

        Falls back to a no-op context manager if the method is unavailable
        (older pyvistaqt releases).
        """
        import contextlib
        suppress = getattr(self.plotter, 'suppress_rendering', None)
        if suppress is None:
            return contextlib.nullcontext()
        # suppress_rendering is a bool property used as a context manager in
        # newer pyvista; try using it as one, falling back to a no-op.
        try:
            import contextlib as _cl

            @_cl.contextmanager
            def _mgr():
                try:
                    self.plotter.suppress_rendering = True
                    yield
                finally:
                    self.plotter.suppress_rendering = False

            return _mgr()
        except Exception:
            return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Scene update (actor-based, no full clear)
    # ------------------------------------------------------------------
    def _update_scene(self, *_args):
        """Redraw 3-D scene using current UI state.

        Routes to the correct update function based on the selected mode.
        All add/remove operations are batched inside ``_suppress_render()``
        so that only the final ``self.plotter.render()`` call fires.
        """
        with self._suppress_render():
            if self.mode_combo.currentIndex() == 1:
                self._update_layer_review()
            else:
                self._update_scene_inner(*_args)

    # ------------------------------------------------------------------
    # Layer Review mode rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _get_layer_colors(num_layers):
        """Generate distinct colors for layers with maximum adjacent contrast.

        Uses the same high-contrast palette as ``resultsChecking.py``:
        12 hand-picked base colors with brightness cycling for larger counts.

        Args:
            num_layers: Total number of layers to generate colors for.

        Returns:
            List of hex color strings like ``['#FF0000', '#00FF00', ...]``.
        """
        base_colors = [
            '#FF0000',  # red
            '#00FF00',  # green
            '#0000FF',  # blue
            '#FFFF00',  # yellow
            '#FF00FF',  # magenta
            '#00FFFF',  # cyan
            '#FF8000',  # orange
            '#8000FF',  # purple
            '#00FF80',  # spring green
            '#FF0080',  # rose
            '#8080FF',  # light blue-purple
            '#80FF80',  # light green
        ]

        if num_layers <= 1:
            return [base_colors[0]]

        colors = []
        if num_layers <= len(base_colors):
            step = max(1, len(base_colors) // num_layers)
            for i in range(num_layers):
                color_index = (i * step) % len(base_colors)
                colors.append(base_colors[color_index])
        else:
            for i in range(num_layers):
                base_idx = i % len(base_colors)
                base_color = base_colors[base_idx]
                cycle = i // len(base_colors)
                if cycle == 0:
                    colors.append(base_color)
                else:
                    # Darken each cycle
                    brightness = max(0.4, 1.0 - cycle * 0.15)
                    hex_c = base_color.lstrip('#')
                    r = max(0, min(255, int(int(hex_c[0:2], 16) * brightness)))
                    g = max(0, min(255, int(int(hex_c[2:4], 16) * brightness)))
                    b = max(0, min(255, int(int(hex_c[4:6], 16) * brightness)))
                    colors.append(f'#{r:02X}{g:02X}{b:02X}')

        return colors

    def _update_layer_review(self):
        """Render all AM/SM layer meshes up to the slider index.

        Uses incremental actor management:
        - Layers within the visible range and matching the type filter
          are added if not already in the scene.
        - Layers that should no longer be shown (beyond slider or
          filtered out) are removed.

        Actor names: ``lr_{global_index}`` for each layer mesh.
        Rendering: same as resultsChecking — add_mesh(mesh) directly, wireframe style.
        """
        cutoff = self.lr_slider.value()
        current_only = self.lr_display_combo.currentIndex() == 1

        # Get filtered layer indices (only AM or SM)
        filtered_indices = self._get_filtered_layer_indices()

        # --- Bounding box ---
        self._remove_actors('bbox')
        if self.chk_lr_bbox.isChecked():
            sb = self.ctx.spaceBox
            s = self.scale
            off = self.xy_center_offset
            bounds = (
                float(sb[0][0]) * s - off[0], float(sb[1][0]) * s - off[0],
                float(sb[0][1]) * s - off[1], float(sb[1][1]) * s - off[1],
                float(sb[0][2]) * s,           float(sb[1][2]) * s,
            )
            try:
                box = pv.Box(bounds=bounds)
                self.plotter.add_mesh(
                    box, style='wireframe', color='black',
                    line_width=1, lighting=False, name='bbox',
                    render=False,
                )
            except Exception:
                pass

        # Origin axes at world origin (0,0,0)
        self._remove_actors('origin_axis_x', 'origin_axis_y', 'origin_axis_z')
        if self.chk_lr_origin_axes.isChecked():
            self._draw_origin_axes(render=False)

        # Determine which layers should be visible
        visible_set: set = set()
        for fi, li in enumerate(filtered_indices):
            if current_only:
                if fi != cutoff:
                    continue
            else:
                if fi > cutoff:
                    continue
            visible_set.add(li)

        # Remove layers that dropped out of visible set
        to_remove = self._lr_displayed - visible_set
        remove_names = [f'lr_{li}' for li in to_remove]
        self._remove_lr_actors(remove_names)
        self._lr_displayed -= to_remove

        # Generate colors matching resultsChecking palette
        n_total = len(self.layers)
        colors = self._get_layer_colors(n_total)

        # Add newly visible layers (processEvents every 50 so UI stays responsive)
        to_add = visible_set - self._lr_displayed
        for k, li in enumerate(sorted(to_add)):
            if k > 0 and k % 50 == 0:
                QApplication.processEvents()
            layer = self.layers[li]
            try:
                mesh = layer.mesh
                if mesh is None or mesh.n_points == 0:
                    continue
                color_hex = colors[layer.global_index] if layer.global_index < len(colors) else '#808080'
                # Clear any scalar data (level_value, time_field) that could
                # override the solid color and cause black patches.
                render_mesh = mesh.copy()
                render_mesh.clear_data()
                self.plotter.add_mesh(
                    render_mesh,
                    color=color_hex,
                    opacity=1.0,
                    style='wireframe',
                    lighting=False,
                    line_width=2,
                    name=f'lr_{li}',
                    render=False,
                )
                self._lr_displayed.add(li)
            except Exception as e:
                print(f'Layer Review: error rendering layer {li}: {e}')

        # Update info label
        type_name = 'AM' if self.lr_type_combo.currentIndex() == 0 else 'SM'
        mode_txt = 'current only' if current_only else 'cumulative'
        self.lr_info_label.setText(
            f'Showing {len(self._lr_displayed)} {type_name} layers  [{mode_txt}]'
        )
        self.plotter.render()

    def _update_scene_inner(self, *_args):
        """Internal implementation of scene update (called inside render-suppressed context)."""
        saved_cam = None
        try:
            saved_cam = self.plotter.camera_position
        except Exception:
            pass

        layer = self._current_layer()
        if layer is None:
            self.layer_info_label.setText("No layers")
            self._remove_actors(*self._ACTOR_NAMES)
            self._remove_path_actors()
            self._last_static_state = None
            if saved_cam is not None:
                try:
                    self.plotter.camera_position = saved_cam
                except Exception:
                    pass
            self.plotter.render()
            return

        self.layer_info_label.setText(
            f"Type: {layer.layer_type}  |  time={layer.time_value:.4f}  |  "
            f"idx={layer.global_index}"
        )

        path_info = self._current_path_info()
        segments = self._get_segments(path_info)
        all_pts = self._all_segment_points(path_info)
        n_total = all_pts.shape[0]
        pt_idx = self.point_slider.value()
        has_point = n_total > 0 and pt_idx < n_total

        current_static_state = (
            self.layer_slider.value(),
            self.chk_bbox.isChecked(),
            self.chk_material.isChecked(),
            self.chk_layer_mesh.isChecked(),
            self.chk_path.isChecked(),
            self.chk_jumps.isChecked(),
            self.chk_removed_points.isChecked(),
            self.chk_origin_axes.isChecked(),
        )

        if self._last_static_state != current_static_state:
            self._last_static_state = current_static_state

            # --- UPDATE STATIC ACTORS ---
            self._remove_actors('bbox', 'material', 'layer_mesh',
                                'origin_axis_x', 'origin_axis_y', 'origin_axis_z')

            # 1. Bounding box
            if self.chk_bbox.isChecked():
                sb = self.ctx.spaceBox
                s = self.scale
                off = self.xy_center_offset
                bounds = (
                    float(sb[0][0]) * s - off[0], float(sb[1][0]) * s - off[0],
                    float(sb[0][1]) * s - off[1], float(sb[1][1]) * s - off[1],
                    float(sb[0][2]) * s,           float(sb[1][2]) * s,
                )
                try:
                    box = pv.Box(bounds=bounds)
                    self.plotter.add_mesh(
                        box, style='wireframe', color='black',
                        line_width=1, lighting=False, name='bbox',
                    )
                except Exception:
                    pass

            # Origin axes at world origin (0,0,0)
            self._remove_actors('origin_axis_x', 'origin_axis_y', 'origin_axis_z')
            if self.chk_origin_axes.isChecked():
                self._draw_origin_axes()

            # 2. Material state -- gold, semi-transparent
            if self.chk_material.isChecked():
                try:
                    surf = self._get_material_surface(
                        self.layer_slider.value(), layer.time_value,
                    )
                    if surf is not None and surf.n_points > 0:
                        self.plotter.add_mesh(
                            surf, color='#FFD700', opacity=0.35,
                            smooth_shading=True, lighting=False,
                            name='material',
                        )
                except Exception as e:
                    print(f"Material state error: {e}")

            # 3. Layer mesh -- same as resultsChecking: add_mesh(mesh) directly
            if self.chk_layer_mesh.isChecked() and layer.mesh is not None:
                try:
                    mesh = layer.mesh
                    if mesh.n_points > 0:
                        self.plotter.add_mesh(
                            mesh, color='#00BFFF', opacity=0.7,
                            style='surface', smooth_shading=True, lighting=True,
                            name='layer_mesh',
                        )
                except Exception as e:
                    print(f'Process View: layer mesh render error: {e}')
                    import traceback
                    traceback.print_exc()

            # 4. Removed collision points (optional, off by default)
            self._remove_actors('removed_points')
            if self.chk_removed_points.isChecked():
                removed_pts = path_info.get('removed_points') if path_info else None
                if removed_pts is not None and len(removed_pts) > 0:
                    try:
                        self.plotter.add_mesh(
                            pv.PolyData(removed_pts),
                            color='red',
                            point_size=8,
                            render_points_as_spheres=True,
                            lighting=False,
                            name='removed_points',
                        )
                    except Exception as e:
                        print(f'Process View: removed_points render error: {e}')

            # 5. Path curves -- colored by travel order (debug-style) + optional jump lines
            actors_to_remove = [
                k for k in list(self.plotter.renderer.actors.keys())
                if k and (k == 'paths' or k.startswith('path_seg_') or k.startswith('jump_'))
            ]
            for name in actors_to_remove:
                try:
                    self.plotter.remove_actor(name)
                except Exception:
                    pass

            if self.chk_path.isChecked() and segments:
                try:
                    all_points = []
                    lines_array = []
                    current_offset = 0

                    for seg in segments:
                        seg_pts = seg['points']
                        n_seg = seg_pts.shape[0]
                        if n_seg < 2:
                            continue
                        all_points.append(seg_pts)
                        indices = np.arange(current_offset, current_offset + n_seg, dtype=np.int64)
                        lines_array.append(n_seg)
                        lines_array.extend(indices)
                        current_offset += n_seg

                    if all_points:
                        all_points_np = np.vstack(all_points)
                        path_poly = pv.PolyData(all_points_np)
                        path_poly.lines = np.array(lines_array)

                        # Fixed dark color for path (no gradient)
                        self.plotter.add_mesh(
                            path_poly,
                            color='#1a1a2e',
                            line_width=4,
                            render_lines_as_tubes=True,
                            lighting=False,
                            name='paths',
                        )

                    # Draw jumps between segments (if checkbox enabled)
                    if self.chk_jumps.isChecked() and len(segments) > 1:
                        safe_moves = None
                        if path_info is not None:
                            filtered = path_info.get('filtered')
                            if filtered is not None:
                                safe_moves = getattr(filtered, 'safe_moves', None)
                        for ji in range(len(segments) - 1):
                            end_pt = segments[ji]['points'][-1]
                            start_pt = segments[ji + 1]['points'][0]
                            if (safe_moves and ji < len(safe_moves)
                                    and safe_moves[ji] is not None):
                                sm = safe_moves[ji]
                                jump_pts = np.vstack([
                                    end_pt[None, :],
                                    sm.retract, sm.travel, sm.approach,
                                ])
                            else:
                                jump_pts = np.vstack([
                                    end_pt[None, :], start_pt[None, :],
                                ])
                            if jump_pts.shape[0] >= 2:
                                jump_line = pv.lines_from_points(jump_pts)
                                self.plotter.add_mesh(
                                    jump_line, color='gray', line_width=1,
                                    opacity=0.5, lighting=False,
                                    style='wireframe',
                                    name=f'jump_{ji}',
                                )
                except Exception:
                    pass

        # --- UPDATE DYNAMIC ACTORS (Always Update) ---
        # Wrap dynamic-actor updates in suppress_rendering so that remove_actor
        # calls never trigger an intermediate blank frame; only the final
        # explicit self.plotter.render() at the end of _update_scene fires.
        # This eliminates the flicker seen when dragging the path-point slider.
        dynamic_actors_to_remove = [
            k for k in list(self.plotter.renderer.actors.keys())
            if k and k.startswith('path_prog_')
        ]
        for name in dynamic_actors_to_remove:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                pass
        self._remove_actors('cur_point')

        if self.chk_path.isChecked() and segments:
            try:
                # Progress highlight: traversed portion per-segment.
                # Each segment is a single clean scanline; draw directly.
                if has_point and pt_idx >= 1:
                    remaining = pt_idx + 1
                    for si, seg in enumerate(segments):
                        seg_pts = seg['points']
                        n_seg = seg_pts.shape[0]
                        if remaining <= 0:
                            break
                        take = min(n_seg, remaining)
                        if take >= 2:
                            trav_line = pv.lines_from_points(seg_pts[:take])
                            self.plotter.add_mesh(
                                trav_line, color='white',
                                line_width=4, lighting=False,
                                name=f'path_prog_{si}',
                            )
                        remaining -= n_seg

                # Current point marker
                if has_point:
                    self.plotter.add_mesh(
                        pv.PolyData(all_pts[pt_idx]),
                        color='red', point_size=10, lighting=False,
                        name='cur_point',
                    )
            except Exception:
                pass

        # 5. Tool shape (uses orientations from segment structure)
        self._remove_actors('tool')
        if self.chk_tool.isChecked() and has_point:
            try:
                seg_idx, local_idx = self._locate_point_in_segments(
                    segments, pt_idx)
                if seg_idx is not None:
                    seg = segments[seg_idx]
                    self._draw_tool(
                        layer, seg['points'][local_idx],
                        seg['orientations'], local_idx,
                    )
            except Exception as e:
                print(f"Tool draw error: {e}")

        if saved_cam is not None:
            try:
                self.plotter.camera_position = saved_cam
            except Exception:
                pass
        self.plotter.render()

    # ------------------------------------------------------------------
    # Draw origin coordinate axes
    # ------------------------------------------------------------------
    def _draw_origin_axes(self, render: bool = True):
        """Draw X/Y/Z arrows at the world origin (0, 0, 0).

        Arrow length is 15 % of the shortest real-world AABB dimension so the
        axes are always visible but don't overwhelm the model.  X = red,
        Y = green, Z = blue.  Each arrow is added as a separate named actor
        (``origin_axis_x``, ``origin_axis_y``, ``origin_axis_z``) so they can
        be toggled without touching any other actor.

        Args:
            render: If False, skip the intermediate ``plotter.render()`` call
                    (used inside Layer Review where the caller renders once at
                    the end).
        """
        sb = self.ctx.spaceBox
        s = self.scale
        real_extent = np.array([
            float(sb[1][0] - sb[0][0]) * s,
            float(sb[1][1] - sb[0][1]) * s,
            float(sb[1][2] - sb[0][2]) * s,
        ])
        axis_len = float(np.min(real_extent[real_extent > 1e-6])) * 0.15

        origin = np.array([0.0, 0.0, 0.0])
        axes = [
            ('origin_axis_x', np.array([1.0, 0.0, 0.0]), 'red',   'X'),
            ('origin_axis_y', np.array([0.0, 1.0, 0.0]), 'green', 'Y'),
            ('origin_axis_z', np.array([0.0, 0.0, 1.0]), 'blue',  'Z'),
        ]
        for name, direction, color, label in axes:
            try:
                arrow = pv.Arrow(
                    start=origin,
                    direction=direction,
                    scale=axis_len,
                    tip_length=0.25,
                    tip_radius=0.05,
                    shaft_radius=0.02,
                    shaft_resolution=12,
                    tip_resolution=12,
                )
                self.plotter.add_mesh(
                    arrow, color=color, lighting=False,
                    name=name, render=render,
                )
                # Label at arrow tip
                tip = origin + direction * axis_len * 1.15
                self.plotter.add_point_labels(
                    [tip], [label],
                    font_size=14, text_color=color,
                    bold=True, show_points=False,
                    name=f'{name}_label',
                    render=render,
                )
            except Exception as e:
                print(f'Origin axes draw error ({name}): {e}')

    # ------------------------------------------------------------------
    # Draw tool mesh
    # ------------------------------------------------------------------
    def _draw_tool(self, layer, point_real, orientations, pt_idx):
        """Draw AM/SM tool geometry at the active path point.

        Args:
            layer: Current ``Layer`` object.
            point_real: Active path point ``(3,)`` in real-world coordinates.
            orientations: Orientation dict from segment (per-segment scope).
            pt_idx: Local index into the segment's orientation arrays.
        """
        s = self.scale
        manu = self.ctx.manu_config
        if layer.layer_type == 'AM':
            cone_h = manu['AMConeHeight'] * s
            cone_a = manu['AMConeHalfAngle']
            z_top = float(self.ctx.spaceBox[1][2]) * s
            axis = None
            if orientations is not None and 'am_axis' in orientations:
                axis = orientations['am_axis'][pt_idx]
            tool_mesh = _build_am_tool_mesh(
                point_real, cone_h, cone_a, z_top, axis=axis)
            self.plotter.add_mesh(
                tool_mesh, color='#2563EB', opacity=0.7, lighting=False,
                name='tool',
            )
        else:
            r_tip = manu['SMToolParas']['SMTipDiameter'] / 2.0 * s
            r_shank = manu['SMToolParas']['SMShankDiameter'] / 2.0 * s
            h = manu['SMToolParas']['SMToolLength'] * s
            R = manu['SMToolParas']['SMHolderDiameter'] / 2.0 * s
            H = manu['SMToolParas']['SMHolderLength'] * s

            if orientations is not None and 'sm_normal_vec' in orientations:
                nv = orientations['sm_normal_vec'][pt_idx]
                tv = orientations['sm_tool_vec'][pt_idx]
            else:
                pt_model = torch.tensor(
                    point_real / s, dtype=torch.float32,
                    device=self.ctx.device,
                ).unsqueeze(0)
                with torch.no_grad():
                    two_vec = self.ctx.model.forward(
                        pt_model, field_type='field3')
                nv = two_vec[0, :3].cpu().numpy()
                tv = two_vec[0, 3:].cpu().numpy()

            tip_diameter = manu['SMToolParas']['SMTipDiameter'] * s
            bc = point_real + nv * (tip_diameter / 2.0)

            tool_mesh = _build_sm_tool_mesh(bc, tv, r_tip, r_shank, h, R, H)
            self.plotter.add_mesh(
                tool_mesh, color='salmon', opacity=0.5, lighting=False,
                name='tool',
            )


# ======================================================================
# Launcher
# ======================================================================

def launch_visualizer(pipeline_result: dict):
    """
    Open the interactive manufacturing visualizer.

    Args:
        pipeline_result: The dict returned by :func:`run_pipeline`.

    Returns:
        ``ManufacturingVisualizer`` window instance.
    """
    app = QApplication.instance()
    standalone = app is None
    if standalone:
        app = QApplication(sys.argv)

    win = ManufacturingVisualizer(pipeline_result)
    win.show()

    if standalone:
        app.exec()
    return win
