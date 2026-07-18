import sys
import os
import json
import numpy as np
import pyvista as pv
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QSlider, QLabel, QCheckBox, QFrame, QPushButton
)
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

import sys
import importlib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ================= 配置参数 =================
MODEL_NAME = 'fertility'
cfg = importlib.import_module(f'configs.config_multi_field_{MODEL_NAME}')
STL_DIR = 'stlFiles'
# 刀具低多边形分辨率，确保远程或老旧显卡流畅运行
_TOOL_RES = 20 

# 刀具参数 (请确保与 pre_main.py 缩放后的数值一致)
TOOL_PARAMS = {
    'r_tip': cfg.MANU_CONFIG['SMToolParas']['SMTipDiameter'] / 2.0,
    'r_shank': cfg.MANU_CONFIG['SMToolParas']['SMShankDiameter'] / 2.0,
    'l_tool': cfg.MANU_CONFIG['SMToolParas']['SMToolLength'],
    'r_holder': cfg.MANU_CONFIG['SMToolParas']['SMHolderDiameter'] / 2.0,
    'l_holder': cfg.MANU_CONFIG['SMToolParas']['SMHolderLength']
}
# ============================================

class SupportVisualizer(QMainWindow):
    """基于 Qt 和 PyVista 的专业支撑可视化窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Pro Support Visualizer - {MODEL_NAME}")
        self.setGeometry(100, 100, 1400, 900)

        # 1. 加载数据
        self.output_dir = STL_DIR
        self.model_path = os.path.join(self.output_dir, f'{MODEL_NAME}_support.stl')
        self.json_path = os.path.join(self.output_dir, f'{MODEL_NAME}_removal_plan.json')
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.json_path):
            print("错误：找不到生成的 STL 或 JSON 文件。")
            sys.exit()

        self.mesh = pv.read(self.model_path)
        with open(self.json_path, 'r') as f:
            self.plan = json.load(f)

        # 2. 界面布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧：PyVista 嵌入式渲染窗口
        self.plotter = QtInteractor(self)
        self.plotter.set_background('white')
        self.plotter.add_axes()
        main_layout.addWidget(self.plotter, 1)

        # 右侧：Qt 控制面板
        self.controls = self._build_controls()
        main_layout.addWidget(self.controls)

        # 3. 初始化渲染状态
        self._update_scene()

    def _build_controls(self):
        """创建右侧控制面板组件"""
        panel = QWidget()
        panel.setFixedWidth(320)
        layout = QVBoxLayout(panel)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # --- 轨迹点控制 ---
        layout.addWidget(QLabel("<b>Trajectory Progress</b>"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(self.plan) - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._update_scene) # 拖动时实时触发更新
        
        self.label_info = QLabel(f"Point: 0 / {len(self.plan)-1}")
        layout.addWidget(self.slider)
        layout.addWidget(self.label_info)

        layout.addWidget(self._sep())

        # --- 显示切换 ---
        layout.addWidget(QLabel("<b>Visibility Toggles</b>"))
        
        self.chk_model = QCheckBox("Show Model & Support")
        self.chk_model.setChecked(True)
        self.chk_model.toggled.connect(self._update_scene)
        layout.addWidget(self.chk_model)

        self.chk_tool = QCheckBox("Show SM Tool")
        self.chk_tool.setChecked(True)
        self.chk_tool.toggled.connect(self._update_scene)
        layout.addWidget(self.chk_tool)

        self.chk_point = QCheckBox("Show Target Point")
        self.chk_point.setChecked(True)
        self.chk_point.toggled.connect(self._update_scene)
        layout.addWidget(self.chk_point)

        layout.addStretch()
        
        # 重置相机按钮
        btn_reset = QPushButton("Reset Camera")
        btn_reset.clicked.connect(self.plotter.reset_camera)
        layout.addWidget(btn_reset)

        return panel

    def _sep(self):
        """创建水平分割线"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _build_sm_tool_mesh(self, pos, axis, params):
        """构建 SM 球头刀几何体（球头 + 刀杆 + 刀柄）"""
        axis = np.array(axis) / (np.linalg.norm(axis) + 1e-12)
        pos = np.array(pos)
        
        r_tip, h = params['r_tip'], params['l_tool']
        r_shank = params['r_shank']
        R, H = params['r_holder'], params['l_holder']

        # 1. 刀尖球体 (Ball Center 偏移逻辑)
        ball_center = pos + axis * r_tip
        sphere = pv.Sphere(radius=r_tip, center=ball_center, 
                           theta_resolution=_TOOL_RES, phi_resolution=_TOOL_RES)
        
        # 2. 刀杆 (Shank Cylinder)
        shank_center = ball_center + axis * (h / 2.0)
        shank = pv.Cylinder(center=shank_center, direction=axis, 
                            radius=r_shank, height=h, resolution=_TOOL_RES)

        # 3. 刀柄 (Holder Cylinder)
        holder_center = ball_center + axis * (h + H / 2.0)
        holder = pv.Cylinder(center=holder_center, direction=axis, 
                             radius=R, height=H, resolution=_TOOL_RES)

        # 合并所有网格
        return sphere.merge(shank).merge(holder)

    def _update_scene(self):
        """基于 Actor 名称的增量更新，防止画面闪烁"""
        idx = self.slider.value()
        data = self.plan[idx]
        self.label_info.setText(f"Point: {idx} / {len(self.plan)-1}")

        # 1. 更新背景模型
        if self.chk_model.isChecked():
            # 使用 name 确保同一个 actor 被覆盖，避免重复叠加
            self.plotter.add_mesh(self.mesh, color="tan", opacity=0.3, 
                                  name="static_mesh", reset_camera=False)
        else:
            self.plotter.remove_actor("static_mesh")

        # 2. 更新刀具形状
        if self.chk_tool.isChecked():
            tool_mesh = self._build_sm_tool_mesh(data['pos'], data['axis'], TOOL_PARAMS)
            self.plotter.add_mesh(tool_mesh, color="salmon", opacity=0.7, 
                                  name="tool", lighting=False, reset_camera=False)
        else:
            self.plotter.remove_actor("tool")

        # 3. 更新目标点标记
        if self.chk_point.isChecked():
            point_mesh = pv.Sphere(radius=0.012, center=data['pos'])
            self.plotter.add_mesh(point_mesh, color="blue", name="cur_point", 
                                  lighting=False, reset_camera=False)
        else:
            self.plotter.remove_actor("cur_point")

        # 4. 更新 UI 文本显示 (position 替代 loc)
        self.plotter.add_text(f"Index: {idx}", position="upper_right", 
                               font_size=10, name="info_label")
        
        # 强制执行渲染刷新
        self.plotter.render()

def launch_visualizer():
    app = QApplication(sys.argv)
    window = SupportVisualizer()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_visualizer()