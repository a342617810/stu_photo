import os
import sys

if hasattr(sys, "frozen"):
    bundle_dir = os.path.dirname(sys.executable)
    os.environ["QT_PLUGIN_PATH"] = os.path.join(bundle_dir, "PyQt5", "Qt", "plugins")
else:
    import site

    for sp in site.getsitepackages():
        # 尝试两种可能的路径
        for qt_dir in ["Qt", "Qt5"]:
            candidate = os.path.join(sp, "PyQt5", qt_dir, "plugins")
            if os.path.isdir(candidate):
                os.environ["QT_PLUGIN_PATH"] = candidate
                break
        if "QT_PLUGIN_PATH" in os.environ:
            break

import cv2
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

# 照片尺寸定义 (毫米转像素，按300dpi计算)
PHOTO_SIZES = {
    "1寸": (295, 413),  # 25mm × 35mm
    "小2寸": (358, 441),  # 33mm × 48mm
    "大2寸": (413, 579),  # 35mm × 49mm
    "5寸": (1500, 2100),  # 127mm × 178mm
    "7寸": (2100, 2970),  # 178mm × 254mm
    "自定义尺寸": (640, 480),
}


class PhotoCaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("学生照片采集系统")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化变量
        self.cap = None
        self.current_frame = None
        self.captured_frame = None
        self.current_student_index = 0
        self.students = []
        self.photo_width = 295
        self.photo_height = 413
        self.save_dir = ""  # 保存路径
        self.use_gpu = False  # 是否使用GPU
        self.has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0  # 是否支持CUDA

        # 加载人脸检测器
        try:
            import os
            import shutil
            import tempfile

            # 优先使用程序目录下的检测器文件
            local_cascade = os.path.join(
                os.path.dirname(__file__), "haarcascade_frontalface_default.xml"
            )

            if os.path.isfile(local_cascade):
                print(f"使用本地人脸检测器: {local_cascade}")
                self.face_cascade = cv2.CascadeClassifier(local_cascade)
            else:
                # 尝试从 cv2.data.haarcascades 加载
                cascade_path = (
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                print(f"尝试从 cv2.data.haarcascades 加载: {cascade_path}")
                self.face_cascade = cv2.CascadeClassifier(cascade_path)

                # 如果失败，尝试复制到临时目录
                if self.face_cascade.empty():
                    # 创建临时目录（不含中文）
                    temp_dir = tempfile.mkdtemp()
                    temp_cascade = os.path.join(
                        temp_dir, "haarcascade_frontalface_default.xml"
                    )

                    # 复制文件到临时目录
                    try:
                        shutil.copy2(cascade_path, temp_cascade)
                        print(f"复制到临时目录: {temp_cascade}")
                        self.face_cascade = cv2.CascadeClassifier(temp_cascade)
                    except Exception as e:
                        print(f"复制文件失败: {e}")

            if self.face_cascade.empty():
                print("警告: 人脸检测器加载失败，拍照功能将不可用")
                self.face_cascade = None
            else:
                print("人脸检测器加载成功")

            # 如果支持CUDA，加载GPU版本的检测器
            self.gpu_face_cascade = None
            if self.has_cuda and self.face_cascade is not None:
                try:
                    self.gpu_face_cascade = cv2.cuda.CascadeClassifier()
                    if os.path.isfile(local_cascade):
                        self.gpu_face_cascade.load(local_cascade)
                    else:
                        self.gpu_face_cascade.load(cascade_path)
                    if self.gpu_face_cascade.empty():
                        print("GPU人脸检测器加载失败")
                        self.gpu_face_cascade = None
                    else:
                        print("GPU人脸检测器加载成功")
                except Exception as e:
                    print(f"加载GPU人脸检测器失败: {e}")
                    self.gpu_face_cascade = None
        except Exception as e:
            print(f"加载人脸检测器时出错: {e}")
            self.face_cascade = None

        # 创建UI
        self.init_ui()

    def init_ui(self):
        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧布局
        left_layout = QVBoxLayout()

        # 摄像头实时画面
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        self.camera_label.setStyleSheet("border: 2px solid gray")
        left_layout.addWidget(self.camera_label)

        # 当前拍摄图片
        self.captured_label = QLabel("拍摄预览")
        self.captured_label.setFixedSize(320, 240)
        self.captured_label.setStyleSheet("border: 2px solid gray")
        self.captured_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.captured_label)

        main_layout.addLayout(left_layout)

        # 右侧布局
        right_layout = QVBoxLayout()

        # 右侧上半部分 - 控制区
        control_layout = QVBoxLayout()

        # 导入Excel按钮
        self.import_btn = QPushButton("导入Excel")
        self.import_btn.clicked.connect(self.import_excel)
        control_layout.addWidget(self.import_btn)

        # 保存路径选择
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(QLabel("保存路径:"))
        self.save_dir_label = QLabel("未选择")
        self.save_dir_label.setStyleSheet("color: gray;")
        save_dir_layout.addWidget(self.save_dir_label)
        self.select_dir_btn = QPushButton("选择路径")
        self.select_dir_btn.clicked.connect(self.select_save_dir)
        save_dir_layout.addWidget(self.select_dir_btn)
        control_layout.addLayout(save_dir_layout)

        # 摄像头选择
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("选择摄像头:"))
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(self.get_available_cameras())
        self.camera_combo.currentIndexChanged.connect(self.switch_camera)
        camera_layout.addWidget(self.camera_combo)
        control_layout.addLayout(camera_layout)

        # 计算设备选择（CPU/GPU）
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("计算设备:"))
        self.device_combo = QComboBox()
        if self.has_cuda:
            self.device_combo.addItems(["CPU", "GPU"])
        else:
            self.device_combo.addItems(["CPU"])
            self.device_combo.setEnabled(False)
        self.device_combo.currentIndexChanged.connect(self.switch_device)
        device_layout.addWidget(self.device_combo)
        control_layout.addLayout(device_layout)

        # 拍照按钮
        btn_layout = QHBoxLayout()
        self.capture_btn = QPushButton("拍照")
        self.capture_btn.clicked.connect(self.capture_photo)
        btn_layout.addWidget(self.capture_btn)

        self.retake_btn = QPushButton("重拍")
        self.retake_btn.clicked.connect(self.retake_photo)
        btn_layout.addWidget(self.retake_btn)

        self.prev_btn = QPushButton("上一人")
        self.prev_btn.clicked.connect(self.prev_student)
        btn_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("下一人")
        self.next_btn.clicked.connect(self.next_student)
        btn_layout.addWidget(self.next_btn)
        control_layout.addLayout(btn_layout)

        # 拍照尺寸选择
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("拍照尺寸:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems(list(PHOTO_SIZES.keys()))
        self.size_combo.currentIndexChanged.connect(self.update_photo_size)
        size_layout.addWidget(self.size_combo)

        self.size_label = QLabel(f"{self.photo_width} × {self.photo_height}")
        size_layout.addWidget(self.size_label)
        control_layout.addLayout(size_layout)

        # 自定义尺寸输入
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("自定义尺寸:"))
        self.width_edit = QLineEdit()
        self.width_edit.setPlaceholderText("宽度")
        self.width_edit.setEnabled(False)
        custom_layout.addWidget(self.width_edit)

        self.height_edit = QLineEdit()
        self.height_edit.setPlaceholderText("高度")
        self.height_edit.setEnabled(False)
        custom_layout.addWidget(self.height_edit)

        self.apply_custom_btn = QPushButton("应用")
        self.apply_custom_btn.clicked.connect(self.apply_custom_size)
        self.apply_custom_btn.setEnabled(False)
        custom_layout.addWidget(self.apply_custom_btn)
        control_layout.addLayout(custom_layout)

        right_layout.addLayout(control_layout)

        # 右侧下半部分 - Excel内容显示
        self.student_table = QTableWidget()
        self.student_table.setColumnCount(2)
        self.student_table.setHorizontalHeaderLabels(["学籍号", "姓名"])
        right_layout.addWidget(self.student_table)

        main_layout.addLayout(right_layout)

        # 初始化摄像头
        self.start_camera(0)

        # 定时器更新画面
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def get_available_cameras(self):
        """获取可用的摄像头列表"""
        cameras = []
        camera_names = self._get_camera_names()

        # 限制检测的摄像头数量，避免虚拟摄像头问题
        for i in range(3):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    if i < len(camera_names) and camera_names[i]:
                        cameras.append(camera_names[i])
                    else:
                        cameras.append(f"摄像头 {i}")
                    cap.release()
            except Exception:
                # 静默忽略虚拟摄像头等错误
                continue
        if not cameras:
            cameras = ["无可用摄像头"]
        return cameras

    def _get_camera_names(self):
        """获取Windows系统中摄像头的友好名称"""
        try:
            import wmi

            c = wmi.WMI()
            camera_names = []
            for device in c.Win32_PnPEntity():
                if device.ConfigManagerErrorCode == 0:
                    # 检查设备是否为人脸检测/摄像头相关
                    name = device.Name or ""
                    caption = device.Caption or ""
                    if any(
                        keyword in (name + caption).lower()
                        for keyword in [
                            "camera",
                            "cam",
                            "webcam",
                            "video",
                            "video device",
                        ]
                    ):
                        camera_names.append(caption if caption else name)
            return camera_names
        except Exception as e:
            print(f"获取摄像头名称失败: {e}")
            return []

    def start_camera(self, index):
        """启动指定索引的摄像头"""
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "警告", f"无法打开摄像头 {index}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"打开摄像头时出错: {e}")
            self.cap = None

    def switch_camera(self, index):
        """切换摄像头"""
        self.start_camera(index)

    def switch_device(self, index):
        """切换计算设备"""
        if index == 0:
            self.use_gpu = False
            print("切换到CPU模式")
        else:
            self.use_gpu = True
            print("切换到GPU模式")

    def update_frame(self):
        """更新摄像头画面"""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # 检测人脸并标记
                frame_with_faces = self.detect_and_draw_faces(frame)
                # 转换为Qt格式显示
                rgb_frame = cv2.cvtColor(frame_with_faces, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                q_image = QImage(
                    rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                self.camera_label.setPixmap(
                    QPixmap.fromImage(q_image).scaled(640, 480, Qt.KeepAspectRatio)
                )

    def detect_and_draw_faces(self, frame):
        """检测人脸并绘制矩形框"""
        if self.face_cascade is None or self.face_cascade.empty():
            return frame  # 如果人脸检测器不可用，直接返回原始帧

        # 复制帧，避免修改原始帧
        frame_copy = frame.copy()
        try:
            if self.use_gpu and self.gpu_face_cascade is not None:
                # GPU模式
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(gray)
                faces = self.gpu_face_cascade.detectMultiScale(gpu_frame)
                if faces is not None:
                    faces = faces.download()
                    for x, y, w, h in faces:
                        cv2.rectangle(
                            frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )
            else:
                # CPU模式
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for x, y, w, h in faces:
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except Exception as e:
            print(f"人脸检测时出错: {e}")
            # 出错时回退到CPU模式
            try:
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                for x, y, w, h in faces:
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except:
                pass

        return frame_copy

    def capture_photo(self):
        """拍照功能"""
        if self.current_frame is None:
            QMessageBox.warning(self, "警告", "请先打开摄像头")
            return

        if not self.students:
            QMessageBox.warning(self, "警告", "请先导入Excel文件")
            return

        # 检测人脸并裁切
        if self.face_cascade is None or self.face_cascade.empty():
            QMessageBox.warning(self, "警告", "人脸检测器不可用，无法进行智能裁切")
            return

        try:
            # 根据模式选择人脸检测方式
            if self.use_gpu and self.gpu_face_cascade is not None:
                # GPU模式
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(gray)
                faces = self.gpu_face_cascade.detectMultiScale(gpu_frame)
                if faces is not None:
                    faces = faces.download()
                else:
                    faces = []
            else:
                # CPU模式
                gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

            if len(faces) == 0:
                QMessageBox.warning(self, "警告", "未检测到人脸")
                return

            # 使用最大的人脸
            face = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = face

            # 按照头宽:照片=2:3的比例计算裁切区域
            # 照片宽度 = 头宽 * 1.5
            photo_width = int(fw * 1.5)
            # 根据目标照片比例计算高度
            photo_height = int(photo_width * self.photo_height / self.photo_width)

            # 人脸在照片中的垂直位置：头顶在照片的黄金分割点附近（约0.382处）
            # 黄金分割比例头顶上方留约38%的空间
            top_ratio = 0.382
            top_offset = int(photo_height * top_ratio)

            # 人脸中心x坐标
            face_center_x = fx + fw // 2

            # 计算裁切区域
            crop_x1 = face_center_x - photo_width // 2
            crop_y1 = fy - top_offset  # 从头顶上方开始
            crop_x2 = crop_x1 + photo_width
            crop_y2 = crop_y1 + photo_height

            # 获取原始图像尺寸
            img_h, img_w = self.current_frame.shape[:2]

            # 处理边界情况
            # 如果裁切区域超出图像边界，需要调整
            if crop_x1 < 0:
                crop_x2 += abs(crop_x1)
                crop_x1 = 0
            if crop_y1 < 0:
                crop_y2 += abs(crop_y1)
                crop_y1 = 0
            if crop_x2 > img_w:
                crop_x1 -= crop_x2 - img_w
                crop_x2 = img_w
            if crop_y2 > img_h:
                crop_y1 -= crop_y2 - img_h
                crop_y2 = img_h

            # 再次检查边界
            crop_x1 = max(0, crop_x1)
            crop_y1 = max(0, crop_y1)
            crop_x2 = min(img_w, crop_x2)
            crop_y2 = min(img_h, crop_y2)

            # 裁切人脸区域
            face_region = self.current_frame[crop_y1:crop_y2, crop_x1:crop_x2]
        except Exception as e:
            QMessageBox.warning(self, "错误", f"人脸检测失败: {e}")
            return

        # 调整尺寸到目标尺寸
        self.captured_frame = cv2.resize(
            face_region, (self.photo_width, self.photo_height)
        )

        # 显示预览
        rgb_frame = cv2.cvtColor(self.captured_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.captured_label.setPixmap(
            QPixmap.fromImage(q_image).scaled(320, 240, Qt.KeepAspectRatio)
        )

        # 保存照片
        self.save_photo()

    def resize_and_crop(self, image, target_width, target_height):
        """按目标尺寸进行等比缩放和裁切"""
        h, w = image.shape[:2]
        target_ratio = target_width / target_height
        current_ratio = w / h

        if current_ratio > target_ratio:
            # 宽度过大，先按高度缩放，然后裁切宽度
            scale = target_height / h
            new_w = int(w * scale)
            new_h = target_height
            resized = cv2.resize(image, (new_w, new_h))
            # 裁切中间部分
            start_x = (new_w - target_width) // 2
            cropped = resized[:, start_x : start_x + target_width]
        else:
            # 高度过大，先按宽度缩放，然后裁切高度
            scale = target_width / w
            new_w = target_width
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h))
            # 裁切中间部分
            start_y = (new_h - target_height) // 2
            cropped = resized[start_y : start_y + target_height, :]

        return cropped

    def save_photo(self):
        """保存照片"""
        if self.captured_frame is None:
            return

        if self.current_student_index < len(self.students):
            student_id = self.students[self.current_student_index]["学籍号"]
            filename = f"{student_id}.jpg"

            # 如果已选择保存路径，直接保存
            if self.save_dir:
                save_path = os.path.join(self.save_dir, filename)
            else:
                # 未选择路径时弹出对话框
                save_path, _ = QFileDialog.getSaveFileName(
                    self, "保存照片", filename, "JPEG文件 (*.jpg)"
                )
                if not save_path:
                    return

            # 使用numpy处理中文路径保存图片
            try:
                cv2.imencode(".jpg", self.captured_frame)[1].tofile(save_path)
                QMessageBox.information(self, "提示", f"照片已保存为: {save_path}")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"保存失败: {e}")

    def select_save_dir(self):
        """选择保存路径"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if dir_path:
            self.save_dir = dir_path
            self.save_dir_label.setText(dir_path)
            self.save_dir_label.setStyleSheet("color: black;")

    def retake_photo(self):
        """重拍功能"""
        self.captured_frame = None
        self.captured_label.clear()
        self.captured_label.setText("拍摄预览")

    def prev_student(self):
        """上一人"""
        if self.students and self.current_student_index > 0:
            self.current_student_index -= 1
            self.update_student_display()

    def next_student(self):
        """下一人"""
        if self.students and self.current_student_index < len(self.students) - 1:
            self.current_student_index += 1
            self.update_student_display()

    def import_excel(self):
        """导入Excel文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "", "Excel文件 (*.xlsx *.xls)"
        )
        if file_path:
            try:
                df = pd.read_excel(file_path)
                if "学籍号" in df.columns and "姓名" in df.columns:
                    self.students = df[["学籍号", "姓名"]].to_dict("records")
                    self.current_student_index = 0
                    self.update_student_table()
                    self.update_student_display()
                    QMessageBox.information(
                        self, "提示", f"成功导入 {len(self.students)} 条记录"
                    )
                else:
                    QMessageBox.warning(
                        self, "警告", 'Excel文件必须包含"学籍号"和"姓名"列'
                    )
            except Exception as e:
                QMessageBox.warning(self, "错误", f"导入失败: {str(e)}")

    def update_student_table(self):
        """更新学生列表表格"""
        self.student_table.setRowCount(len(self.students))
        for i, student in enumerate(self.students):
            self.student_table.setItem(i, 0, QTableWidgetItem(str(student["学籍号"])))
            self.student_table.setItem(i, 1, QTableWidgetItem(str(student["姓名"])))

    def update_student_display(self):
        """更新当前学生显示"""
        if self.students and 0 <= self.current_student_index < len(self.students):
            self.student_table.selectRow(self.current_student_index)

    def update_photo_size(self, index):
        """更新照片尺寸"""
        size_name = self.size_combo.itemText(index)
        if size_name == "自定义尺寸":
            self.width_edit.setEnabled(True)
            self.height_edit.setEnabled(True)
            self.apply_custom_btn.setEnabled(True)
        else:
            self.width_edit.setEnabled(False)
            self.height_edit.setEnabled(False)
            self.apply_custom_btn.setEnabled(False)
            self.photo_width, self.photo_height = PHOTO_SIZES[size_name]
            self.size_label.setText(f"{self.photo_width} × {self.photo_height}")

    def apply_custom_size(self):
        """应用自定义尺寸"""
        try:
            width = int(self.width_edit.text())
            height = int(self.height_edit.text())
            if width > 0 and height > 0:
                self.photo_width = width
                self.photo_height = height
                self.size_label.setText(f"{self.photo_width} × {self.photo_height}")
                QMessageBox.information(self, "提示", "自定义尺寸已应用")
            else:
                QMessageBox.warning(self, "警告", "尺寸必须为正整数")
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的数字")

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        if self.cap is not None:
            self.cap.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PhotoCaptureApp()
    window.show()
    sys.exit(app.exec_())
