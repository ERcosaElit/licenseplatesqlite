import os
import cv2
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QFileDialog, QLabel, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from video_processor import VideoProcessor
from manual import ManualLicensePlateDialog



class ImageProcessingThread(QThread):
    result_ready = pyqtSignal(object)

    def __init__(self, processor, image_path):
        super().__init__()
        self.processor = processor
        self.image_path = image_path

    def run(self):
        image = cv2.imread(self.image_path)
        if image is not None:
            processed_image = self.processor.process_single_image(image)
            self.result_ready.emit(processed_image)


class VideoLicensePlateGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_processor = VideoProcessor()
        self.video_file_path = None
        self.current_image = None
        self.image_processing_thread = None
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface components"""
        self.setWindowTitle("Rendszám Felismerő")
        # self.setWindowIcon(QIcon('C:\\PythonProject\\imagines.png'))
        self.setGeometry(100, 100, 1280, 720)

        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Top area for file selection and controls
        top_layout = QHBoxLayout()

        # Button to browse for video files
        self.browse_button = QPushButton("Videó betallózása")
        self.browse_button.setMinimumWidth(150)
        self.browse_button.clicked.connect(self.browse_video)
        top_layout.addWidget(self.browse_button)

        # Button to browse for image files
        self.browse_image_button = QPushButton("Kép betallózása")
        self.browse_image_button.setMinimumWidth(150)
        self.browse_image_button.clicked.connect(self.browse_image)
        top_layout.addWidget(self.browse_image_button)

        # Button for manual license plate input
        self.manual_button = QPushButton("Kézi rendszám bevitel")
        self.manual_button.setMinimumWidth(150)
        self.manual_button.clicked.connect(self.open_manual_dialog)
        top_layout.addWidget(self.manual_button)

        # File path display
        self.file_path_label = QLabel("Nincs kiválasztott fájl")
        self.file_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_layout.addWidget(self.file_path_label)

        # Play/Pause button
        self.play_button = QPushButton("Lejátszás")
        self.play_button.setMinimumWidth(150)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        top_layout.addWidget(self.play_button)

        main_layout.addLayout(top_layout)

        # Video display area
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.video_label)

        # Status bar for additional information
        self.statusBar().showMessage("Kész a betöltésre")

    def open_manual_dialog(self):
        """Kézi rendszám beviteli ablak megnyitása"""
        from manual import ManualLicensePlateDialog  # Relatív importálás a jelenlegi mappából
        dialog = ManualLicensePlateDialog(self)
        dialog.exec_()

    def browse_video(self):
        """Open file dialog to select a video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Videó kiválasztása", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )

        if file_path:
            self.video_file_path = file_path
            self.file_path_label.setText(os.path.basename(file_path))
            self.statusBar().showMessage(f"Videó betöltve: {os.path.basename(file_path)}")

            # Reset processor state
            self.video_processor.reset()

            # Load the video
            if self.video_processor.load_video(file_path):
                self.play_button.setEnabled(True)
            else:
                self.statusBar().showMessage("Hiba a videó betöltésekor")
                self.play_button.setEnabled(False)

    def browse_image(self):
        """Open file dialog to select an image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Kép kiválasztása", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )

        if file_path:
            self.statusBar().showMessage(f"Kép betöltve: {os.path.basename(file_path)}")
            self.file_path_label.setText(os.path.basename(file_path))

            # Load and display original image
            image = cv2.imread(file_path)
            if image is not None:
                # Store original image
                self.current_image = image

                # Display image
                self.display_image(image)

                # Process image in a separate thread
                if self.image_processing_thread is not None and self.image_processing_thread.isRunning():
                    self.image_processing_thread.terminate()
                    self.image_processing_thread.wait()

                self.statusBar().showMessage("Kép feldolgozása folyamatban...")
                self.image_processing_thread = ImageProcessingThread(self.video_processor, file_path)
                self.image_processing_thread.result_ready.connect(self.handle_processed_image)
                self.image_processing_thread.start()
            else:
                self.statusBar().showMessage("Hiba a kép betöltésekor")

    def handle_processed_image(self, processed_image):
        """Handle the processed image from the thread"""
        if processed_image is not None:
            self.current_image = processed_image
            self.display_image(processed_image)
            self.statusBar().showMessage("Kép feldolgozása befejezve")

    def display_image(self, image):
        """Display the image with proper scaling when window is resized"""
        if image is None:
            return

        # Convert OpenCV BGR image to RGB for Qt
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        height, width, channel = rgb_image.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        # Display the scaled image
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.setAlignment(Qt.AlignCenter)

    def toggle_play(self):
        """Toggle video playback"""
        if self.video_processor.video_thread and self.video_processor.video_thread.isRunning():
            # Stop playback
            self.video_processor.stop_processing()
            self.play_button.setText("Lejátszás")
            self.statusBar().showMessage("Lejátszás leállítva")
        else:
            # Start playback
            if self.video_file_path:
                # Reload the video to start from the beginning
                self.video_processor.load_video(self.video_file_path)

                # Start processing
                video_thread = self.video_processor.start_processing()
                video_thread.frame_ready.connect(self.display_frame)
                self.play_button.setText("Szünet")
                self.statusBar().showMessage("Lejátszás folyamatban...")

    def display_frame(self, frame):
        """Display the processed frame"""
        if frame is None:
            return

        # Store current frame for resize events
        self.current_image = frame

        # Convert OpenCV BGR image to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        height, width, channel = rgb_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        """Handle window resize event to properly scale the image"""
        if self.current_image is not None:
            self.display_image(self.current_image)

        # Call the parent class resizeEvent
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop all threads before closing
        if self.image_processing_thread is not None and self.image_processing_thread.isRunning():
            self.image_processing_thread.terminate()
            self.image_processing_thread.wait()

        self.video_processor.reset()
        event.accept()
