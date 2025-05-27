import cv2
import numpy as np
import torch
from paddleocr import PaddleOCR
from PyQt5.QtCore import QThread, pyqtSignal
from database import VehicleDatabase
from PIL import Image, ImageDraw, ImageFont
import unicodedata
import os
import time
import re
from queue import Queue
from threading import Thread


class VideoThread(QThread):
    frame_ready = pyqtSignal(object)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.running = False

        self.frame_queue = Queue(maxsize=60)
        self.processed_queue = Queue(maxsize=60)

    def run(self):
        self.running = True
        self.processor.ocr_counter = 0


        read_thread = Thread(target=self.read_frames)
        read_thread.daemon = True
        read_thread.start()

        processing_thread = Thread(target=self.process_frames)
        processing_thread.daemon = True
        processing_thread.start()


        while self.running:
            if not self.processed_queue.empty():
                processed_frame = self.processed_queue.get()
                self.frame_ready.emit(processed_frame)
                self.msleep(1)
            else:
                self.msleep(1)

    def read_frames(self):

        fps = self.processor.video.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03
        last_read_time = time.time()

        while self.running and self.processor.video is not None:

            current_time = time.time()
            elapsed = current_time - last_read_time
            if elapsed < delay:
                time.sleep(0.001)
                continue

            last_read_time = current_time
            ret, frame = self.processor.video.read()
            if not ret:
                break


            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:

                continue

    def process_frames(self):
        frame_count = 0
        while self.running:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = self.frame_queue.get()
            if frame is None:
                break

            frame_count += 1

            do_ocr = (frame_count % 15 == 0)


            processed_frame = self.processor.process_frame_with_frame(frame, do_ocr)
            if processed_frame is not None and not self.processed_queue.full():
                self.processed_queue.put(processed_frame)
            self.frame_queue.task_done()

    def stop(self):
        self.running = False
        self.wait()


class VideoProcessor:
    def __init__(self, yolo_model_path=None):
        self.video = None
        self.yolo_model = None
        self.reader = None
        self.yolo_model_path = yolo_model_path or 'C:\\PythonProject\\sajatyolo\\runs\\detect\\train8\\weights\\best.pt'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.plate_saved = False
        self.last_recognized_text = None
        self.ocr_counter = 0
        self.video_thread = None
        self.vehicle_db = VehicleDatabase()
        self.vehicle_data = None

        self.recognized_plates = {}


        try:
            font_path = "C:\\Windows\\Fonts\\arial.ttf"
            if os.path.exists(font_path):
                self.font = ImageFont.truetype(font_path, 20)
                self.large_font = ImageFont.truetype(font_path, 30)
            else:
                self.font = ImageFont.load_default()
                self.large_font = ImageFont.load_default()
        except IOError:
            self.font = ImageFont.load_default()
            self.large_font = ImageFont.load_default()

        # Check CUDA availability
        print(f"CUDA elérhető: {torch.cuda.is_available()}")
        print(f"CUDA eszközök száma: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"CUDA eszköz neve: {torch.cuda.get_device_name(0)}")

        self.initialize()

    def format_license_plate(self, text):

        # Ellenőrizzük, hogy a szöveg 4 betű, kötőjel, majd 3 szám formátumú-e
        if re.match(r'^[A-Za-z]{4}-[0-9]{3}$', text):
            # XX-XX-000 formátumra alakítjuk
            return text[:2] + '-' + text[2:]
        return text

    def draw_text_with_unicode(self, img, text, position, font_size=20, color=(255, 255, 255)):
        #Ékezetes szöveget ír a képre PIL segítségével
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font = self.large_font if font_size > 25 else self.font

        draw.text(position, text, font=font, fill=color)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def draw_text_with_background(self, img, text, position, font_size=20, text_color=(255, 255, 255),
                                  bg_color=(0, 0, 0), alpha=0.6):
        #Szöveget ír a képre átlátszó háttérrel

        overlay = img.copy()

        # Megfelelő méretű háttér rajzolása
        pil_img_temp = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw_temp = ImageDraw.Draw(pil_img_temp)
        font = self.large_font if font_size > 25 else self.font


        try:
            text_width, text_height = draw_temp.textbbox((0, 0), text, font=font)[2:]
        except AttributeError:

            text_width, text_height = draw_temp.textsize(text, font=font)

        # Háttér
        x, y = position
        cv2.rectangle(overlay, (x - 5, y - 5), (x + text_width + 5, y + text_height + 5), bg_color, -1)

        # Átlátszóvá tétel
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        # Szöveg
        return self.draw_text_with_unicode(img, text, position, font_size, text_color)

    def initialize(self):

        gpu_available = self.device == 'cuda'
        try:
            self.reader = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=gpu_available,
                enable_mkldnn=True,

                rec_batch_num=6,
                det_db_thresh=0.3,
                det_db_box_thresh=0.5
            )
            print(f"PaddleOCR initialized with GPU: {gpu_available}")
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            self.reader = None

        # YOLO
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.yolo_model_path)
            if self.device == 'cuda':
                self.yolo_model.to(self.device).half()
            else:
                self.yolo_model.to(self.device)
            print(f"YOLO model loaded successfully and moved to {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.yolo_model = None

    def load_video(self, file_path):
        try:
            if self.video is not None:
                self.video.release()
            self.video = cv2.VideoCapture(file_path)
            success = self.video.isOpened()
            if success:
                self.video.set(cv2.CAP_PROP_BUFFERSIZE, 60)
                print(f"Video loaded successfully: {file_path}")
            else:
                print(f"Failed to open video: {file_path}")
            return success
        except Exception as e:
            print(f"Error loading video: {e}")
            return False

    def start_processing(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
        self.video_thread = VideoThread(self)
        self.video_thread.start()
        return self.video_thread

    def stop_processing(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()

    def process_single_image(self, image):

        if image is None:
            return None

        scale_percent = 30
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        small_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


        display_image = image.copy()


        detected_plates = []

        if self.yolo_model is not None:
            try:

                results = self.yolo_model(small_image, conf=0.4)
                for result in results:

                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    boxes = boxes * (100 / scale_percent)
                    boxes = boxes.astype(int)

                    for box in boxes:
                        x1, y1, x2, y2 = box

                        cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)


                        if self.reader is not None:
                            try:
                                if y1 < 0: y1 = 0
                                if x1 < 0: x1 = 0
                                if y2 >= image.shape[0]: y2 = image.shape[0] - 1
                                if x2 >= image.shape[1]: x2 = image.shape[1] - 1

                                license_plate_img = image[y1:y2, x1:x2]


                                if license_plate_img.size > 0 and license_plate_img.shape[0] > 15 and \
                                        license_plate_img.shape[1] > 50:

                                    gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)
                                    ocr_result = self.reader.ocr(gray, cls=False)

                                    if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                                        text = ocr_result[0][0][1][0]
                                        confidence = ocr_result[0][0][1][1]

                                        if confidence > 0.5:

                                            text = self.format_license_plate(text)


                                            if text not in self.recognized_plates:

                                                vehicle_data = self.vehicle_db.get_vehicle_data(text)

                                                self.recognized_plates[text] = time.time()


                                                detected_plates.append({
                                                    'text': text,
                                                    'confidence': confidence,
                                                    'box': (x1, y1, x2, y2),
                                                    'vehicle_data': vehicle_data
                                                })


                                                cv2.imwrite(f"plate_capture_{text}.jpg", license_plate_img)
                                                print(
                                                    f"Mentve: plate_capture_{text}.jpg - Felismert szöveg: {text}, Megbízhatóság: {confidence}")
                                            else:

                                                vehicle_data = self.vehicle_db.vehicle_cache.get(text)
                                                detected_plates.append({
                                                    'text': text,
                                                    'confidence': confidence,
                                                    'box': (x1, y1, x2, y2),
                                                    'vehicle_data': vehicle_data
                                                })
                            except Exception as e:
                                print(f"OCR error: {e}")


                if detected_plates:
                    display_image = self.display_multiple_plates_on_image(display_image, detected_plates)
            except Exception as e:
                print(f"YOLO error: {e}")

        return display_image

    def display_multiple_plates_on_image(self, image, plates):
        #Megjeleníti a felismert rendszámokat és járműadatokat a képen
        if not plates:
            return image

        result_image = image.copy()
        y_offset = 10

        for i, plate_info in enumerate(plates):

            result_image = self.draw_text_with_background(
                result_image,
                f"Rendszám #{i + 1}: {plate_info['text']}",
                (20, y_offset),
                font_size=30,
                text_color=(255, 255, 255),
                bg_color=(0, 0, 0),
                alpha=0.6
            )

            y_offset += 40


            if plate_info['vehicle_data']:
                vehicle_data = plate_info['vehicle_data']
                result_image = self.draw_text_with_background(
                    result_image,
                    f"Üzembentartó: {vehicle_data['uzembentarto']}",
                    (40, y_offset),
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.6
                )

                y_offset += 25
                result_image = self.draw_text_with_background(
                    result_image,
                    f"Márka/Model: {vehicle_data['marka']} {vehicle_data['model']}",
                    (40, y_offset),
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.6
                )

                y_offset += 25
                result_image = self.draw_text_with_background(
                    result_image,
                    f"Gyártási dátum: {vehicle_data['gyartas_datum']} | Szín: {vehicle_data['szin']}",
                    (40, y_offset),
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.6
                )

                y_offset += 25
                result_image = self.draw_text_with_background(
                    result_image,
                    f"Műszaki dátuma: {vehicle_data['muszaki_datum']} | Mo-i Forgalomba helyezés: {vehicle_data['forgalomba_helyezes']}",
                    (40, y_offset),
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.6
                )

                y_offset += 40

        return result_image

    def process_frame_with_frame(self, frame, do_ocr=True):
        if frame is None:
            return None


        scale_percent = 80
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        small_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


        display_frame = frame.copy() if do_ocr else frame


        current_time = time.time()
        plates_to_remove = []
        for plate, timestamp in self.recognized_plates.items():
            if current_time - timestamp > 60:
                plates_to_remove.append(plate)
        for plate in plates_to_remove:
            del self.recognized_plates[plate]


        if self.last_recognized_text:

            cv2.rectangle(display_frame, (10, 10), (450, 50), (0, 0, 0), -1)

            display_frame = self.draw_text_with_unicode(
                display_frame,
                f"Rendszám: {self.last_recognized_text}",
                (20, 15),
                font_size=30,
                color=(255, 255, 255)
            )


            if self.vehicle_data:
                cv2.rectangle(display_frame, (10, 50), (1000, 180), (0, 0, 0), -1)

                display_frame = self.draw_text_with_unicode(
                    display_frame,
                    f"Üzembentartó: {self.vehicle_data['uzembentarto']}",
                    (20, 55),
                    color=(255, 255, 255)
                )

                display_frame = self.draw_text_with_unicode(
                    display_frame,
                    f"Márka/Model: {self.vehicle_data['marka']} {self.vehicle_data['model']}",
                    (20, 80),
                    color=(255, 255, 255)
                )

                display_frame = self.draw_text_with_unicode(
                    display_frame,
                    f"Gyártási dátum: {self.vehicle_data['gyartas_datum']} | Szín: {self.vehicle_data['szin']}",
                    (20, 105),
                    color=(255, 255, 255)
                )

                display_frame = self.draw_text_with_unicode(
                    display_frame,
                    f"Műszaki dátuma: {self.vehicle_data['muszaki_datum']} | Mo-i Forgalomba helyezés dátum: {self.vehicle_data['forgalomba_helyezes']}",
                    (20, 130),
                    color=(255, 255, 255)
                )

        if self.yolo_model is not None:
            try:

                results = self.yolo_model(small_frame, conf=0.4)
                for result in results:

                    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                    boxes = boxes * (100 / scale_percent)
                    boxes = boxes.astype(int)

                    for box in boxes:
                        x1, y1, x2, y2 = box

                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


                        if do_ocr and self.reader is not None:
                            try:

                                if y1 < 0: y1 = 0
                                if x1 < 0: x1 = 0
                                if y2 >= frame.shape[0]: y2 = frame.shape[0] - 1
                                if x2 >= frame.shape[1]: x2 = frame.shape[1] - 1

                                license_plate_img = frame[y1:y2, x1:x2]


                                if license_plate_img.size > 0 and license_plate_img.shape[0] > 15 and \
                                        license_plate_img.shape[1] > 50:

                                    gray = cv2.cvtColor(license_plate_img, cv2.COLOR_BGR2GRAY)

                                    ocr_result = self.reader.ocr(gray, cls=False)

                                    if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
                                        text = ocr_result[0][0][1][0]
                                        confidence = ocr_result[0][0][1][1]

                                        if confidence > 0.5:

                                            text = self.format_license_plate(text)


                                            if text not in self.recognized_plates:
                                                print(f"New license plate detected: {text}")

                                                self.recognized_plates[text] = current_time

                                                self.last_recognized_text = text

                                                self.vehicle_data = self.vehicle_db.get_vehicle_data(text)

                                                if not os.path.exists(f"plate_capture_{text}.jpg"):
                                                    cv2.imwrite(f"plate_capture_{text}.jpg", license_plate_img)
                                                    print(
                                                        f"Saved: plate_capture_{text}.jpg - Text: {text}, Confidence: {confidence}")
                            except Exception as e:
                                print(f"OCR error: {e}")
            except Exception as e:
                print(f"YOLO error: {e}")


        if 'license_plate_img' in locals():
            del license_plate_img
        if 'gray' in locals():
            del gray

        return display_frame

    def reset(self):
        self.stop_processing()
        if self.video is not None:
            self.video.release()
            self.video = None
        self.plate_saved = False
        self.last_recognized_text = None
        self.vehicle_data = None
        self.ocr_counter = 0
        self.recognized_plates = {}
        self.vehicle_db.disconnect()
