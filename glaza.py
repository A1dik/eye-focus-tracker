"""
Eye Gaze Tracking - Simple OpenCV version
Показывает видео когда ты отводишь взгляд от экрана
"""

import cv2
import time
import os
import numpy as np
import pygame

class EyeTracker:
    def __init__(self):
        # Веб-камера
        self.webcam = cv2.VideoCapture(0)
        if not self.webcam.isOpened():
            raise Exception("Cannot open webcam")

        # Детекторы лица и глаз
        cascade_path = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cascade_path + 'haarcascade_eye.xml')

        # Параметры
        self.look_away_threshold = 0.8  
        self.look_away_start = None
        self.warning_video = None
        self.video_playing = False
        self.warning_count = 0
        self.start_time = time.time()

        # История для сглаживания
        self.gaze_history = []
        self.history_size = 3

        # Инициализация pygame для звука
        pygame.mixer.init()
        self.warning_sound = None

        # Загрузка видео
        if os.path.exists("warning.mp4"):
            self.warning_video = cv2.VideoCapture("warning.mp4")
            print(" Video loaded")
        else:
            print("[WARN] No warning.mp4 found, will use text")

        # Загрузка звука
        if os.path.exists("warning_audio.wav"):
            self.warning_sound = pygame.mixer.Sound("warning_audio.wav")
            print(" Audio loaded (volume x100)")
        else:
            print("[WARN] No warning_audio.wav found, video will be silent")

        print(" Eye Tracker initialized")

    def detect_gaze(self, frame):
        """Определяет направление взгляда: center, left, right, или None"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = frame.shape[:2]

        # Улучшаем контрастность
        gray = cv2.equalizeHist(gray)

        # Ищем лицо (более агрессивные параметры)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None, frame

        # Берём первое лицо
        x, y, w, h = faces[0]

        # СТРОГАЯ ПРОВЕРКА: если лицо слишком низко (наклонил голову) - это отведение взгляда
        face_center_y = y + h // 2
        face_relative_y = face_center_y / frame_height

        # Если лицо в нижней трети экрана = смотрит вниз (в телефон)
        if face_relative_y > 0.65:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Красная рамка
            cv2.putText(frame, "HEAD DOWN!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return 'down', frame  # Новый статус - смотрит вниз

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Ищем глаза в области лица (более мягкие параметры)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)

        if len(eyes) < 1:  # Хотя бы один глаз!
            return None, frame

        # Анализируем положение зрачков
        gaze_ratios = []

        for (ex, ey, ew, eh) in eyes[:2]:  # Берём до 2 глаз
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Область глаза
            eye = roi_gray[ey:ey+eh, ex:ex+ew]

            # Находим самое тёмное место (зрачок) - адаптивный порог
            _, threshold = cv2.threshold(eye, 60, 255, cv2.THRESH_BINARY_INV)

            # Находим контуры
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Самый большой контур = зрачок
                contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(contour)

                if M['m00'] != 0:
                    # Центр зрачка
                    pupil_x = int(M['m10'] / M['m00'])

                    # Относительное положение (0 = слева, 1 = справа)
                    ratio = pupil_x / ew
                    gaze_ratios.append(ratio)

                    # Рисуем точку
                    cv2.circle(roi_color, (ex + pupil_x, ey + eh//2), 3, (0, 0, 255), -1)

        if not gaze_ratios:
            return None, frame

        # Среднее положение
        avg = np.mean(gaze_ratios)

        # Добавляем в историю для сглаживания
        self.gaze_history.append(avg)
        if len(self.gaze_history) > self.history_size:
            self.gaze_history.pop(0)

        # Сглаженное значение
        smooth_avg = np.mean(self.gaze_history)

        # Определяем направление (СТРОГИЕ границы - узкий "центр")
        if smooth_avg < 0.45:
            return 'left', frame
        elif smooth_avg > 0.55:
            return 'right', frame
        else:
            return 'center', frame

    def show_warning(self):
        """Показывает кадр предупреждения"""
        if self.warning_video and self.warning_video.isOpened():
            ret, frame = self.warning_video.read()
            if not ret:
                # Перемотка в начало
                self.warning_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.warning_video.read()
            if ret:
                return frame

        # Текстовое предупреждение
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        pulse = int(150 + 50 * abs(np.sin(time.time() * 3)))
        frame[:] = (0, 0, pulse)

        cv2.putText(frame, "ATTENTION!", (150, 200),
                   cv2.FONT_HERSHEY_BOLD, 2, (255, 255, 255), 5)
        cv2.putText(frame, "Look at the screen!", (80, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        return frame

    def draw_ui(self, frame, gaze):
        """Рисует интерфейс"""
        # Статус
        if gaze == 'center':
            color = (0, 255, 0)
            text = "LOOKING CENTER"
        elif gaze in ['left', 'right']:
            color = (0, 165, 255)
            text = f"LOOKING {gaze.upper()}"
        elif gaze == 'down':
            color = (0, 0, 255)
            text = "HEAD DOWN - LOOK UP!"
        else:
            color = (0, 0, 255)
            text = "NOT DETECTED"

        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Таймер
        if self.look_away_start:
            elapsed = time.time() - self.look_away_start
            remaining = max(0, self.look_away_threshold - elapsed)
            cv2.putText(frame, f"Warning in: {remaining:.1f}s", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Статистика
        runtime = int(time.time() - self.start_time)
        cv2.putText(frame, f"Warnings: {self.warning_count} | Time: {runtime}s",
                   (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Инструкции
        cv2.putText(frame, "ESC=exit | SPACE=dismiss", (20, frame.shape[0] - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        """Главный цикл"""
        print("\n[START] Eye Tracker is running")
        print("[INFO] Press ESC to exit, SPACE to dismiss warning\n")

        try:
            while True:
                ret, frame = self.webcam.read()
                if not ret:
                    print("[ERROR] Cannot read from webcam")
                    break

                # ВСЕГДА определяем направление взгляда (даже во время показа видео!)
                gaze, annotated = self.detect_gaze(frame.copy())

                # Логика отслеживания
                # Если НЕ смотрит в центр (смотрит в сторону ИЛИ не детектится)
                if not gaze or gaze != 'center':
                    # Смотрит в сторону или лицо не видно
                    if not self.look_away_start:
                        self.look_away_start = time.time()
                        status = gaze if gaze else "NOT DETECTED"
                        print(f"[INFO] Started looking away: {status}")

                    # Проверяем порог
                    elapsed = time.time() - self.look_away_start
                    if elapsed >= self.look_away_threshold:
                        if not self.video_playing:
                            self.video_playing = True
                            self.warning_count += 1
                            print(f"[WARNING] Look away #{self.warning_count} after {elapsed:.1f}s")

                            # Перематываем видео в начало
                            if self.warning_video:
                                self.warning_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

                            # Останавливаем предыдущий звук и запускаем заново
                            if self.warning_sound:
                                self.warning_sound.stop()  # Останавливаем если играет
                                self.warning_sound.play(-1)  # -1 = зацикленное воспроизведение
                                print("[SOUND] Playing warning audio (looped)")
                else:
                    # Смотрит в центр - ЗАКРЫВАЕМ ВИДЕО АВТОМАТИЧЕСКИ
                    if self.look_away_start:
                        print(f"[INFO] Looking back at center")
                    self.look_away_start = None

                    if self.video_playing:
                        self.video_playing = False
                        cv2.destroyWindow("WARNING - Look at screen!")

                        # Останавливаем звук
                        if self.warning_sound:
                            self.warning_sound.stop()

                        print("[INFO] Looking back - warning dismissed")

                # Показываем
                if self.video_playing:
                    warning_frame = self.show_warning()
                    cv2.imshow("WARNING - Look at screen!", warning_frame)
                else:
                    display = self.draw_ui(annotated, gaze if gaze else "NOT DETECTED")
                    cv2.imshow("Eye Tracker", display)

                # Клавиши
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    print("\n[EXIT] Exiting...")
                    break
                elif key == 32:  # SPACE - ручное закрытие
                    if self.video_playing:
                        self.video_playing = False
                        cv2.destroyWindow("WARNING - Look at screen!")

                        # Останавливаем звук
                        if self.warning_sound:
                            self.warning_sound.stop()

                        print("[INFO] Warning manually dismissed")

        except KeyboardInterrupt:
            print("\n[STOP] Interrupted")
        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка"""
        runtime = int(time.time() - self.start_time)
        print(f"\n[STATS] Runtime: {runtime}s, Warnings: {self.warning_count}")

        # Останавливаем звук если играет
        if self.warning_sound:
            self.warning_sound.stop()

        self.webcam.release()
        if self.warning_video:
            self.warning_video.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print(" Done")


if __name__ == "__main__":
    print("=" * 60)
    print("EYE GAZE TRACKER - OpenCV")
    print("=" * 60)

    try:
        tracker = EyeTracker()
        tracker.run()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\n[HELP] Make sure:")
        print("  1. pip install opencv-python numpy")
        print("  2. Webcam is connected")
        print("  3. Good lighting on your face")
