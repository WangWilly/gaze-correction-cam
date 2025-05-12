from utils.config import get_config
import argparse

import socket
import struct
import pickle
import cv2
import dlib
import sys

from threading import Thread
from utils.logger import Logger


################################################################################


class GazeCorrectedDisplayerConfig:
    def __init__(self):
        # General parameters
        self.predict_param_path: str = "./lm_feat/shape_predictor_68_face_landmarks.dat"

        self.recver_ip = "localhost"
        self.recver_port: int = 5005

        # static parameters
        self.face_detect_size: list[int] = [320, 240]
        self.size_video: list[int] = [640, 480]
        self.x_ratio: float = 0.0
        self.y_ratio: float = 0.0

    @classmethod
    def parse_from(
        cls, general_cfg: argparse.Namespace
    ) -> "GazeCorrectedDisplayerConfig":
        cfg = cls()
        cfg.recver_ip = general_cfg.tar_ip
        cfg.recver_port = general_cfg.recver_port
        cfg.x_ratio = cfg.size_video[0] / cfg.face_detect_size[0]
        cfg.y_ratio = cfg.size_video[1] / cfg.face_detect_size[1]

        return cfg


################################################################################


class GazeCorrectedDisplayer:
    def __init__(self, shared_v, lock):
        # Initialize logger
        self.logger = Logger(self.__class__.__name__)

        ########################################################################

        general_cfg, _ = get_config()
        cfg = GazeCorrectedDisplayerConfig.parse_from(general_cfg)
        self.cfg = cfg

        ########################################################################

        # face detection
        self.detector = dlib.get_frontal_face_detector()
        if self.detector is None:
            self.logger.log("Error: No face detector found")
            sys.exit(1)

        self.predictor = dlib.shape_predictor(cfg.predict_param_path)
        if self.predictor is None:
            self.logger.log("Error: No face predictor found")
            sys.exit(1)

        self.logger.log("Face detector and predictor loaded successfully")

        ########################################################################

        self.video_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.video_recv.bind((cfg.recver_ip, cfg.recver_port))
        self.video_recv.listen(10)
        self.logger.log("Socket now listening")
        self.conn, addr = self.video_recv.accept()
        self.logger.log(f"Connection from: {addr}")
        self.start_recv(shared_v, lock)

    ############################################################################

    def face_detection(self, frame, shared_v, lock):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(
            gray, (self.cfg.face_detect_size[0], self.cfg.face_detect_size[1])
        )
        detections = self.detector(face_detect_gray, 0)
        coor_remote_head_center = [0, 0]
        for _, bx in enumerate(detections):
            coor_remote_head_center = [
                int((bx.left() + bx.right()) * self.cfg.x_ratio / 2),
                int((bx.top() + bx.bottom()) * self.cfg.y_ratio / 2),
            ]
            break

        # share remote participant's eye to the main process
        lock.acquire()
        shared_v[0] = coor_remote_head_center[0]
        shared_v[1] = coor_remote_head_center[1]
        lock.release()

    ############################################################################

    def start_recv(self, shared_v, lock):
        data = b""
        payload_size = struct.calcsize("L")
        # self.logger.log(f"payload_size: {payload_size}")

        while True:
            while len(data) < payload_size:
                data += self.conn.recv(4096)
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += self.conn.recv(4096)
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            # Check if received a stop command
            if isinstance(frame, bytes) and frame == b"stop":
                self.logger.log("Received stop command")
                self.cleanup()
                break

            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # face detection
            self.video_recv_hd_thread = Thread(
                target=self.face_detection, args=(frame, shared_v, lock)
            )
            self.video_recv_hd_thread.start()

            try:
                cv2.imshow("Remote", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.cleanup()
                    break
            except Exception as e:
                self.logger.log(f"Display error: {e}")
                self.cleanup()
                break

    def cleanup(self):
        """Clean up resources properly"""
        self.logger.log("Cleaning up resources...")
        try:
            # Close the video window
            cv2.destroyWindow("Remote")
        except:
            pass

        try:
            # Properly shutdown the socket
            self.conn.shutdown(socket.SHUT_RDWR)
            self.conn.close()
            self.video_recv.close()
            self.logger.log("Socket connections closed")
        except Exception as e:
            self.logger.log(f"Error while closing socket: {e}")
