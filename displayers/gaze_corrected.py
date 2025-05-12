from config import get_config

import socket
import struct
import pickle
import cv2
import dlib
import sys

from threading import Thread


################################################################################


class GazeCorrectedDisplayer:
    def __init__(self, shared_v, lock):
        ########################################################################

        conf, _ = get_config()
        if conf.mod != "flx":
            sys.exit("Wrong Model selection: flx or deepwarp")

        size_video = [640, 480]

        ########################################################################

        self.face_detect_size = [320, 240]
        self.x_ratio = size_video[0] / self.face_detect_size[0]
        self.y_ratio = size_video[1] / self.face_detect_size[1]

        ########################################################################

        # face detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "./lm_feat/shape_predictor_68_face_landmarks.dat"
        )
        if self.detector is None:
            print("Error: No face detector found")
            sys.exit(1)
        if self.predictor is None:
            print("Error: No face predictor found")
            sys.exit(1)

        print("Face detector and predictor loaded successfully")

        ########################################################################

        self.video_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket created")

        self.video_recv.bind(("", conf.recver_port))
        self.video_recv.listen(10)
        print("Socket now listening")

        # If no client connects to the server, the program will hang at the accept() call.
        # TODO: To avoid this, we can set a timeout for the socket.
        self.conn, addr = self.video_recv.accept()
        print("Connection from: ", addr)
        self.start_recv(shared_v, lock)

    ############################################################################

    def face_detection(self, frame, shared_v, lock):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detect_gray = cv2.resize(
            gray, (self.face_detect_size[0], self.face_detect_size[1])
        )
        detections = self.detector(face_detect_gray, 0)
        coor_remote_head_center = [0, 0]
        for k, bx in enumerate(detections):
            coor_remote_head_center = [
                int((bx.left() + bx.right()) * self.x_ratio / 2),
                int((bx.top() + bx.bottom()) * self.y_ratio / 2),
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
        print("payload_size: {}".format(payload_size))
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
                print("Received stop command")
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
                print(f"Display error: {e}")
                self.cleanup()
                break

    def cleanup(self):
        """Clean up resources properly"""
        print("Cleaning up resources...")
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
            print("Socket connections closed")
        except Exception as e:
            print(f"Error while closing socket: {e}")
