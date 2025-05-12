import socket
import cv2
import pickle
import struct

################################################################################

HOST = "127.0.0.1"  # Server IP address
PORT = 5005  # Server port

################################################################################

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))
print("Connected to server at {}:{}".format(HOST, PORT))

# Note: If using macOS with Continuity Camera, ensure the correct camera device is selected.
# The warning about AVCaptureDeviceTypeExternal being deprecated can be ignored if functionality is unaffected.

cap = cv2.VideoCapture(1)  # Open webcam (ensure the correct camera index is used)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # JPEG quality

################################################################################

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result, frame = cv2.imencode(".jpg", frame, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)

    client_socket.sendall(struct.pack("L", size) + data)

    # Decode the frame for display
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    cv2.imshow("Sending Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

################################################################################

cap.release()
client_socket.close()
cv2.destroyAllWindows()
