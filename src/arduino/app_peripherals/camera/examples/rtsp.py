import cv2

# A freely available RTSP stream for testing.
# Note: Public streams can be unreliable and may go offline without notice.
rtsp_url = "rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa"

print(f"Attempting to connect to RTSP stream: {rtsp_url}")

# Create a VideoCapture object, letting OpenCV automatically select the backend
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
else:
    print("Successfully connected to RTSP stream.")

    # Read one frame from the stream
    ret, frame = cap.read()

    if ret:
        print(f"Successfully read a frame. Frame dimensions: {frame.shape}")
        # You could now do processing on the frame, for example:
        # height, width, channels = frame.shape
        # print(f"Frame details: Width={width}, Height={height}, Channels={channels}")
    else:
        print("Error: Failed to read a frame from the stream, it may have ended or there was a network issue.")

    # Release the capture object
    cap.release()
    print("Stream capture released.")