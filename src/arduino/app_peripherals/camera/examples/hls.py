import cv2

# URL to an HLS playlist
hls_url = 'https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8'

cap = cv2.VideoCapture(hls_url)

if cap.isOpened():
    print("Successfully opened HLS stream.")
    ret, frame = cap.read()
    if ret:
        print("Successfully read a frame from the stream.")
        # You can now process the 'frame'
    else:
        print("Failed to read a frame.")
    cap.release()
else:
    print("Error: Could not open HLS stream.")