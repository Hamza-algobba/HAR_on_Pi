# import vlc
# import numpy as np
# import cv2
# from mmengine.config import Config
# from mmaction.apis import init_recognizer, inference_recognizer
# import time
# import os
# import torch
# import torch.nn.functional as F

# # Configuration
# config_file = 'tsm_ucf_crime.py'
# checkpoint_file = 'epoch_48.pth'
# label_file = 'refined_classes.txt'
# rtsp_url = "rtsp://admin:theconclusion%231@10.7.57.206:554/cam/realmonitor?channel=1&subtype=0"
# frame_interval = 8

# # Load labels
# with open(label_file, 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# # Initialize model
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# try:
#     model = init_recognizer(config_file, checkpoint_file, device=device)
#     print("[INFO] Model loaded and initialized.")
# except Exception as e:
#     print(f"[ERROR] Failed to load model: {e}")
#     exit()

# # Initialize VLC
# print(f"[INFO] Connecting to RTSP stream: {rtsp_url}")
# instance = vlc.Instance()
# player = instance.media_player_new()
# media = instance.media_new(rtsp_url)
# player.set_media(media)
# player.play()
# time.sleep(5)  # Allow stream to buffer

# # Frame capture loop
# print("[INFO] Starting frame capture...")
# frame_count = 0
# predictions = ["Waiting for inference..."]

# while True:
#     try:
#         snapshot_path = f"/tmp/frame_{frame_count}.jpg"
#         player.video_take_snapshot(0, snapshot_path, 640, 360)


#         # Wait until the snapshot file exists
#         start_time = time.time()
#         while not os.path.exists(snapshot_path):
#             if time.time() - start_time > 3:  # Timeout after 3 seconds
#                 print(f"[WARNING] Snapshot {snapshot_path} not found. Skipping frame.")
#                 break
#             time.sleep(0.1)

#         if not os.path.exists(snapshot_path):
#             frame_count += 1
#             continue

#         frame = cv2.imread(snapshot_path)
#         if frame is None:
#             print(f"[WARNING] Could not read snapshot: {snapshot_path}")
#             frame_count += 1
#             continue

#         # Inference every `frame_interval`
#         if frame_count % frame_interval == 0:
#             temp_video_path = f"/tmp/temp_video_{frame_count}.mp4"
#             height, width, _ = frame.shape

#             # Save frame as video
#             out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
#             for _ in range(8):
#                 out.write(frame)
#             out.release()

#             # Run inference
#             try:
#                 results = inference_recognizer(model, temp_video_path)
#                 pred_scores = results.pred_score.squeeze()  # Shape: (3, 2)

#                 print(f"[DEBUG] pred_scores shape: {pred_scores.shape}")

#                 # Average across the segments (axis=0) → shape: (2,)
#                 avg_scores = pred_scores.mean(dim=0)
#                 print(f"[DEBUG] avg_scores shape: {avg_scores.shape}")

#                 # Apply softmax to get probabilities
#                 probs = F.softmax(avg_scores, dim=0).cpu().numpy()
#                 print(f"[DEBUG] probs shape: {probs.shape}")

#                 # Format predictions
#                 predictions = [f"{labels[i]}: {float(probs[i]) * 100:.2f}%" for i in range(len(labels))]
#                 predicted_class = labels[np.argmax(probs)]
#                 predictions.insert(0, f"🚨 Predicted: {predicted_class.upper()}")

#                 print(f"[INFO] Prediction: {predicted_class} | Probabilities: {predictions[1:]}")
#             except Exception as e:
#                 print(f"[ERROR] Inference failed: {e}")
#                 predictions = ["Inference error occurred."]


#         # Display predictions
#         overlay = frame.copy()
#         cv2.rectangle(overlay, (10, 10), (400, 10 + 30 * len(predictions)), (0, 0, 0), -1)
#         for i, text in enumerate(predictions):
#             color = (0, 255, 0) if "Predicted" in text else (255, 255, 255)
#             cv2.putText(overlay, text, (15, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         cv2.imshow("Real-Time Anomaly Detection", overlay)

#         # Exit on 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("[INFO] Exiting...")
#             break

#         frame_count += 1

#     except Exception as e:
#         print(f"[ERROR] Unexpected issue: {e}")
#         break

# # Cleanup
# player.stop()
# cv2.destroyAllWindows()
# print("[INFO] Resources released successfully.")



# Including localizing module

import vlc
import numpy as np
import cv2
from collections import deque
from mmengine.config import Config
from mmaction.apis import init_recognizer, inference_recognizer
import time
import os
import torch
import torch.nn.functional as F

# Configuration
config_file = 'tsm_ucf_crime.py'
checkpoint_file = 'epoch_48.pth'
label_file = 'refined_classes.txt'
rtsp_url = "rtsp://admin:theconclusion%231@10.7.57.206:554/cam/realmonitor?channel=1&subtype=0"
frame_interval = 8
frame_buffer_size = 10  # Store last 10 frames for differencing
box_smoothing_alpha = 0.2  # Weight for exponential moving average
box_persistence = 15  # Frames to keep the bounding box after anomaly disappears

# Load labels
with open(label_file, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize model
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'  # Use CPU for inference
try:
    model = init_recognizer(config_file, checkpoint_file, device=device)
    print("[INFO] Model loaded and initialized.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Initialize VLC
print(f"[INFO] Connecting to RTSP stream: {rtsp_url}")
instance = vlc.Instance()
player = instance.media_player_new()
media = instance.media_new(rtsp_url)
player.set_media(media)
player.play()
time.sleep(5)  # Allow stream to buffer

# Frame buffer for differencing
frame_buffer = deque(maxlen=frame_buffer_size)
last_box = None  # Last detected bounding box
box_counter = 0  # Keeps track of how long the box should persist

# Frame capture loop
print("[INFO] Starting frame capture...")
frame_count = 0
predictions = ["Waiting for inference..."]

while True:
    try:
        snapshot_path = f"/tmp/frame_{frame_count}.jpg"
        player.video_take_snapshot(0, snapshot_path, 640, 360)

        # Wait until the snapshot file exists
        start_time = time.time()
        while not os.path.exists(snapshot_path):
            if time.time() - start_time > 3:  # Timeout after 3 seconds
                print(f"[WARNING] Snapshot {snapshot_path} not found. Skipping frame.")
                break
            time.sleep(0.1)

        if not os.path.exists(snapshot_path):
            frame_count += 1
            continue

        frame = cv2.imread(snapshot_path)
        if frame is None:
            print(f"[WARNING] Could not read snapshot: {snapshot_path}")
            frame_count += 1
            continue

        # Store frame in buffer
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_buffer.append(gray_frame)

        # Inference every `frame_interval`
        if frame_count % frame_interval == 0:
            temp_video_path = f"/tmp/temp_video_{frame_count}.mp4"
            height, width, _ = frame.shape

            # Save frame as video
            out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (width, height))
            for _ in range(8):
                out.write(frame)
            out.release()

            # Run inference
            try:
                results = inference_recognizer(model, temp_video_path)
                pred_scores = results.pred_score.squeeze()  # Shape: (3, 2)

                print(f"[DEBUG] pred_scores shape: {pred_scores.shape}")

                # Average across the segments (axis=0) → shape: (2,)
                avg_scores = pred_scores.mean(dim=0)
                print(f"[DEBUG] avg_scores shape: {avg_scores.shape}")

                # Apply softmax to get probabilities
                probs = F.softmax(avg_scores, dim=0).cpu().numpy()
                print(f"[DEBUG] probs shape: {probs.shape}")

                # Format predictions
                predictions = [f"{labels[i]}: {float(probs[i]) * 100:.2f}%" for i in range(len(labels))]
                predicted_class = labels[np.argmax(probs)]
                predictions.insert(0, f"🚨 Predicted: {predicted_class.upper()}")

                print(f"[INFO] Prediction: {predicted_class} | Probabilities: {predictions[1:]}")

                # Anomaly Localization (Only when "anomaly" is predicted)
                if predicted_class.lower() == "anomaly" and len(frame_buffer) > 5:
                    print("[INFO] Running anomaly localization...")
                    
                    # Get previous and future frames
                    prev_frame = frame_buffer[-6]  # 5 frames before
                    next_frame = frame_buffer[-1]  # Current frame
                    
                    # Compute absolute difference
                    diff1 = cv2.absdiff(prev_frame, next_frame)
                    
                    # Threshold to identify regions of movement
                    _, thresh = cv2.threshold(diff1, 30, 255, cv2.THRESH_BINARY)
                    
                    # Find contours
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Get the largest moving area
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)

                        # 🔹 Scale bounding box (increase size by 20%)
                        scale_factor = 1.2
                        x = max(0, int(x - (w * (scale_factor - 1) / 2)))  # Expand left
                        y = max(0, int(y - (h * (scale_factor - 1) / 2)))  # Expand up
                        w = min(next_frame.shape[1] - x, int(w * scale_factor))  # Expand width
                        h = min(next_frame.shape[0] - y, int(h * scale_factor))  # Expand height

                        # 🔹 Smooth bounding box using Exponential Moving Average
                        if last_box is None:
                            last_box = (x, y, w, h)
                        else:
                            last_x, last_y, last_w, last_h = last_box
                            x = int(box_smoothing_alpha * x + (1 - box_smoothing_alpha) * last_x)
                            y = int(box_smoothing_alpha * y + (1 - box_smoothing_alpha) * last_y)
                            w = int(box_smoothing_alpha * w + (1 - box_smoothing_alpha) * last_w)
                            h = int(box_smoothing_alpha * h + (1 - box_smoothing_alpha) * last_h)
                            last_box = (x, y, w, h)

                        # Set persistence counter
                        box_counter = box_persistence

                        print(f"[INFO] Bounding box (scaled & smoothed): x={x}, y={y}, w={w}, h={h}")


            except Exception as e:
                print(f"[ERROR] Inference failed: {e}")
                predictions = ["Inference error occurred."]

        # Display predictions
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 10 + 30 * len(predictions)), (0, 0, 0), -1)
        for i, text in enumerate(predictions):
            color = (0, 255, 0) if "Predicted" in text else (255, 255, 255)
            cv2.putText(overlay, text, (15, 35 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw smoothed bounding box if anomaly persists
        if box_counter > 0 and last_box is not None:
            x, y, w, h = last_box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
            box_counter -= 1

        cv2.imshow("Real-Time Anomaly Localization", overlay)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

        frame_count += 1

    except Exception as e:
        print(f"[ERROR] Unexpected issue: {e}")
        break

# Cleanup
player.stop()
cv2.destroyAllWindows()
print("[INFO] Resources released successfully.")

