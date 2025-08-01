import cv2
import mediapipe as mp
import os
import numpy as np

label = input("Enter label (A–Z): ").upper()
save_dir = f"LandmarkDataset/{label}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
draw = mp.solutions.drawing_utils

count = 0
start = False

print("👉 Press 's' to start saving landmark images")
print("👉 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    # White canvas
    white = 255 * np.ones((400, 400, 3), dtype=np.uint8)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            draw.draw_landmarks(
                white, hand_landmark, mp.solutions.hands.HAND_CONNECTIONS,
                draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                draw.DrawingSpec(color=(0, 0, 0), thickness=2)
            )

            # Save landmark image
            if start:
                filename = os.path.join(save_dir, f"{count}.jpg")
                resized = cv2.resize(white, (400, 400))
                cv2.imwrite(filename, resized)
                print(f"✅ Saved {filename}")
                count += 1

    # Overlay text on canvas
    cv2.putText(white, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(white, f"Count: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
    cv2.imshow("Landmark Canvas", white)

    key = cv2.waitKey(1)
    if key == ord('s'):
        start = True
        print("📸 Started capturing...")
    elif key == ord('q') or count >= 500:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Finished saving {count} images for label '{label}'")
