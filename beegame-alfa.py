import cv2
import mediapipe as mp

# Initialize FaceMesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Load bee image
bee_img = cv2.imread('bee-1.png', cv2.IMREAD_UNCHANGED)
bee_x, bee_y = 0, 0  # starting position of bee

prev_left_hand_center = None
prev_right_hand_center = None
bee_kicked = False
bee_kick_direction = (0, 0)
frame_counter = 0
# Given a current position (x, y) and a target position (target_x, target_y), 
# this function returns new coordinates (new_x, new_y) closer to the target.
def move_towards(x, y, target_x, target_y, step=1):
    if x < target_x:
        x += step
    elif x > target_x:
        x -= step

    if y < target_y:
        y += step
    elif y > target_y:
        y -= step

    return x, y

def compute_hand_center(hand_landmarks):
    all_x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    all_y_coords = [landmark.y for landmark in hand_landmarks.landmark]
    center_x = int(sum(all_x_coords) / len(all_x_coords) * frame_width)
    center_y = int(sum(all_y_coords) / len(all_y_coords) * frame_height)
    return center_x, center_y


def compute_face_center(face_landmarks):
    all_x_coords = [landmark.x for landmark in face_landmarks.landmark]
    all_y_coords = [landmark.y for landmark in face_landmarks.landmark]
    
    center_x = int(sum(all_x_coords) / len(all_x_coords) * frame_width)
    center_y = int(sum(all_y_coords) / len(all_y_coords) * frame_height)

    return center_x, center_y


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    bg_img = background_img.copy()
    
    if bg_img.shape[2] == 3:
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t, overlay_size)

    b, g, r, a = cv2.split(img_to_overlay_t)
    mask = cv2.medianBlur(a, 5)

    h, w, _ = img_to_overlay_t.shape

    # Ensure x, y, h, and w values do not exceed the frame's dimensions
    y = max(0, min(y, bg_img.shape[0] - h))
    x = max(0, min(x, bg_img.shape[1] - w))

    roi = bg_img[y:y+h, x:x+w]
    
    if roi.shape[2] == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2BGRA)

    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

    return bg_img



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Set the resolution to Full HD
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:

        print("Failed to grab the frame.")
        continue
    if ret:
        frame_counter += 1
        # Mirror the frame horizontally
    
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  #  if frame_counter % 10 == 0:
       
    # Process the frame and get the hand landmarks
    hand_results = hands.process(rgb_frame)

    # If hand landmarks are found, draw them
    if hand_results.multi_hand_landmarks:
        for landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            hand_center = compute_hand_center(landmarks)
    
            # Dummy function to determine if the hand is left or right
            # This function can be improved based on MediaPipe's handedness detection
            def is_left_hand(landmarks):
                return landmarks.landmark[mp_hands.HandLandmark.WRIST].x < 0.5

            if is_left_hand(landmarks):  
                if prev_left_hand_center:
                    left_direction = (hand_center[0] - prev_left_hand_center[0], hand_center[1] - prev_left_hand_center[1])
                prev_left_hand_center = hand_center
            else:
                if prev_right_hand_center:
                    right_direction = (hand_center[0] - prev_right_hand_center[0], hand_center[1] - prev_right_hand_center[1])
                prev_right_hand_center = hand_center
            collision_threshold = 30  # Adjust as necessary
            left_distance_to_bee = ((bee_x - prev_left_hand_center[0]) ** 2 + (bee_y - prev_left_hand_center[1]) ** 2) ** 0.5 if prev_left_hand_center else float('inf')
            right_distance_to_bee = ((bee_x - prev_right_hand_center[0]) ** 2 + (bee_y - prev_right_hand_center[1]) ** 2) ** 0.5 if prev_right_hand_center else float('inf')

            # Check if either hand has "kicked" the bee
            if left_distance_to_bee < collision_threshold:
                bee_kicked = True
                bee_kick_direction = left_direction
            elif right_distance_to_bee < collision_threshold:
                bee_kicked = True
                bee_kick_direction = right_direction

            bee_speed = 1
            if bee_kicked:
                bee_x += bee_kick_direction[0] * bee_speed
                bee_y += bee_kick_direction[1] * bee_speed

            if bee_x < 0 or bee_x > frame_width or bee_y < 0 or bee_y > frame_height:
                bee_kicked = False



    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

    
        for face_landmarks in results.multi_face_landmarks:
    # ... (rest of the code for drawing and checking collision)

            # Assuming you have some way to compute the face's center, for instance:
            face_center_x, face_center_y = compute_face_center(face_landmarks) 
            # Note: You'll need to implement compute_face_center.

            # Move the bee towards the face's center
            bee_x, bee_y = move_towards(bee_x, bee_y, face_center_x, face_center_y, step=2) # step can be adjusted as per your requirement
        
            collision = False
            for landmark in face_landmarks.landmark:
                lm_x, lm_y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                if abs(bee_x - lm_x) < 30 and abs(bee_y - lm_y) < 30:  # threshold for collision
                    collision = True
                    break

            if collision:
                    # Drawing a rectangle around the face
                    h, w, _ = frame.shape
                    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 2)  # Red color

            

                # Draw face landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
                    #  mp_drawing.draw_landmarks(frame, face_landmarks, drawing_spec=drawing_spec)

# Display the mirrored frame
   # Overlay bee image onto the frame
    frame = overlay_transparent(frame, bee_img, bee_x, bee_y)
    #cv2.imshow("Mirrored Webcam Feed", mirrored_frame)
    cv2.imshow("Bee Attack Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

face_mesh.close()
cap.release()
cv2.destroyAllWindows()
