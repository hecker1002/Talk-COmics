import easyocr
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.distance import cdist


''' Improve this Text Detection and assignment to Speech bubbles  '''

def print_dial( panel ) :
        
    # Load image
    img_path = panel 
    img = cv2.imread(img_path)

    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], verbose=False)
    text_results = reader.readtext(img_path)

    # Step 1: Extract speech bubbles with bounding boxes
    speech_bubbles = []

    for result in text_results:
        bbox, text, prob = result  
        x_min = min(bbox[0][0], bbox[3][0])  
        y_min = min(bbox[0][1], bbox[1][1])  
        speech_bubbles.append((x_min, y_min, text))

    # Step 2: Sort bubbles by Y (top to bottom) and X (left to right)
   
    speech_bubbles.sort(key=lambda x: (x[ 1 ], x[  0 ]))

    # Step 3: Merge close text regions into a single speech bubble
    merged_bubbles = []
    threshold = 5  # Adjust for better merging

    for x, y, text in speech_bubbles:
        if merged_bubbles and abs(y - merged_bubbles[-1][0][1]) < threshold:
            merged_bubbles[-1].append((x, y, text))  # we are tryong to MERGE  the text with nearest text ( NOT just append that why appended at alst (-1) idnex )

        else:
            merged_bubbles.append([(x, y, text)])

    # Step 4: Ensure text in bubbles is ordered correctly
    final_speech_bubbles = []

    # GOING INTO EACH BUBBLE AND JOING TOGETHER THE TEXT of eahc buble by " " and storing its extreme coorifnate with concatenated text 
    for bubble in merged_bubbles:
        bubble.sort(key=lambda x: x[0])  # Left to right sorting

        text_content = " ".join([item[2] for item in bubble])
        bubble_x, bubble_y = bubble[0][0], bubble[0][1]

        final_speech_bubbles.append((text_content, bubble_x, bubble_y))  # Store text + position

    # Load YOLOv8 for face detection
    face_model = YOLO("yolov11n-face.pt")  # Standard YOLOv8 model

    # Detect faces
    results = face_model(img)
    faces = []

    '''  reslts ( yolo_model( img )) ->  actuall alway form a lsit of proceesed img ( even if 1 img is apssed stil a lsit of 1 item )'''
    '''  for box in result.boxes.xyxy is actually traversign through each bbox ( ie throuhj rahc face detcted and catching its x , y ,coordiantes )'''
    for result in results:
        for box in result.boxes.xyxy:
            x_min, y_min, x_max, y_max = map(int, box) # convert coordiantes to int 

            # fidn midpoint of face 
            face_center_x = (x_min + x_max) // 2
            face_center_y = (y_min + y_max) // 2

            faces.append((face_center_x, face_center_y, y_min, y_max))  # Store face positions ( mid pint and upper head and lower chin )

    # # Ensure at least 2 faces are detected
    # if len(faces) < 2:
    #     print("Error: Less than 2 faces detected. Adjust YOLO threshold.")

    # Sort faces top to bottom
    faces.sort(key=lambda x: x[2])

    # we made a dict ( with face index (char) and is coorepsondig dialogue ( cuurently an empty list ))
    character_dialogues = {i: [] for i in range(len(faces))}

    ''' ONLY play with positions for now '''

    # Convert to NumPy arrays for distance calculation
    bubble_positions = np.array([(x, y) for _, x, y in final_speech_bubbles])
    face_positions = np.array([(x, y) for x, y, _, _ in faces])

    # Compute distances between speech bubbles and faces
    ''' Disatbce b/w COM of face and COM of speech bubble'''
    if face_positions.shape[0] > 0:
        distances = cdist(bubble_positions, face_positions) # cdis -> euclidean distance 


        for i, (text, x, y) in enumerate(final_speech_bubbles):
            closest_face_idx = np.argmin(distances[i])  # Get closest face index
            closest_face = faces[closest_face_idx]
            ''' COmpare the euclidean dis b.w centers of sppech bubble and center of face and when assign it to the face for which its min.
            and this we do for ALL detected sppech buble s( with tex t, pos (x , y ))'''

            # Ensure speech bubble is ABOVE or NEAR the character’s mouth
            face_top, face_bottom = closest_face[2], closest_face[3]
            if y < face_bottom + 30:  # Increased 50-pixel margin for accuracy
                character_dialogues[closest_face_idx].append(text)

    # Print structured conversation
    #print("=== Character Dialogues ===")

    speech = ""

    for char_id, dialogues in character_dialogues.items():
        user_name = f"User {char_id + 1}"
        speak = ""
        if dialogues:
            speak = f"{user_name}: " + " ".join(dialogues)
            #print(f"{user_name}: " + " ".join(dialogues))
        else:
            speak = f"{user_name}: [No detected dialogue]"
            #print(f"{user_name}: [No detected dialogue]")

        speech+=( speak  ) 
    
    return speech

def extract_speech_from_panels(panel_list):
    speech = []
    for panel in panel_list:
        spoke = print_dial(panel + ".jpg")
        speech.append(spoke)
    return speech
    

if __name__ == "__main__":
    panels = ['panel_0', 'panel_1', 'panel_2']  # Remove this when using Gradio
    speech = [print_dial(i + ".jpg") for i in panels]


    