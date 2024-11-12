import cv2
import requests
import numpy as np
from collections import Counter
import time

token='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyIjp7ImlkIjoiNjcyZThlZDYzM2VkZDhjYzVkMzU0NDNjIn0sImlhdCI6MTczMTM1MjMxN30.3CdlPew-L0-fHlDHXlUWh5ZHKZXh8fJiIxsSHnuHw-g'

# Device-specific location for ad targeting
device_location = "Railway Stations"

# Map age clusters and genders for database compatibility
ageList = {
    1: '(0-2)',
    2: '(4-6)',
    3: '(8-12)',
    4: '(15-20)',
    5: '(25-32)',
    6: '(38-43)',
    7: '(48-53)',
    8: '(60-100)'
}
genderList = {0: 'M', 1: 'F'}

def fetch_file_upload(location, age_cluster_key, gender):
    url = "http://localhost:5000/display/fetchAdIds"
    
    response = requests.post(url, json={"locationName": location, "ageGroup": str(age_cluster_key), "gender": str(gender)})
    
    # Step 3: Checking if the request for ad IDs was successful
    if response.status_code == 200:
        adIds = response.json().get('adIds', [])
        print(f"Fetched ads: {adIds}")
        
        # Step 4: Looping through each ad ID
        for ad_id in adIds:
            print(f"Processing ad ID: {ad_id}")
            ad_url = f"http://localhost:5000/display/{ad_id}"
            
            ad_response = requests.get(ad_url, json={"adId": ad_id})

            # Step 5: Checking if fetching ad data was successful
            if ad_response.status_code == 200:
                ad_data = ad_response.json().get('ad', {})
                file_upload = ad_data.get('fileUpload')
                file_type = ad_data.get('type')
                userId=ad_data.get('userId')
                adName=ad_data.get('adName')
                deductedAmount=ad_data.get('creditsDeducted')

                print(f"Ad data - File Upload: {file_upload}, Type: {file_type}")

                # Step 6: If file_upload is found, call display_ad
                if file_upload:
                    
                    reducePointsUrl=f"http://localhost:5000/display/reduceCredits"
                    res = requests.post(reducePointsUrl, json={
                        "userId": userId,
                        "credits": deductedAmount
                    })
                    if(res.status_code==200):
                        print('Points Deducted: ', deductedAmount)
                        timeline_url = "http://localhost:5000/display/addToTimeline"
                        res = requests.post(timeline_url, json={
                        "userId": userId,
                        "adId": ad_id,
                        "adName": adName,
                        "locationName": location,
                        "deductedAmount": deductedAmount
                        })
                        if res.status_code == 201:
                            display_ad(file_upload, file_type)
                        else:
                            print(f"failed to play ad {res.status_code}")    
                else:
                    print(f"No file upload found for ad ID: {ad_id}")
            else:
                print(f"Failed to fetch ad data for ad ID {ad_id}. Status code: {ad_response.status_code}")
    else:
        print(f"Failed to fetch ad IDs. Status code: {response.status_code}")


def display_ad(ad_url, ad_type):
    start_time = time.time()

    # Create a named window and set it to fullscreen mode
    cv2.namedWindow("Advertisement", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Advertisement", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if ad_type == "image":
        resp = requests.get(ad_url)
        if resp.status_code == 200:
            ad_data = np.frombuffer(resp.content, np.uint8)
            ad_img = cv2.imdecode(ad_data, cv2.IMREAD_COLOR)
            if ad_img is not None:
                cv2.imshow("Advertisement", ad_img)
                while cv2.getWindowProperty("Advertisement", cv2.WND_PROP_VISIBLE) >= 1:
                    if time.time() - start_time > 10:  # Display for 10 seconds
                        break
                    cv2.waitKey(1)
            else:
                print("Failed to decode image.")
        else:
            print("Failed to fetch image from URL.")
    
    elif ad_type == "video":
        cap = cv2.VideoCapture(ad_url)
        if not cap.isOpened():
            print("Failed to fetch video from URL.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (time.time() - start_time > 10):  # Display for 10 seconds or until video ends
                break
            cv2.imshow("Advertisement", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to stop the video early
                break

        cap.release()
    else:
        print("Unsupported ad format.")

    cv2.destroyAllWindows()



# Paths to models
faceProto = "d:/CrowdCast/caffe model/opencv_face_detector.pbtxt"
faceModel = "d:/CrowdCast/caffe model/opencv_face_detector_uint8.pb"
ageProto = "d:/CrowdCast/caffe model/age_deploy.prototxt"
ageModel = "d:/CrowdCast/caffe model/age_net.caffemodel"
genderProto = "d:/CrowdCast/caffe model/gender_deploy.prototxt"
genderModel = "d:/CrowdCast/caffe model/gender_net.caffemodel"

def faceBox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, bboxs

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Open video capture
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video stream or file.")
    exit()

age_predictions = []
gender_predictions = []
mode_age = None
mode_gender = None
frame_count = 0
initial_frames = 100

while True:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break

    # Detect faces and predict age and gender
    frame, bboxs = faceBox(faceNet, frame)

    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender with error handling
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender_key = genderPred[0].argmax()
        gender = genderList.get(gender_key, 'Unknown')

        # Predict age with error handling
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age_cluster_key = agePred[0].argmax() + 1  # Age cluster numbers start from 1
        age = ageList.get(age_cluster_key, 'Unknown')

        if frame_count < initial_frames:
            age_predictions.append(age_cluster_key)
            gender_predictions.append(gender_key)
            frame_count += 1
        else:
            # Calculate the most common age and gender from predictions
            mode_age = Counter(age_predictions).most_common(1)[0][0]
            mode_gender = Counter(gender_predictions).most_common(1)[0][0]
            print(f"Predicted Gender: {genderList[mode_gender]}, Predicted Age Cluster: {mode_age}")
            fetch_file_upload(device_location, mode_age, genderList[mode_gender])
            
            # Reset for the next prediction cycle
            age_predictions.clear()
            gender_predictions.clear()
            frame_count = 0
        label = f"{genderList[mode_gender] if mode_gender is not None else gender}, {ageList[mode_age] if mode_age is not None else age}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Smart Advertisement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
