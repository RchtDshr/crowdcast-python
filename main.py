import cv2
import requests
import numpy as np
from pymongo import MongoClient
from collections import Counter

# MongoDB connection
client = MongoClient("mongodb+srv://rach:crowdcast-backend@cluster0.ullhh.mongodb.net")
db = client["test"]
location_collection = db["locations"]
advertisement_collection = db["advertisements"]

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

# Function to fetch ad IDs based on location
def fetch_file_upload(location, age_cluster_key, gender):
    try:
        url = "http://localhost:5000/api/fetchAdIds"
        response = requests.post(url, json={"locationName": location, "ageGroup": str(age_cluster_key), "gender": str(gender)})
        
        if response.status_code == 200:
            ad_ids = response.json().get("adIds", [])
            file_uploads = []
            for ad in ad_ids:
                ad_url = f"http://localhost:5000/api/{ad}"
                ad_response = requests.get(ad_url, json={"adId": ad})
                
                if ad_response.status_code == 200:
                    ad_data = ad_response.json().get('ad', {})
                    file_upload = ad_data.get('fileUpload')
                    file_type = ad_data.get('type')
                    if file_upload:
                        file_uploads.append(file_upload)
                        print("File Upload URL:", file_upload)
                        display_ad(file_upload, file_type)
                        
                else:
                    print(f"Failed to fetch ad data for ad ID {ad}. Status code: {ad_response.status_code}")
            
            return file_uploads
        else:
            print(f"Failed to fetch ad IDs. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"An error occurred while fetching ad IDs: {e}")
        return []

# Function to display ads based on predicted age group and gender
def show_ad_based_on_prediction(location, age_cluster_key, gender):
    try:
        ad_ids = fetch_file_upload(location, age_cluster_key, gender)
        
        if not ad_ids:
            print("No ad IDs retrieved.")
            return

        for ad_id in ad_ids:
            ad_data = advertisement_collection.find_one({"_id": ad_id})
            if ad_data:
                ad_url = ad_data["fileUpload"]  # Cloudinary URL
                ad_type = ad_data["type"]  # Get type field to check if it's an image or video
                print(f"Displaying {ad_type} ad from URL:", ad_url)
    except Exception as e:
        print(f"An error occurred in show_ad_based_on_prediction: {e}")

def display_ad(ad_url, ad_type):
    if ad_type == "image":
        # Display as image
        resp = requests.get(ad_url)
        if resp.status_code == 200:
            ad_data = np.frombuffer(resp.content, np.uint8)
            ad_img = cv2.imdecode(ad_data, cv2.IMREAD_COLOR)
            if ad_img is not None:
                cv2.imshow("Advertisement", ad_img)
                cv2.waitKey(0)  # Wait until a key is pressed to close ad
            else:
                print("Failed to decode image.")
        else:
            print("Failed to fetch image from URL.")
    
    elif ad_type == "video":
        # Display as video
        cap = cv2.VideoCapture(ad_url)
        if not cap.isOpened():
            print("Failed to fetch video from URL.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Advertisement", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to stop the video
                break

        cap.release()
    else:
        print("Unsupported ad format.")

    cv2.destroyAllWindows()


# Model paths
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

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
        elif mode_age is None and mode_gender is None:
            mode_age = Counter(age_predictions).most_common(1)[0][0]
            mode_gender = Counter(gender_predictions).most_common(1)[0][0]
            print(f"Predicted Gender: {genderList[mode_gender]}, Predicted Age Cluster: {mode_age}")
            fetch_file_upload(device_location, age_cluster_key, genderList[mode_gender])

        label = f"{genderList[mode_gender] if mode_gender is not None else gender}, {ageList[mode_age] if mode_age is not None else age}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Smart Advertisement", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
