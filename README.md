import cv2

# Load pre-trained face and emotion detection models
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = cv2.dnn.readNetFromTensorflow('emotion_detection.pb')

# Load the image
image = cv2.imread('image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Loop over each face
for (x, y, w, h) in faces:
    face_roi = gray[y:y+h, x:x+w]
    
    # Preprocess the face for emotion detection
    blob = cv2.dnn.blobFromImage(face_roi, 1.0, (48, 48), (0, 0, 0), swapRB=True, crop=False)
    emotion_model.setInput(blob)
    emotions = emotion_model.forward()
    
    # Get the dominant emotion
    emotion_label = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion_index = emotions[0].argmax()
    dominant_emotion = emotion_label[emotion_index]
    
    # Draw bounding box around the face and label the emotion
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with emotion labels
cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
