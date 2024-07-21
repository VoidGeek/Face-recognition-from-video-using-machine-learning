import cv2 as cv
import numpy as np

def visualize(image, faces, result_text=None, similarity_score=None, thickness=2):
    for face in faces:
        coords = face[:-1].astype(np.int32)
        cv.rectangle(image, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
        cv.circle(image, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
        cv.circle(image, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
        cv.circle(image, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
        cv.circle(image, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
        cv.circle(image, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
        
        # Display coordinates in terminal
        print(f"Coordinates: {coords}")

        # Display similarity score if available
        if similarity_score is not None:
            cv.putText(image, f"Similarity Score: {similarity_score:.2f}", (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
            print(f"Similarity Score: {similarity_score:.2f}")

    if result_text:
        cv.putText(image, result_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        
        # Display result text in terminal
        print(f"Result: {result_text}")

# Path to the reference image
reference_image_path = "E:\\AIML\\AIML-Face-Recognition-Project\\query(1).png"

ref_image = cv.imread(reference_image_path)  # Read the reference image

score_threshold = 0.9
nms_threshold = 0.3
top_k = 5000

# Initialize face detector
faceDetector = cv.FaceDetectorYN.create("face_detection_yunet_2023mar.onnx", "", (ref_image.shape[1], ref_image.shape[0]), score_threshold, nms_threshold, top_k)
faceInAdhaar = faceDetector.detect(ref_image)

if faceInAdhaar[1] is not None:
    visualize(ref_image, faceInAdhaar[1])
    cv.imshow("Reference Image", ref_image)  # Display the reference image
    cv.waitKey(0)

# Initialize face recognizer
recognizer = cv.FaceRecognizerSF.create("face_recognition_sface_2021dec.onnx", "")

# Open a connection to the webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

l2_similarity_threshold = 1.128

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image from webcam.")
        break

    faceDetector.setInputSize((frame.shape[1], frame.shape[0]))
    faceInQuery = faceDetector.detect(frame)

    result_text = None

    if faceInQuery[1] is not None:
        # Perform face recognition and matching
        if faceInAdhaar[1] is not None:
            face1_align = recognizer.alignCrop(ref_image, faceInAdhaar[1][0])
            face2_align = recognizer.alignCrop(frame, faceInQuery[1][0])

            face1_feature = recognizer.feature(face1_align)
            face2_feature = recognizer.feature(face2_align)

            l2_score = recognizer.match(face1_feature, face2_feature, cv.FaceRecognizerSF_FR_NORM_L2)
            
            # Display similarity score in terminal
            print(f"L2 Similarity Score: {l2_score}")

            if l2_score <= l2_similarity_threshold:
                result_text = "Same Identity"
            else:
                result_text = "Different Identity"

            visualize(frame, faceInQuery[1], result_text, l2_score)  # Pass similarity score to visualize function
        else:
            visualize(frame, faceInQuery[1], result_text)
    else:
        visualize(frame, [], result_text)

    cv.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("Live Video", frame)  # Display the live video with detected faces and result text

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
