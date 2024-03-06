import os.path
import cv2
import mediapipe as mp
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# Import helper functions for detection, extraction, and drawing landmarks
from helpers.detect import mediapipe_detection
from helpers.extract import extract_keypoints
from helpers.draw_landmarks import draw_styled_landmarks

# Initialize MediaPipe holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


left_hand_landmarks_list = []
right_hand_landmarks_list = []

# Structuring the folders for training data

DATA_PATH = os.path.join('MP_Data')
actions = np.array(['Hello','Thanks','ILoveYou'])

number_of_sequences = 30
sequences_length = 30

# Create a mapping from labels to numbers
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# Load the training data
for action in actions:
    for seq in range(1, 31):
        window = []
        for frame_num in range(1, sequences_length + 1):
            res = np.load(os.path.join(DATA_PATH, action, str(seq), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Convert sequences and labels to numpy arrays
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(y_test.shape)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Build the LSTM model and load the pre-trained weights
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, 'relu'))
model.add(Dense(32, 'relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.load_weights("ISLR.keras")
model.summary()

res = model.predict(X_train)

# Evaluate the model with confusion matrix and accuracy score

yhat = model.predict(X_train)

ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))
print(accuracy_score(ytrue, yhat))

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)] # Define colors for visualization


# Function to visualize the probabilities of actions
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    return output_frame


sequence = []
sentence = []
predictions = []
threshold = 0.4

cap = cv2.VideoCapture(0)

# Initialize the holistic model with a minimum detection confidence and tracking confidence of 0.5
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # Loop while the video capture is open
    while cap.isOpened():
        # Read a frame from the video capture
        ret, frame = cap.read()

        # Detect the landmarks in the frame using the holistic model
        image, results = mediapipe_detection(frame, holistic)
        # Draw the landmarks on the image
        draw_styled_landmarks(image, results)

        # Extract the keypoints from the results
        keypoints = extract_keypoints(results)
        # Insert the keypoints at the beginning of the sequence
        sequence.insert(0, keypoints)
        # Keep only the last 30 keypoints in the sequence
        sequence = sequence[:30]

        # If the sequence has 30 keypoints
        if len(sequence) == 30:
            # Predict the action in the sequence
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            # Print the predicted action
            print(actions[np.argmax(res)])
            # Append the index of the predicted action to the predictions
            predictions.append(np.argmax(res))

            # If the most common prediction in the last 10 predictions is the current prediction
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        # If the current action is different from the last action in the sentence
                        if actions[np.argmax(res)] != sentence[-1]:
                            # Append the current action to the sentence
                            sentence.append(actions[np.argmax(res)])
                    else:
                        # Append the current action to the sentence
                        sentence.append(actions[np.argmax(res)])

            # If the sentence has more than 5 actions
            if len(sentence) > 5:
                # Keep only the last 5 actions in the sentence
                sentence = sentence[-5:]

            # Visualize the probabilities of the actions on the image
            image = prob_viz(res, actions, image, colors)

        cv2.rectangle(image, (0, 0), (640, 60), (0,0,0), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        left_hand_landmarks_list.append(results.left_hand_landmarks)
        right_hand_landmarks_list.append(results.right_hand_landmarks)

        cv2.imshow("Sign language recognition with LSTM Neural network", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()


