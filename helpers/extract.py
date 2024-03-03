import numpy as np


def extract_keypoints(results):
    """
    Extract keypoints from Mediapipe results.

    This function extracts keypoints from Mediapipe results for pose, face, left hand, and right hand.

    Args:

    - results: The results object from Mediapipe.

    Returns:

    - keypoints (numpy.ndarray): Array contains pose, face, left hand, and right hand keypoints.
    """

    # Extract pose keypoints or set to zeros if not available
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

    # Extract face keypoints or set to zeros if not available
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)

    # Extract left hand keypoints or set to zeros if not available
    left_hand_keypoints = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)

    # Extract right hand keypoints from results
    right_hand_keypoints = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)

    # Concatenate all keypoints
    return np.concatenate([pose, face, left_hand_keypoints, right_hand_keypoints])
