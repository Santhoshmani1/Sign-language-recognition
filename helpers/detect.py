import cv2


def mediapipe_detection(image, model):
    """
     Perform detection using a Mediapipe model.

     Args:

     - image (numpy.ndarray): The input image in BGR format.

     - model: The Mediapipe model for detection.

     Returns:

     - image (numpy.ndarray): The processed image in BGR format.

     - results: The results of the detection process.
     """

    # Converting from BG2 to RGB format in order to process the image with original colors & preventing cv2 color filter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False  # rewriting the writable flag to prevent accidental modifications to the image
    results = model.process(image)
    image.flags.writeable = True   # re-enabling the writable Flag to True after inference

    # Restoring image back to BGR format from RGB format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results
