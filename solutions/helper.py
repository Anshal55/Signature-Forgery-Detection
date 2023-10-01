import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from solutions import similarity_model, distinction_model

# Load the similarity model and its pre-trained weights
sim_model = similarity_model.get_similarity_model()
sim_model.load_weights(
    "/home/anshal/Projects/Signature-Forgery-Detection/models/similarity/signet_english_v3.h5"
)

# Load the distinction model and its pre-trained weights
dis_model = distinction_model.get_disiction_model()
dis_model.load_weights(
    "/home/anshal/Projects/Signature-Forgery-Detection/models/distinction/signet_v4.h5"
)


def load_and_preprocess_image(
    image_path: str, target_size: tuple = (155, 220), color_mode: str = "grayscale"
):
    """
    Load and preprocess an image.

    Args:
        image_path (str): The path to the image file.
        target_size (tuple): The target size (height, width) for resizing the image.
        color_mode (str): The color mode for loading the image ('grayscale' or 'rgb').

    Returns:
        tuple: A tuple containing a preprocessed image as a NumPy array and the original PIL image object.
    """
    # Load the image as a PIL image object
    img = load_img(image_path, target_size=target_size, color_mode=color_mode)

    # Convert the image to a NumPy array and scale it
    img_array = img_to_array(img) / 255.0

    # Add a dimension to mimic the batch size
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img


def predict_function(image1, image2, threshold: int = 30, use_cross: bool = True):
    """
    Predict whether two images represent real or fake signatures.

    Args:
        image1 (numpy.ndarray): The preprocessed image data for the first image.
        image2 (numpy.ndarray): The preprocessed image data for the second image.
        threshold (int): The threshold for classifying the images (default is 30).
        use_cross (bool): Whether to use cross-checking with the distinction model (default is True).

    Returns:
        str: A string indicating the prediction result.
    """
    if use_cross:
        initial_pred = dis_model.predict([image1, image2])
        init_res, similarity = match_results(initial_pred, threshold=threshold)
        print("First Check: ", init_res)
        if similarity <= 70:
            return init_res

    prediction = sim_model.predict([image1, image2])
    res, _ = match_results(prediction, threshold=threshold)

    print("Second Check: ", res)

    return res


def match_results(prediction: np.ndarray, threshold: int = 30) -> tuple:
    """
    Match and classify images based on a prediction.

    Args:
        prediction (numpy.ndarray): The prediction result.
        threshold (int): The threshold for classifying the images (default is 30).

    Returns:
        tuple: A tuple containing the classification result as a string and the similarity score.
    """
    res = "UNK"
    similarity = max((1 - prediction[0][0]) * 100, 0)

    if prediction[0][0] * 100 < threshold:
        res = f"Real signature. Similarity = {(similarity):.2f}%, Distance = {prediction[0][0]:.4f}"
    else:
        res = f"Fake signature. Similarity = {similarity:.2f}%"

    return res, similarity
