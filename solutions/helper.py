import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from solutions import similarity_model, distinction_model

sim_model = similarity_model.get_main_model()
sim_model.load_weights(
    "/home/anshal/Projects/Signature-Forgery-Detection/models/similarity/signet_english_v3.h5"
)

dis_model = distinction_model.get_disiction_model()
dis_model.load_weights(
    "/home/anshal/Projects/Signature-Forgery-Detection/models/distinction/signet_v4.h5"
)


def load_and_preprocess_image(
    image_path, target_size=(155, 220), color_mode="grayscale"
):
    # Load image (as PIL image object)
    img = load_img(image_path, target_size=target_size, color_mode=color_mode)

    # Convert image to numpy array and scale it
    img_array = img_to_array(img) / 255.0

    # Add a dimension to mimic the batch size
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, img


def predict_function(image1, image2, threshold=30, use_cross=True):
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


def match_results(prediction: int, threshold=30) -> str:
    res = "UNK"
    similarity = max((1 - prediction[0][0]) * 100, 0)

    if prediction[0][0] * 100 < threshold:
        res = f"Real signature. Similarity = {(similarity):.2f}%, Distance = {prediction[0][0]:.4f}"

    else:
        res = f"Fake signature. Similarity = {similarity:.2f}%"

    return res, similarity
