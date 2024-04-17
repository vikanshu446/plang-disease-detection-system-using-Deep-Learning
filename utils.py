# Imporiting Necessary Libraries
import tensorflow as tf
import numpy as np
from PIL import Image


# Cleanig image    
def clean_image(image):
    # Converting PIL image to numpy array
    image = np.array(image)
    
    # Resizing the image to the desired shape without using ANTIALIAS
    image = np.array(Image.fromarray(image).resize((512, 512)))
    
    # Adding batch dimension to the image
    image = image[np.newaxis, :, :, :3]  # Keeping only the first 3 channels
    return image
    
    
def get_prediction(model, image):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    
    # Inputting the image to keras generators
    test = datagen.flow(image)
    
    # Predict from the image
    predictions = model.predict(test)
    predictions_arr = np.array(np.argmax(predictions))
    
    return predictions, predictions_arr
    

# Making the final results 
def make_results(predictions, predictions_arr):
    
    result = {}
    if int(predictions_arr) == 0:
        result = {"status": " is Healthy ",
                    "prediction": f"{int(predictions[0][0].round(2)*100)}%"}
    if int(predictions_arr) == 1:
        result = {"status": '  has Early_Blight ',
                    "prediction": f"{int(predictions[0][1].round(2)*100)}%"}
    if int(predictions_arr) == 2:
        result = {"status": ' has Late_Blight ',
                    "prediction": f"{int(predictions[0][2].round(2)*100)}%"}
    if int(predictions_arr) == 3:
        result = {"status": ' has Multiple Disease ',
                    "prediction": f"{int(predictions[0][3].round(2)*100)}%"}
    return result