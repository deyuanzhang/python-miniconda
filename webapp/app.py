import io, traceback

from flask import Flask, request, g, jsonify, send_file
from flask import send_file
from PIL import Image, ExifTags
from scipy.misc import imresize
import numpy as np 
from keras.models import load_model
import tensorflow as tf

app = Flask(__name__, instance_relative_config=True)

print("Loading model")
model = load_model('./model/tiramisu_2_classes_with_weights.h5')
graph = tf.get_default_graph()

def ml_predict(image):
	with graph.as_default():
		prediction = model.predict(image[None, :, :, :])
	prediction = prediction.reshape((224, 224, -1))
	return prediction

def rotate_by_exif(image):
    try :
        for orientation in ExifTags.TAGS.keys() :
            if ExifTags.TAGS[orientation]=='Orientation' : break
        exif=dict(image._getexif().items())
        if not orientation in exif:
            return image

        if   exif[orientation] == 3 :
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6 :
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8 :
            image=image.rotate(90, expand=True)
        return image
    except:
        traceback.print_exc()
        return image

@app.route('/predict', methods=['POST'])
def predict():
	image = request.files['file']
	image = Image.open(image)
	image = rotate_by_exif(image)
	resized_image = imresize(image, (224, 224, -1))

	prediction = ml_predict(resized_image[:, :, 0:3])
	prediction = imresize(prediction[:, :, 1], (image.height, image.width))

	transparent_image = np.append(np.array(image)[:, :, 0:3], prediction[: , :, None], axis=-1)
	transparent_image = Image.fromarray(transparent_image)

	byte_io = io.BytesIO()
	transparent_image.save(byte_io, 'PNG')
	byte_io.seek(0)
	return send_file(byte_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
