import tensorflow.keras as keras
import tensorflow as tf

# https://docs.djl.ai/docs/tensorflow/how_to_import_tensorflow_models_in_DJL.html

model_name = "resnet"
loaded_model = keras.models.load_model(f"{model_name}.h5")
tf.saved_model.save(loaded_model, f"build/tensorflow/{model_name}/1/")