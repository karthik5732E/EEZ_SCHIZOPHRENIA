import tensorflow as tf

# Load original model
model = tf.keras.models.load_model("schizophrenia_cnn_model.keras", compile=False)

# Save in H5 format (more compatible)
model.save("schizophrenia_model.h5")

print("Model converted successfully!")