import tensorflow as tf
import json

# Load images
data = tf.keras.utils.image_dataset_from_directory(
    r"C:\vaibhav\dog-cat-class\train",
    image_size=(64, 64),
    batch_size=2
)

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
model.fit(data, epochs=50)

# Model Summary
model.summary()

# Prediction
img = tf.keras.utils.load_img(
    r"C:\vaibhav\dog-cat-class\test\p3.jpg",
    target_size=(64,64)
)

x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)

p = model.predict(x)
print(data.class_names[p.argmax()])

# Save Model
model.save("CAT.keras")

# Save Class Names
with open("classes.json", "w") as f:
    json.dump(data.class_names, f)

# Load Class Names
with open("classes.json", "r") as f:
    class_names = json.load(f)

# Load Saved Model
m = tf.keras.models.load_model("CAT.keras")

# Test Another Image
img = tf.keras.utils.load_img(
    r"C:\vaibhav\dog-cat-class\test\p5.jpg",
    target_size=(64,64)
)

x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)

p = m.predict(x)

print(p)
print(class_names[p.argmax()])