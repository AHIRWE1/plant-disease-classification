"""
MobileNetV2 transfer-learning model for PlantVillage classification.
"""


def build_model(
    number_of_classes,
    input_shape=(224, 224, 3),
    weights="imagenet",
    dropout_rate=0.2,
    learning_rate=1e-3
):
    """
    Create a frozen MobileNetV2 feature extractor and custom classifier head.
    """
    import tensorflow as tf

    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2.")
    if weights not in ("imagenet", None):
        raise ValueError("weights must be 'imagenet' or None.")
    if not 0 <= dropout_rate < 1:
        raise ValueError("dropout_rate must be in the range [0, 1).")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="leaf_image")
    features = base_model(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(features)
    features = tf.keras.layers.Dropout(
        dropout_rate,
        name="classifier_dropout"
    )(features)
    outputs = tf.keras.layers.Dense(
        number_of_classes,
        activation="softmax",
        name="class_probabilities"
    )(features)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_classifier")
    compile_model(model, learning_rate)
    return model, base_model


def compile_model(model, learning_rate):
    """Compile the classifier for integer-encoded labels."""
    import tensorflow as tf

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


def unfreeze_last_layers(base_model, number_of_layers=20):
    """
    Unfreeze the last requested non-BatchNormalization layers for fine-tuning.
    """
    import tensorflow as tf

    if number_of_layers < 1:
        raise ValueError("number_of_layers must be positive.")

    base_model.trainable = True
    for layer in base_model.layers:
        layer.trainable = False

    unfrozen_layers = 0
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            continue
        if unfrozen_layers == number_of_layers:
            break
        layer.trainable = True
        unfrozen_layers += 1

    if unfrozen_layers == 0:
        raise ValueError("The base model has no layers available to unfreeze.")

    return unfrozen_layers
