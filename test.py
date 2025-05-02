import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNet, MobileNetV2, VGG16, VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chemin vers le dataset HG14 (structure: HG14/class_name/*.jpg)
dataset_dir = "./HG14"

# Paramètres globaux
img_size = (128, 128)
batch_size = 20
epochs = 50
num_classes = 14

# Prétraitement et Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Fonction de construction d'un modèle base
def build_model(base_model_func):
    base = base_model_func(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
    # Freeze the base model layers for fine-tuning
    base.trainable = False  # You can later unfreeze specific layers for fine-tuning
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Entraîner plusieurs modèles
model_fns = {
    "MobileNet": MobileNet,
    "MobileNetV2": MobileNetV2,
    "VGG16": VGG16,
    "VGG19": VGG19
}

trained_models = {}
val_preds = []
history_dict = {}

for name, model_fn in model_fns.items():
    print(f" Entraînement de {name}...")
    model = build_model(model_fn)
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=1)
    trained_models[name] = model
    history_dict[name] = history.history
    
    # Prédictions sur l'ensemble de validation pour pondération
    val_generator.reset()
    pred = model.predict(val_generator, verbose=0)
    val_preds.append(pred)

    # Sauvegarder les poids du modèle
    model.save(f"{name}_weights.h5")

# Fonction d'ensemble avec pondération Dirichlet simulée
weights = np.random.dirichlet(np.ones(len(val_preds)), size=1)[0]
print(f"Pondérations Dirichlet simulées : {weights}")

ensemble_pred = sum(w * p for w, p in zip(weights, val_preds))
ensemble_labels = np.argmax(ensemble_pred, axis=1)
true_labels = np.argmax(np.concatenate([val_generator[i][1] for i in range(len(val_generator))]), axis=1)

# Évaluer la performance
accuracy = accuracy_score(true_labels, ensemble_labels)
print(f" Accuracy de l'ensemble (simulé Dirichlet) : {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(true_labels, ensemble_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=val_generator.class_indices.keys(), yticklabels=val_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot ROC curve and AUC
fpr, tpr, roc_auc = {}, {}, {}

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(true_labels == i, ensemble_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# ROC Curves for each class
plt.figure(figsize=(10, 7))
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Class')
plt.legend(loc='best')
plt.show()

# Plot Training/Validation Loss and Accuracy Curves
for name, history in history_dict.items():
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'{name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Accuracy plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Class-wise Accuracy Bar Plot
class_accuracies = np.mean(np.equal(np.argmax(ensemble_pred, axis=1), true_labels), axis=0)
plt.figure(figsize=(10, 7))
plt.bar(range(num_classes), class_accuracies)
plt.xticks(range(num_classes), val_generator.class_indices.keys(), rotation=90)
plt.title('Class-wise Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.show()

# Optionally, if you want to fine-tune after the initial training:
for name, model in trained_models.items():
    print(f"Fine-tuning {name}...")
    # Unfreeze some layers of the base model for fine-tuning
    base_model = model.layers[0]
    base_model.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=1)

    # Re-save the weights after fine-tuning
    model.save(f"{name}_fine_tuned_weights.h5")
