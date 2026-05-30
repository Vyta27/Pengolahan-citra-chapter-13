"""
Tugas 3: Transfer Learning dengan Dataset Buah
Perbandingan MobileNetV2 (Transfer Learning) vs CNN dari Nol
Dataset: Fruit and Vegetable Image Recognition (Kaggle)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─── Konfigurasi Dataset ───────────────────────────────────────────────────────
# Sesuaikan path ini dengan lokasi folder hasil extract di komputer kamu
DATASET_PATH = "archive (1)/images"   # path utama berisi subfolder kelas

IMG_SIZE     = 100      # ukuran gambar yang dipakai (100x100)
BATCH_SIZE   = 32
EPOCHS_PHASE1 = 10     # fase freeze (latih classifier saja)
EPOCHS_PHASE2 = 10     # fase fine-tuning (buka 20 layer terakhir)
NUM_CLASSES  = 9        # jumlah kelas buah/sayur dalam dataset
LEARNING_RATE_FINE = 1e-5

# ─── Import Library ────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("=" * 60)
print("  TUGAS 3: Transfer Learning vs CNN dari Nol")
print("=" * 60)
print(f"TensorFlow versi: {tf.__version__}")

# ─── 1. Load Dataset ───────────────────────────────────────────────────────────
print("\n[1/7] Memuat dataset...")

# Cek apakah path ada
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Folder '{DATASET_PATH}' tidak ditemukan!")
    print("Pastikan dataset sudah di-extract dan path di atas sudah benar.")
    print("Contoh struktur yang benar:")
    print("  archive (1)/")
    print("    images/")
    print("      apple/")
    print("      banana/")
    print("      ...")
    exit()

# Ambil daftar kelas
kelas_list = sorted([d for d in os.listdir(DATASET_PATH)
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
print(f"Kelas yang ditemukan ({len(kelas_list)}): {kelas_list}")

# ─── 2. Visualisasi Contoh Gambar ─────────────────────────────────────────────
print("\n[2/7] Membuat visualisasi contoh gambar...")

fig, axes = plt.subplots(2, 5, figsize=(15, 7))
fig.suptitle("Contoh Gambar Dataset Buah & Sayur", fontsize=16, fontweight='bold')
axes = axes.flatten()

for i, kelas in enumerate(kelas_list[:10]):
    folder = os.path.join(DATASET_PATH, kelas)
    gambar_list = [f for f in os.listdir(folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if gambar_list:
        from PIL import Image
        img_path = os.path.join(folder, gambar_list[0])
        img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
        axes[i].imshow(img)
        axes[i].set_title(kelas.capitalize(), fontsize=10, fontweight='bold')
        axes[i].axis('off')

for j in range(len(kelas_list), 10):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("contoh_buah_dataset.png", dpi=150, bbox_inches='tight')
print("  → Tersimpan: contoh_buah_dataset.png")
plt.show()

# ─── 3. Persiapan Data Generator ──────────────────────────────────────────────
print("\n[3/7] Menyiapkan data generator...")

# Augmentasi untuk training (mencegah overfitting)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Validasi hanya rescale, tanpa augmentasi
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

num_classes = len(train_gen.class_indices)
class_names = list(train_gen.class_indices.keys())
print(f"  Jumlah kelas: {num_classes}")
print(f"  Data training: {train_gen.samples} gambar")
print(f"  Data validasi: {val_gen.samples} gambar")

# ─── 4. Buat Model CNN dari Nol ───────────────────────────────────────────────
print("\n[4/7] Membangun model CNN dari nol...")

def buat_cnn_dari_nol(num_classes, img_size=100):
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                      input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 2
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 3
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Block 4
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name="CNN_dari_Nol")
    return model

model_cnn = buat_cnn_dari_nol(num_classes, IMG_SIZE)
model_cnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model_cnn.summary()

# ─── 5. Buat Model Transfer Learning (MobileNetV2) ────────────────────────────
print("\n[5/7] Membangun model Transfer Learning (MobileNetV2)...")

def buat_transfer_learning(num_classes, img_size=100):
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False   # Freeze semua layer (Fase 1)

    inputs  = keras.Input(shape=(img_size, img_size, 3))
    x       = base_model(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_TransferLearning")
    return model, base_model

model_tl, base_model = buat_transfer_learning(num_classes, IMG_SIZE)
model_tl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ─── 6. Training ──────────────────────────────────────────────────────────────
print("\n[6/7] Melatih model...")
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
]

# --- Training CNN dari Nol ---
print("\n--- Training CNN dari Nol ---")
hist_cnn = model_cnn.fit(
    train_gen,
    epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# --- Transfer Learning: Fase 1 (Freeze) ---
print("\n--- Transfer Learning Fase 1: Freeze base model ---")
train_gen.reset(); val_gen.reset()
hist_tl_p1 = model_tl.fit(
    train_gen,
    epochs=EPOCHS_PHASE1,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# --- Transfer Learning: Fase 2 (Fine-tuning) ---
print("\n--- Transfer Learning Fase 2: Fine-tuning 20 layer terakhir ---")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model_tl.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_gen.reset(); val_gen.reset()
hist_tl_p2 = model_tl.fit(
    train_gen,
    epochs=EPOCHS_PHASE2,
    validation_data=val_gen,
    callbacks=callbacks,
    verbose=1
)

# Gabung history transfer learning
hist_tl_acc     = hist_tl_p1.history['accuracy']     + hist_tl_p2.history['accuracy']
hist_tl_val_acc = hist_tl_p1.history['val_accuracy'] + hist_tl_p2.history['val_accuracy']
hist_tl_loss    = hist_tl_p1.history['loss']         + hist_tl_p2.history['loss']
hist_tl_val_los = hist_tl_p1.history['val_loss']     + hist_tl_p2.history['val_loss']

# ─── 7. Evaluasi & Visualisasi ────────────────────────────────────────────────
print("\n[7/7] Evaluasi dan visualisasi hasil...")

val_gen.reset()
pred_cnn = np.argmax(model_cnn.predict(val_gen, verbose=0), axis=1)
val_gen.reset()
pred_tl  = np.argmax(model_tl.predict(val_gen, verbose=0), axis=1)
true_labels = val_gen.classes

acc_cnn = np.mean(pred_cnn == true_labels)
acc_tl  = np.mean(pred_tl  == true_labels)
print(f"\nAkurasi CNN dari Nol    : {acc_cnn*100:.2f}%")
print(f"Akurasi Transfer Learning: {acc_tl*100:.2f}%")

# ─── Plot Lengkap ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Hasil Perbandingan Transfer Learning vs CNN dari Nol",
             fontsize=18, fontweight='bold', y=0.98)

# (a) Kurva Akurasi Training
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(hist_cnn.history['accuracy'],     label='CNN Train', linewidth=2)
ax1.plot(hist_cnn.history['val_accuracy'], label='CNN Val',   linewidth=2, linestyle='--')
ax1.plot(hist_tl_acc,                      label='TL Train',  linewidth=2)
ax1.plot(hist_tl_val_acc,                  label='TL Val',    linewidth=2, linestyle='--')
ax1.axvline(EPOCHS_PHASE1, color='gray', linestyle=':', label=f'Fine-tune mulai ep.{EPOCHS_PHASE1}')
ax1.set_title("Kurva Akurasi Training", fontweight='bold')
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Akurasi")
ax1.legend(); ax1.grid(True, alpha=0.3)

# (b) Kurva Loss Training
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(hist_cnn.history['loss'],     label='CNN Train', linewidth=2)
ax2.plot(hist_cnn.history['val_loss'], label='CNN Val',   linewidth=2, linestyle='--')
ax2.plot(hist_tl_loss,                 label='TL Train',  linewidth=2)
ax2.plot(hist_tl_val_los,              label='TL Val',    linewidth=2, linestyle='--')
ax2.axvline(EPOCHS_PHASE1, color='gray', linestyle=':')
ax2.set_title("Kurva Loss Training", fontweight='bold')
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(True, alpha=0.3)

# (c) Confusion Matrix CNN
ax3 = fig.add_subplot(gs[1, 0])
cm_cnn = confusion_matrix(true_labels, pred_cnn)
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax3)
ax3.set_title(f"Confusion Matrix — CNN dari Nol\nAkurasi: {acc_cnn*100:.2f}%", fontweight='bold')
ax3.set_xlabel("Prediksi"); ax3.set_ylabel("Aktual")
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax3.get_yticklabels(), rotation=0,  fontsize=8)

# (d) Confusion Matrix Transfer Learning
ax4 = fig.add_subplot(gs[1, 1])
cm_tl = confusion_matrix(true_labels, pred_tl)
sns.heatmap(cm_tl, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=ax4)
ax4.set_title(f"Confusion Matrix — Transfer Learning\nAkurasi: {acc_tl*100:.2f}%", fontweight='bold')
ax4.set_xlabel("Prediksi"); ax4.set_ylabel("Aktual")
plt.setp(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.setp(ax4.get_yticklabels(), rotation=0,  fontsize=8)

# (e) Analisis Error — gambar yang salah prediksi oleh model terbaik
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')

best_pred   = pred_tl if acc_tl >= acc_cnn else pred_cnn
best_name   = "Transfer Learning" if acc_tl >= acc_cnn else "CNN dari Nol"
salah_idx   = np.where(best_pred != true_labels)[0]

if len(salah_idx) > 0:
    n_show = min(8, len(salah_idx))
    sample  = salah_idx[:n_show]
    gs_err  = gridspec.GridSpecFromSubplotSpec(1, n_show, subplot_spec=gs[2, :])

    # Kumpulkan gambar validasi
    val_gen.reset()
    all_imgs = []
    for batch_x, _ in val_gen:
        all_imgs.append(batch_x)
        if len(all_imgs) * BATCH_SIZE >= val_gen.samples:
            break
    all_imgs = np.concatenate(all_imgs)[:val_gen.samples]

    for j, idx in enumerate(sample):
        ax_err = fig.add_subplot(gs_err[j])
        ax_err.imshow(all_imgs[idx])
        ax_err.set_title(
            f"Aktual: {class_names[true_labels[idx]]}\nPrediksi: {class_names[best_pred[idx]]}",
            fontsize=7, color='red'
        )
        ax_err.axis('off')

    ax5.set_title(f"Analisis Error — {best_name} (gambar yang salah prediksi)",
                  fontweight='bold', fontsize=12, pad=20)
else:
    ax5.text(0.5, 0.5, "Semua gambar validasi diprediksi dengan benar!",
             ha='center', va='center', fontsize=14, transform=ax5.transAxes)

plt.savefig("hasil_transfer_learning.png", dpi=150, bbox_inches='tight')
print("  → Tersimpan: hasil_transfer_learning.png")
plt.show()

# ─── Laporan Klasifikasi ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("  LAPORAN KLASIFIKASI — CNN dari Nol")
print("="*60)
print(classification_report(true_labels, pred_cnn, target_names=class_names))

print("="*60)
print("  LAPORAN KLASIFIKASI — Transfer Learning (MobileNetV2)")
print("="*60)
print(classification_report(true_labels, pred_tl,  target_names=class_names))

print("\n" + "="*60)
print(f"  RINGKASAN AKHIR")
print("="*60)
print(f"  CNN dari Nol           : {acc_cnn*100:.2f}%")
print(f"  Transfer Learning (TL) : {acc_tl*100:.2f}%")
pemenang = "Transfer Learning" if acc_tl >= acc_cnn else "CNN dari Nol"
selisih  = abs(acc_tl - acc_cnn) * 100
print(f"  Model terbaik          : {pemenang} (unggul {selisih:.2f}%)")
print("="*60)
print("\nSelesai! Output disimpan:")
print("  - contoh_buah_dataset.png")
print("  - hasil_transfer_learning.png")