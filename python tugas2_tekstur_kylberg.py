# ============================================================
# TUGAS 2: Perbandingan Fitur untuk Klasifikasi Tekstur
# Mata Kuliah : Pengolahan Citra Digital
# Dataset     : Kylberg Texture Dataset (Kaggle)
# Metode      : LBP, HOG, GLCM x SVM, Random Forest
# ============================================================
#
# PERSIAPAN DATASET:
# 1. Buka: https://www.kaggle.com/datasets/jmexpert1/kylberg-texture-dataset-v10
# 2. Download dan extract
# 3. Struktur folder yang diharapkan:
#    kylberg/
#      ├── blanket1/   (berisi file .png)
#      ├── canvas1/
#      ├── cushion1/
#      └── ... (dst)
# 4. Ubah variabel DATASET_PATH di bawah sesuai lokasi folder

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from skimage.feature import local_binary_pattern, hog
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import exposure

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import Pipeline
import time

# ============================================================
# KONFIGURASI — SESUAIKAN PATH INI
# ============================================================
DATASET_PATH = 'archive'   # folder dataset Kylberg
IMG_SIZE     = (128, 128)  # resize semua gambar ke ukuran ini
MAX_PER_CLASS = 40         # jumlah gambar per kelas (hemat RAM)
N_CLASSES     = 10         # ambil 10 kelas pertama saja

print("=" * 65)
print("TUGAS 2: PERBANDINGAN FITUR TEKSTUR")
print("Dataset : Kylberg Texture Dataset (Kaggle)")
print("Metode  : LBP, HOG, GLCM × SVM, Random Forest")
print("=" * 65)

# ============================================================
# LANGKAH 1: MUAT DATASET
# ============================================================
print("\n[1] Memuat dataset dari folder:", DATASET_PATH)

def load_dataset(root_path, img_size, max_per_class, n_classes):
    """Muat gambar dari folder, resize, konversi grayscale."""
    root = Path(root_path)
    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()])[:n_classes]
    
    images, labels = [], []
    class_names = []
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        class_names.append(class_name)
        img_files = list(class_dir.glob('*.png')) + \
                    list(class_dir.glob('*.jpg')) + \
                    list(class_dir.glob('*.tiff'))
        img_files = img_files[:max_per_class]
        
        for img_path in img_files:
            try:
                img = Image.open(img_path).convert('L')  # grayscale
                img = img.resize(img_size)
                images.append(np.array(img))
                labels.append(class_name)
            except Exception:
                continue
        print(f"  {class_name:<20}: {len(img_files)} gambar dimuat")
    
    return np.array(images), np.array(labels), class_names

X_imgs, y_labels, class_names = load_dataset(
    DATASET_PATH, IMG_SIZE, MAX_PER_CLASS, N_CLASSES)

le = LabelEncoder()
y = le.fit_transform(y_labels)

print(f"\n  Total gambar   : {len(X_imgs)}")
print(f"  Jumlah kelas   : {len(class_names)}")
print(f"  Ukuran gambar  : {IMG_SIZE}")
print(f"  Kelas          : {class_names}")

# Tampilkan contoh gambar per kelas
print("\n[2] Menampilkan contoh gambar per kelas...")
n_show = min(len(class_names), 10)
fig, axes = plt.subplots(1, n_show, figsize=(16, 3))
for i, cls in enumerate(class_names[:n_show]):
    idx = np.where(y_labels == cls)[0][0]
    axes[i].imshow(X_imgs[idx], cmap='gray')
    axes[i].set_title(cls, fontsize=8, rotation=15, ha='right')
    axes[i].axis('off')
plt.suptitle('Contoh Gambar Tekstur — Kylberg Dataset (Kaggle)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('contoh_tekstur_kylberg.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Disimpan: contoh_tekstur_kylberg.png")

# ============================================================
# LANGKAH 2: FUNGSI EKSTRAKSI FITUR
# ============================================================
print("\n[3] Mendefinisikan fungsi ekstraksi fitur...")

# ---------- LBP (Local Binary Pattern) ----------
def extract_lbp(images, radius=3, n_points=24, n_bins=64):
    """
    LBP: mengkodekan pola intensitas piksel terhadap tetangganya.
    - radius    : jarak tetangga
    - n_points  : jumlah titik sampling
    - n_bins    : jumlah bin histogram
    Output: histogram LBP (distribusi pola lokal)
    """
    features = []
    for img in images:
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins,
                               range=(0, n_points + 2), density=True)
        features.append(hist)
    return np.array(features)

# ---------- HOG (Histogram of Oriented Gradients) ----------
def extract_hog(images):
    """
    HOG: distribusi arah gradien dalam sel-sel lokal.
    Sangat efektif menangkap struktur tekstur berbasis tepi.
    Output: vektor fitur HOG
    """
    features = []
    for img in images:
        feat = hog(img,
                   orientations=9,
                   pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2),
                   block_norm='L2-Hys',
                   transform_sqrt=True)
        features.append(feat)
    return np.array(features)

# ---------- GLCM (Gray-Level Co-occurrence Matrix) ----------
def extract_glcm(images, distances=[1, 3], angles=[0, np.pi/4, np.pi/2]):
    """
    GLCM: matriks statistik frekuensi kemunculan pasangan piksel.
    Properti yang diekstrak: contrast, dissimilarity, homogeneity,
    energy, correlation, ASM — untuk setiap kombinasi jarak & sudut.
    """
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity',
             'energy', 'correlation', 'ASM']
    for img in images:
        img_uint8 = (img / img.max() * 255).astype(np.uint8) \
                    if img.max() > 0 else img.astype(np.uint8)
        # Kuantisasi ke 64 level untuk efisiensi
        img_q = (img_uint8 // 4).astype(np.uint8)
        glcm = graycomatrix(img_q, distances=distances, angles=angles,
                            levels=64, symmetric=True, normed=True)
        feat = []
        for prop in props:
            vals = graycoprops(glcm, prop).ravel()
            feat.extend(vals)
        features.append(feat)
    return np.array(features)

# ============================================================
# LANGKAH 3: EKSTRAKSI SEMUA FITUR
# ============================================================
print("\n[4] Mengekstrak fitur dari semua gambar...")

t0 = time.time()
print("  Mengekstrak LBP...", end=' ', flush=True)
X_lbp = extract_lbp(X_imgs)
print(f"selesai — shape: {X_lbp.shape} ({time.time()-t0:.1f}s)")

t0 = time.time()
print("  Mengekstrak HOG...", end=' ', flush=True)
X_hog = extract_hog(X_imgs)
print(f"selesai — shape: {X_hog.shape} ({time.time()-t0:.1f}s)")

t0 = time.time()
print("  Mengekstrak GLCM...", end=' ', flush=True)
X_glcm = extract_glcm(X_imgs)
print(f"selesai — shape: {X_glcm.shape} ({time.time()-t0:.1f}s)")

feature_sets = {
    'LBP' : X_lbp,
    'HOG' : X_hog,
    'GLCM': X_glcm,
}

# ============================================================
# LANGKAH 4: VISUALISASI FITUR
# ============================================================
print("\n[5] Memvisualisasikan fitur yang diekstrak...")

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
sample_idx = [np.where(y == c)[0][0] for c in range(min(4, len(class_names)))]

for col, idx in enumerate(sample_idx):
    img = X_imgs[idx]
    cls = class_names[y[idx]]

    # Baris 0: Gambar asli
    axes[0, col].imshow(img, cmap='gray')
    axes[0, col].set_title(f'{cls}', fontsize=9, fontweight='bold')
    axes[0, col].axis('off')

    # Baris 1: Visualisasi LBP
    lbp_img = local_binary_pattern(img, 24, 3, method='uniform')
    axes[1, col].imshow(lbp_img, cmap='gray')
    axes[1, col].set_title('LBP map', fontsize=8)
    axes[1, col].axis('off')

    # Baris 2: Visualisasi HOG
    _, hog_img = hog(img, orientations=9, pixels_per_cell=(16, 16),
                     cells_per_block=(2, 2), block_norm='L2-Hys',
                     visualize=True, transform_sqrt=True)
    hog_img_eq = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    axes[2, col].imshow(hog_img_eq, cmap='magma')
    axes[2, col].set_title('HOG map', fontsize=8)
    axes[2, col].axis('off')

axes[0, 0].set_ylabel('Gambar Asli', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('LBP', fontsize=10, fontweight='bold')
axes[2, 0].set_ylabel('HOG', fontsize=10, fontweight='bold')

plt.suptitle('Visualisasi Ekstraksi Fitur Tekstur per Kelas',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('visualisasi_fitur_tekstur.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Disimpan: visualisasi_fitur_tekstur.png")

# ============================================================
# LANGKAH 5: TRAINING & EVALUASI SEMUA KOMBINASI
# ============================================================
print("\n[6] Melatih dan mengevaluasi semua kombinasi (6 kombinasi)...")
print(f"    {'Kombinasi':<25} {'Akurasi Test':>13} {'F1 Score':>10} {'CV Mean':>10} {'Waktu(s)':>10}")
print("    " + "-" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    'SVM'          : SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
}

all_results = {}

for feat_name, X_feat in feature_sets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y)

    for clf_name, clf in classifiers.items():
        combo = f"{feat_name} + {clf_name}"

        # Pipeline: scaler + classifier
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf.__class__(**clf.get_params()))
        ])

        t0 = time.time()
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        elapsed = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(pipe, X_feat, y, cv=cv,
                                    scoring='accuracy', n_jobs=-1)

        all_results[combo] = {
            'accuracy'  : acc,
            'f1_score'  : f1,
            'cv_mean'   : cv_scores.mean(),
            'cv_std'    : cv_scores.std(),
            'train_time': elapsed,
            'y_pred'    : y_pred,
            'y_test'    : y_test,
            'pipeline'  : pipe,
            'feature'   : feat_name,
            'classifier': clf_name,
        }
        print(f"    {combo:<25} {acc:>12.4f} {f1:>10.4f} "
              f"{cv_scores.mean():>10.4f} {elapsed:>10.2f}")

# ============================================================
# LANGKAH 6: VISUALISASI PERBANDINGAN HASIL
# ============================================================
print("\n[7] Membuat visualisasi perbandingan hasil...")

combos    = list(all_results.keys())
accs      = [all_results[c]['accuracy'] for c in combos]
f1s       = [all_results[c]['f1_score'] for c in combos]
cv_means  = [all_results[c]['cv_mean']  for c in combos]
cv_stds   = [all_results[c]['cv_std']   for c in combos]

feat_colors = {'LBP': '#4C72B0', 'HOG': '#DD8452', 'GLCM': '#55A868'}
bar_colors  = [feat_colors[all_results[c]['feature']] for c in combos]

fig = plt.figure(figsize=(18, 12))
gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

# --- Plot 1: Akurasi Test Set ---
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(range(len(combos)), accs, color=bar_colors, alpha=0.85,
               edgecolor='white', linewidth=1.2)
ax1.set_xticks(range(len(combos)))
ax1.set_xticklabels([c.replace(' + ', '\n') for c in combos], fontsize=8)
ax1.set_ylabel('Akurasi', fontsize=11)
ax1.set_title('Akurasi Test Set\nper Kombinasi Fitur + Classifier', fontsize=11, fontweight='bold')
ax1.set_ylim([max(0, min(accs) - 0.1), 1.02])
for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
patches = [mpatches.Patch(color=c, label=f) for f, c in feat_colors.items()]
ax1.legend(handles=patches, fontsize=8, title='Fitur')
ax1.grid(True, alpha=0.3, axis='y')

# --- Plot 2: CV Mean ± Std ---
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(len(combos))
ax2.bar(x, cv_means, color=bar_colors, alpha=0.85,
        edgecolor='white', linewidth=1.2)
ax2.errorbar(x, cv_means, yerr=cv_stds, fmt='none',
             color='black', capsize=5, linewidth=1.5)
ax2.set_xticks(x)
ax2.set_xticklabels([c.replace(' + ', '\n') for c in combos], fontsize=8)
ax2.set_ylabel('Akurasi CV', fontsize=11)
ax2.set_title('Akurasi Cross-Validation 5-Fold\n(Mean ± Std)', fontsize=11, fontweight='bold')
ax2.set_ylim([max(0, min(cv_means) - 0.1), 1.02])
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(handles=patches, fontsize=8, title='Fitur')

# --- Plot 3: F1 Score ---
ax3 = fig.add_subplot(gs[0, 2])
ax3.barh(range(len(combos)), f1s, color=bar_colors, alpha=0.85,
         edgecolor='white', linewidth=1.2)
ax3.set_yticks(range(len(combos)))
ax3.set_yticklabels([c.replace(' + ', '\n') for c in combos], fontsize=8)
ax3.set_xlabel('F1 Score (Weighted)', fontsize=11)
ax3.set_title('F1 Score per Kombinasi', fontsize=11, fontweight='bold')
ax3.set_xlim([max(0, min(f1s) - 0.1), 1.02])
for i, f1 in enumerate(f1s):
    ax3.text(f1 + 0.003, i, f'{f1:.3f}', va='center', fontsize=8, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='x')

# --- Plot 4 & 5: Confusion Matrix kombinasi terbaik & terburuk ---
best_combo  = max(all_results, key=lambda c: all_results[c]['accuracy'])
worst_combo = min(all_results, key=lambda c: all_results[c]['accuracy'])

for ax_idx, (combo, title_prefix) in enumerate(
        [(best_combo, 'Terbaik'), (worst_combo, 'Terburuk')]):
    ax = fig.add_subplot(gs[1, ax_idx])
    cm = confusion_matrix(all_results[combo]['y_test'],
                          all_results[combo]['y_pred'])
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel('Prediksi', fontsize=10)
    ax.set_ylabel('Aktual', fontsize=10)
    ax.set_title(f'Confusion Matrix — {title_prefix}\n{combo}\n'
                 f'(Akurasi: {all_results[combo]["accuracy"]:.3f})',
                 fontsize=9, fontweight='bold')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            color = 'white' if val > cm.max() / 2 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=7, color=color)
    plt.colorbar(im, ax=ax)

# --- Plot 6: Heatmap ringkasan ---
ax6 = fig.add_subplot(gs[1, 2])
feat_list = ['LBP', 'HOG', 'GLCM']
clf_list  = ['SVM', 'Random Forest']
heat_data = np.array([[all_results[f'{f} + {c}']['accuracy']
                        for c in clf_list] for f in feat_list])
im6 = ax6.imshow(heat_data, cmap='YlOrRd', vmin=heat_data.min()-0.05,
                 vmax=heat_data.max()+0.01)
ax6.set_xticks([0, 1]); ax6.set_yticks([0, 1, 2])
ax6.set_xticklabels(clf_list, fontsize=10)
ax6.set_yticklabels(feat_list, fontsize=10)
ax6.set_title('Heatmap Akurasi\nFitur × Classifier', fontsize=11, fontweight='bold')
for i in range(3):
    for j in range(2):
        ax6.text(j, i, f'{heat_data[i,j]:.3f}',
                 ha='center', va='center', fontsize=12, fontweight='bold',
                 color='white' if heat_data[i,j] > heat_data.mean() else 'black')
plt.colorbar(im6, ax=ax6)

plt.suptitle('Tugas 2: Perbandingan Fitur Tekstur (LBP, HOG, GLCM)\n'
             'dengan Classifier SVM dan Random Forest — Kylberg Dataset',
             fontsize=13, fontweight='bold', y=1.01)
plt.savefig('hasil_perbandingan_fitur.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Disimpan: hasil_perbandingan_fitur.png")

# ============================================================
# LANGKAH 7: LAPORAN LENGKAP
# ============================================================
print("\n" + "=" * 65)
print("LAPORAN KLASIFIKASI LENGKAP — KOMBINASI TERBAIK")
print("=" * 65)
print(f"\nKombinasi terbaik: {best_combo}")
print(f"Akurasi test set : {all_results[best_combo]['accuracy']:.4f}\n")
print(classification_report(
    all_results[best_combo]['y_test'],
    all_results[best_combo]['y_pred'],
    target_names=class_names))

# ============================================================
# LANGKAH 8: RINGKASAN AKHIR
# ============================================================
print("=" * 65)
print("RINGKASAN AKHIR — SEMUA KOMBINASI")
print("=" * 65)
print(f"\n  {'Kombinasi':<25} {'Akurasi':>9} {'F1':>7} {'CV Mean':>9} {'CV Std':>8}")
print("  " + "-" * 60)
for combo in sorted(all_results, key=lambda c: all_results[c]['accuracy'], reverse=True):
    r = all_results[combo]
    tag = " ← TERBAIK" if combo == best_combo else ""
    print(f"  {combo:<25} {r['accuracy']:>8.4f} {r['f1_score']:>7.4f} "
          f"{r['cv_mean']:>9.4f} {r['cv_std']:>8.4f}{tag}")

print(f"""
  Kesimpulan Analisis:
  - Metode fitur terbaik : {all_results[best_combo]['feature']}
  - Classifier terbaik  : {all_results[best_combo]['classifier']}
  - Kombinasi terbaik   : {best_combo}
    (Akurasi: {all_results[best_combo]['accuracy']:.4f})
  - Kombinasi terburuk  : {worst_combo}
    (Akurasi: {all_results[worst_combo]['accuracy']:.4f})

  File output:
    - contoh_tekstur_kylberg.png   : contoh gambar dataset
    - visualisasi_fitur_tekstur.png: visualisasi LBP & HOG
    - hasil_perbandingan_fitur.png : grafik perbandingan lengkap
""")
print("=" * 65)
print("SELESAI")
print("=" * 65)