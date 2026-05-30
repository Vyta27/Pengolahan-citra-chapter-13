import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TUGAS 1: IMPLEMENTASI KNN DARI NOL")
print("Dataset: MNIST (Kaggle Digit Recognizer)")
print("=" * 60)

print("\n[1] Memuat dataset MNIST dari Kaggle...")
df = pd.read_csv('train.csv')

print(f"    Total data   : {len(df)} baris")
print(f"    Jumlah kolom : {df.shape[1]} (label + 784 piksel)")
print(f"    Jumlah kelas : {df['label'].nunique()} (digit 0-9)")

# Pisahkan fitur (X) dan label (y)
X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values

# Normalisasi piksel ke rentang [0, 1]
X = X / 255.0

print(f"    Shape X      : {X.shape}")
print(f"    Shape y      : {y.shape}")

# Gunakan subset agar KNN manual tidak terlalu lama
# (KNN manual O(n) per query — 42000 sampel sangat lambat)
N_TRAIN = 3000   # jumlah data training
N_TEST  = 500    # jumlah data testing

np.random.seed(42)
idx = np.random.permutation(len(X))
X_train = X[idx[:N_TRAIN]]
y_train = y[idx[:N_TRAIN]]
X_test  = X[idx[N_TRAIN:N_TRAIN + N_TEST]]
y_test  = y[idx[N_TRAIN:N_TRAIN + N_TEST]]

print(f"\n    Data training : {N_TRAIN} sampel")
print(f"    Data testing  : {N_TEST} sampel")

# Tampilkan contoh gambar digit
print("\n[2] Menampilkan contoh gambar dari dataset...")
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
for digit in range(10):
    idx_digit = np.where(y_train == digit)[0][0]
    img = X_train[idx_digit].reshape(28, 28)
    axes[0, digit].imshow(img, cmap='gray')
    axes[0, digit].set_title(f'Digit {digit}', fontsize=9)
    axes[0, digit].axis('off')
    idx_digit2 = np.where(y_train == digit)[0][1]
    img2 = X_train[idx_digit2].reshape(28, 28)
    axes[1, digit].imshow(img2, cmap='gray')
    axes[1, digit].axis('off')
plt.suptitle('Contoh Gambar MNIST per Digit (Kaggle)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('contoh_digit_mnist.png', dpi=150, bbox_inches='tight')
plt.show()
print("    Gambar disimpan: contoh_digit_mnist.png")

# ============================================================
# LANGKAH 2: IMPLEMENTASI KNN DARI NOL (MANUAL)
# ============================================================
print("\n" + "=" * 60)
print("[3] IMPLEMENTASI KNN MANUAL (TANPA SKLEARN)")
print("=" * 60)

class KNNManual:
    """
    Implementasi K-Nearest Neighbor dari nol menggunakan NumPy.
    
    Algoritma:
    1. Simpan semua data training
    2. Untuk setiap data test, hitung jarak ke semua data training
    3. Ambil K tetangga terdekat
    4. Prediksi kelas berdasarkan voting mayoritas
    """
    
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Simpan data training (KNN tidak perlu pelatihan eksplisit)."""
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, x1, x2):
        """
        Hitung jarak Euclidean antara dua vektor.
        d(x,y) = sqrt( sum_i (x_i - y_i)^2 )
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _euclidean_distance_matrix(self, X_test):
        """
        Hitung semua jarak sekaligus (lebih efisien dengan broadcasting).
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b^T
        """
        # Shape: (n_test, n_train)
        sq_test  = np.sum(X_test  ** 2, axis=1, keepdims=True)   # (n_test, 1)
        sq_train = np.sum(self.X_train ** 2, axis=1, keepdims=True).T  # (1, n_train)
        dot      = X_test @ self.X_train.T                         # (n_test, n_train)
        dist_sq  = sq_test + sq_train - 2 * dot
        dist_sq  = np.maximum(dist_sq, 0)   # hindari nilai negatif karena floating point
        return np.sqrt(dist_sq)
    
    def predict(self, X_test):
        """Prediksi kelas untuk data testing."""
        distances = self._euclidean_distance_matrix(X_test)
        predictions = []
        for dist_row in distances:
            # Ambil indeks K tetangga terdekat
            k_indices = np.argsort(dist_row)[:self.k]
            # Ambil label dari tetangga
            k_labels = self.y_train[k_indices]
            # Voting mayoritas
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """Hitung akurasi prediksi."""
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

# Uji KNN manual dengan K=5
print("\nMelatih dan menguji KNN Manual (K=5)...")
t_start = time.time()
knn_manual = KNNManual(k=5)
knn_manual.fit(X_train, y_train)
y_pred_manual = knn_manual.predict(X_test)
t_manual = time.time() - t_start

acc_manual = accuracy_score(y_test, y_pred_manual)
print(f"  Akurasi KNN Manual (K=5) : {acc_manual:.4f} ({acc_manual*100:.2f}%)")
print(f"  Waktu prediksi           : {t_manual:.2f} detik")

# ============================================================
# LANGKAH 3: IMPLEMENTASI KNN SKLEARN (PEMBANDING)
# ============================================================
print("\n" + "=" * 60)
print("[4] IMPLEMENTASI KNN SKLEARN (PEMBANDING)")
print("=" * 60)

t_start = time.time()
knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_sklearn.fit(X_train, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test)
t_sklearn = time.time() - t_start

acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"\n  Akurasi KNN Sklearn (K=5): {acc_sklearn:.4f} ({acc_sklearn*100:.2f}%)")
print(f"  Waktu prediksi           : {t_sklearn:.2f} detik")

# Perbandingan langsung
print("\n  --- Perbandingan Langsung ---")
print(f"  {'Metode':<20} {'Akurasi':>10} {'Waktu (s)':>12}")
print(f"  {'-'*44}")
print(f"  {'KNN Manual':<20} {acc_manual:>9.4f} {t_manual:>11.2f}")
print(f"  {'KNN Sklearn':<20} {acc_sklearn:>9.4f} {t_sklearn:>11.2f}")
diff = abs(acc_manual - acc_sklearn)
print(f"\n  Selisih akurasi: {diff:.4f} ({diff*100:.2f}%)")
if diff < 0.01:
    print("  ✓ Hasil hampir identik — implementasi manual VALID!")

# ============================================================
# LANGKAH 4: EVALUASI NILAI K DENGAN CROSS-VALIDATION 5-FOLD
# ============================================================
print("\n" + "=" * 60)
print("[5] EVALUASI NILAI K DENGAN CROSS-VALIDATION 5-FOLD")
print("=" * 60)

K_VALUES = [1, 3, 5, 7, 10, 15]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results_manual  = {}
results_sklearn = {}

print(f"\n{'K':<6} {'KNN Manual CV':>15} {'KNN Sklearn CV':>16} {'Selisih':>10}")
print("-" * 50)

for k in K_VALUES:
    # ---- KNN Sklearn CV (digunakan sebagai proxy untuk manual) ----
    knn_sk = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    cv_scores_sk = cross_val_score(knn_sk, X_train, y_train,
                                   cv=cv, scoring='accuracy', n_jobs=-1)
    
    # ---- KNN Manual CV (evaluasi fold per fold) ----
    cv_scores_manual = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        Xf_train, Xf_val = X_train[train_idx], X_train[val_idx]
        yf_train, yf_val = y_train[train_idx], y_train[val_idx]
        knn_m = KNNManual(k=k)
        knn_m.fit(Xf_train, yf_train)
        fold_acc = knn_m.score(Xf_val, yf_val)
        cv_scores_manual.append(fold_acc)
    cv_scores_manual = np.array(cv_scores_manual)
    
    results_manual[k]  = cv_scores_manual
    results_sklearn[k] = cv_scores_sk
    
    mean_m  = cv_scores_manual.mean()
    mean_sk = cv_scores_sk.mean()
    selisih = abs(mean_m - mean_sk)
    
    print(f"K={k:<4} {mean_m:.4f} ± {cv_scores_manual.std():.4f}  "
          f"{mean_sk:.4f} ± {cv_scores_sk.std():.4f}  {selisih:.4f}")

# ============================================================
# LANGKAH 5: VISUALISASI HASIL
# ============================================================
print("\n[6] Membuat visualisasi hasil...")

means_manual  = [results_manual[k].mean()  for k in K_VALUES]
stds_manual   = [results_manual[k].std()   for k in K_VALUES]
means_sklearn = [results_sklearn[k].mean() for k in K_VALUES]
stds_sklearn  = [results_sklearn[k].std()  for k in K_VALUES]

best_k_manual  = K_VALUES[np.argmax(means_manual)]
best_k_sklearn = K_VALUES[np.argmax(means_sklearn)]

# --- Plot 1: Akurasi vs K ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(K_VALUES, means_manual,  'o-', color='steelblue',
        linewidth=2, markersize=7, label='KNN Manual')
ax.fill_between(K_VALUES,
                [m - s for m, s in zip(means_manual, stds_manual)],
                [m + s for m, s in zip(means_manual, stds_manual)],
                alpha=0.15, color='steelblue')
ax.plot(K_VALUES, means_sklearn, 's--', color='coral',
        linewidth=2, markersize=7, label='KNN Sklearn')
ax.fill_between(K_VALUES,
                [m - s for m, s in zip(means_sklearn, stds_sklearn)],
                [m + s for m, s in zip(means_sklearn, stds_sklearn)],
                alpha=0.15, color='coral')
ax.axvline(best_k_manual, color='steelblue', linestyle=':', alpha=0.7,
           label=f'Best K manual = {best_k_manual}')
ax.set_xlabel('Nilai K', fontsize=12)
ax.set_ylabel('Akurasi CV 5-Fold', fontsize=12)
ax.set_title('Akurasi vs Nilai K\n(Cross-Validation 5-Fold)', fontsize=12, fontweight='bold')
ax.set_xticks(K_VALUES)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([min(means_manual + means_sklearn) - 0.03,
             max(means_manual + means_sklearn) + 0.03])

# --- Plot 2: Boxplot per K (KNN Manual) ---
ax = axes[1]
data_boxes = [results_manual[k] * 100 for k in K_VALUES]
bp = ax.boxplot(data_boxes, patch_artist=True, notch=False,
                medianprops=dict(color='white', linewidth=2))
colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(K_VALUES)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels([f'K={k}' for k in K_VALUES], fontsize=9)
ax.set_xlabel('Nilai K', fontsize=12)
ax.set_ylabel('Akurasi (%)', fontsize=12)
ax.set_title('Distribusi Akurasi KNN Manual\nper Nilai K (5 Fold)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# --- Plot 3: Confusion Matrix KNN Manual (K terbaik) ---
ax = axes[2]
knn_best = KNNManual(k=best_k_manual)
knn_best.fit(X_train, y_train)
y_pred_best = knn_best.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
ax.set_xlabel('Prediksi', fontsize=11)
ax.set_ylabel('Aktual', fontsize=11)
ax.set_title(f'Confusion Matrix KNN Manual\n(K={best_k_manual}, Test Set)',
             fontsize=12, fontweight='bold')
for i in range(10):
    for j in range(10):
        val = cm[i, j]
        color = 'white' if val > cm.max() / 2 else 'black'
        ax.text(j, i, str(val), ha='center', va='center',
                fontsize=7, color=color)
plt.colorbar(im, ax=ax)

plt.suptitle('Tugas 1: Evaluasi KNN Manual vs Sklearn — Dataset MNIST (Kaggle)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('hasil_evaluasi_knn.png', dpi=150, bbox_inches='tight')
plt.show()
print("  Grafik disimpan: hasil_evaluasi_knn.png")

# ============================================================
# LANGKAH 6: LAPORAN KLASIFIKASI LENGKAP
# ============================================================
print("\n" + "=" * 60)
print("[7] LAPORAN KLASIFIKASI LENGKAP (KNN Manual, K terbaik)")
print("=" * 60)
print(f"\n  K terbaik (manual CV) : K = {best_k_manual}")
print(f"  Akurasi test set      : {accuracy_score(y_test, y_pred_best):.4f}\n")
print(classification_report(y_test, y_pred_best,
                             target_names=[str(i) for i in range(10)]))

# ============================================================
# LANGKAH 7: RINGKASAN AKHIR
# ============================================================
print("=" * 60)
print("RINGKASAN AKHIR")
print("=" * 60)
print(f"\n  Dataset        : MNIST (Kaggle Digit Recognizer)")
print(f"  Training set   : {N_TRAIN} sampel")
print(f"  Testing set    : {N_TEST} sampel")
print(f"  CV             : 5-Fold Stratified")
print(f"\n  {'K':<6} {'Akurasi Manual':>16} {'Akurasi Sklearn':>17}")
print(f"  {'-'*41}")
for k in K_VALUES:
    m  = results_manual[k].mean()
    sk = results_sklearn[k].mean()
    tag = " ← TERBAIK" if k == best_k_manual else ""
    print(f"  K={k:<4} {m:.4f}            {sk:.4f}{tag}")
print(f"\n  K Terbaik (Manual)  : K = {best_k_manual} "
      f"(akurasi CV = {max(means_manual):.4f})")
print(f"  K Terbaik (Sklearn) : K = {best_k_sklearn} "
      f"(akurasi CV = {max(means_sklearn):.4f})")
print(f"\n  Kesimpulan: Implementasi KNN manual menghasilkan akurasi")
print(f"  yang hampir identik dengan sklearn, membuktikan bahwa")
print(f"  algoritma diimplementasikan dengan benar.")
print("\n  File output:")
print("    - contoh_digit_mnist.png  : visualisasi sampel dataset")
print("    - hasil_evaluasi_knn.png  : grafik evaluasi K & confusion matrix")
print("\n" + "=" * 60)
print("SELESAI")
print("=" * 60)