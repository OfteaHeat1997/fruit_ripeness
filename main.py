import os, random
import cv2
import matplotlib.pyplot as plt

# 1) Buscar automáticamente la carpeta 'test' en todo 'data'
def find_test_dir(root="data"):
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath).lower() == "test":
            # comprobamos que dentro hay subcarpetas (clases)
            subdirs = [d for d in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, d))]
            if subdirs:
                return dirpath
    raise FileNotFoundError("No encontré ninguna carpeta llamada 'test' con subcarpetas de clases dentro de 'data/'.")

test_dir = find_test_dir("data")
print("TEST DIR:", test_dir)

# 2) Listar clases y conteo
classes = [c for c in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, c))]
if not classes:
    raise RuntimeError("La carpeta 'test' no contiene subcarpetas de clases.")

print("\nClases encontradas:")
for c in classes:
    cdir = os.path.join(test_dir, c)
    n = len([f for f in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, f))])
    print(f"- {c}: {n} imágenes")

# 3) Mostrar 1 imagen aleatoria por clase (robusto)
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()

for i, c in enumerate(classes[:9]):  # hasta 9 clases
    cdir = os.path.join(test_dir, c)
    files = [f for f in os.listdir(cdir) if os.path.isfile(os.path.join(cdir, f))]
    if not files:
        axes[i].set_title(f"{c} (vacío)")
        axes[i].axis("off")
        continue
    img_path = os.path.join(cdir, random.choice(files))
    img = cv2.imread(img_path)
    if img is None:
        axes[i].set_title(f"{c} (no legible)")
        axes[i].axis("off")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].set_title(c)
    axes[i].axis("off")

plt.tight_layout()
plt.show()


