# Computer Vision Lab Exam – Practice Workbook (Labs 1–6)

AI-4002 · FAST NUCES Karachi · Built from your Lab 1–6 manuals and task sheets (Lab 2 topics added from the announcement).

## How to use this workbook

The exam is open book, closed internet, with printed Python/OpenCV cheat sheets allowed. It has 3 questions, and Q1 is MCQs. That means:

1. **Q1 (MCQs)** rewards knowing the *concepts and function signatures*. Use the MCQ bank at the end.
2. **Q2 and Q3 (coding)** reward writing a *complete, working pipeline fast*. Every lab task in this workbook has a full model solution. Cover the solution, write your own, then compare.
3. Your printed cheat sheet is the companion doc. Practise **with it open**, so you know exactly where each snippet lives.

For each lab you get: key theory, then practice tasks with solutions, then "what the examiner will probably twist".

### Universal exam template (memorise this)

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('image.jpg')                 # BGR, uint8, shape (H, W, 3)
if img is None:
    print('Error: Image not found.')
else:
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ... processing ...
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.imshow(rgb);  plt.title('Original'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(gray, cmap='gray'); plt.title('Result'); plt.axis('off')
    plt.tight_layout(); plt.show()
```

**Five rules that lose marks if forgotten**

- OpenCV loads **BGR**; Matplotlib shows **RGB**. Convert before `plt.imshow`.
- Single-channel images need `cmap='gray'` in `plt.imshow`.
- Always do the `if image is None` safety check.
- Image indexing is `img[y, x]` (row, column), but OpenCV *function* points are `(x, y)`; `.shape` is `(height, width)` while `cv2.resize`/`warpAffine` sizes are `(width, height)`.
- Use **relative paths** and keep kernel sizes **odd**.

---

# LAB 1: Python, OpenCV and Matplotlib basics

## Theory you must know

- **Digital image** = 2-D function f(x, y). The origin (0,0) is the **top-left**. Rows go down, columns go right. A grayscale pixel is 0 (black) to 255 (white). A colour image has shape (H, W, 3).
- **Pixel, resolution, grayscale, RGB, histogram, feature, edge detection, segmentation, homography, SIFT, SURF, HOG, CNN** are the terminology list (manual §3). Likely MCQ material.
- **Libraries**: OpenCV (general CV, C++/Python), Scikit-Image, Pillow, TorchVision, TensorFlow, Keras, PyTorch, OpenVINO (Intel optimisation), Caffe (CNNs), Detectron2 (object detection, PyTorch), Hugging Face.
- **Fundamental DIP steps**: Image acquisition → enhancement → restoration → colour image processing → wavelets/multiresolution → compression → morphological processing → segmentation → representation and description → object recognition. All of these sit around a knowledge base. Segmentation is called the most difficult step.
- **Purpose of DIP** (5 groups): visualisation, sharpening/restoration, retrieval, measurement of pattern, recognition.

## Practice 1.1: Environment validation

Install and validate OpenCV and Matplotlib.

```python
# Cell 1
%pip install opencv-python
%pip install matplotlib

# Cell 2
import cv2
import matplotlib.pyplot as plt
print('OpenCV version:', cv2.__version__)
print('Matplotlib version:', plt.matplotlib.__version__)
```

## Practice 1.2: Object-oriented Grocery Manager

**Task.** Class `GroceryManager` with `add_item(item, quantity, price)`, `remove_item(item)`, `view_list()`, `calculate_total()` and error handling for removing a missing item.

```python
class GroceryManager:
    def __init__(self):
        self.items = {}                       # {name: {'quantity': q, 'price': p}}

    def add_item(self, item, quantity, price):
        if quantity <= 0 or price < 0:
            raise ValueError('Quantity must be > 0 and price must be >= 0')
        if item in self.items:                # same item again -> update quantity
            self.items[item]['quantity'] += quantity
            self.items[item]['price'] = price
        else:
            self.items[item] = {'quantity': quantity, 'price': price}
        print(f'Added {quantity} x {item} @ {price}')

    def remove_item(self, item):
        if item not in self.items:
            print(f"Error: '{item}' does not exist in the list.")
            return False
        del self.items[item]
        print(f'Removed {item}')
        return True

    def view_list(self):
        if not self.items:
            print('The grocery list is empty.')
            return
        print(f"{'Item':<15}{'Qty':<6}{'Price':<10}{'Subtotal':<10}")
        for name, d in self.items.items():
            print(f"{name:<15}{d['quantity']:<6}{d['price']:<10}{d['quantity']*d['price']:<10}")

    def calculate_total(self):
        return sum(d['quantity'] * d['price'] for d in self.items.values())

g = GroceryManager()
g.add_item('Milk', 2, 250)
g.add_item('Bread', 1, 120)
g.view_list()
g.remove_item('Eggs')          # triggers the error handler
print('Total:', g.calculate_total())
```

## Practice 1.3: Nested-dictionary student record system

**Task.** ID → {'Name', 'Major', 'Grades'}. Write (a) a function that returns the name of the student with the highest average, (b) a search-by-major function that prints matches.

```python
students = {
    101: {'Name': 'Ali',   'Major': 'AI', 'Grades': [85, 90, 78]},
    102: {'Name': 'Sara',  'Major': 'CS', 'Grades': [92, 88, 95]},
    103: {'Name': 'Hamza', 'Major': 'AI', 'Grades': [70, 65, 80]},
}

def top_student(records):
    best_name, best_avg = None, -1
    for sid, info in records.items():
        if not info['Grades']:
            continue                              # avoid division by zero
        avg = sum(info['Grades']) / len(info['Grades'])
        if avg > best_avg:
            best_name, best_avg = info['Name'], avg
    return best_name

def search_by_major(records, major):
    found = False
    for sid, info in records.items():
        if info['Major'].lower() == major.lower():
            print(sid, info['Name'])
            found = True
    if not found:
        print('No students in that major.')

print(top_student(students))          # Sara
search_by_major(students, 'AI')       # Ali, Hamza
```

## Practice 1.4: Safe loading and RGB display (Task 3)

```python
import cv2, matplotlib.pyplot as plt
image_path = 'Sukuna.jpeg'
image = cv2.imread(image_path)
if image is None:
    print('Error: Image not found.')
else:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.title('King of Curses - Ryomen Sukuna', fontsize=16, color='darkred', pad=15)
    plt.imshow(rgb)
    plt.show()
```

## Practice 1.5: Blank canvas, target board and bounding box (Task 4)

Create an 800×800 black image, draw 5 concentric circles of alternating colours, add a rectangle around the outer circle and print the canvas centre.

```python
canvas = np.zeros((800, 800, 3), dtype=np.uint8)       # black, (H, W, 3)
cx, cy = canvas.shape[1] // 2, canvas.shape[0] // 2    # (400, 400)

radii  = [350, 280, 210, 140, 70]                       # draw LARGEST first
colors = [(0,0,255), (255,255,255), (0,0,255), (255,255,255), (0,0,255)]  # BGR
for r, c in zip(radii, colors):
    cv2.circle(canvas, (cx, cy), r, c, -1)             # -1 = filled

R = radii[0]
cv2.rectangle(canvas, (cx - R, cy - R), (cx + R, cy + R), (0, 255, 0), 3)
print('Canvas centre (x, y):', (cx, cy))

plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
```

**Examiner twist:** drawing the small circles first and the big ones later hides them. Always draw largest to smallest when filling.

## Practice 1.6: Heavy Gaussian blur and exact-centre ROI (Task 5)

```python
img = cv2.imread('image.jpg')
if img is None:
    print('Error: Image not found.')
else:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    blur = cv2.GaussianBlur(rgb, (25, 25), 0)

    h, w = rgb.shape[:2]
    y0, x0 = h // 2 - 150, w // 2 - 150                 # 300x300 centred
    roi_orig = rgb[y0:y0 + 300, x0:x0 + 300]
    roi_blur = blur[y0:y0 + 300, x0:x0 + 300]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(roi_orig); plt.title('Original ROI', fontsize=14, color='green'); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(roi_blur); plt.title('Blurred ROI',  fontsize=14, color='red');   plt.axis('off')
    plt.show()
```

Slicing syntax is `image[y_start:y_end, x_start:x_end]`. The image must be at least 300×300, so guard with `if h >= 300 and w >= 300`.

## Practice 1.7: Alpha blend + typography (Task 6)

Semi-transparent blue caption bar covering the bottom 20%, with a title.

```python
img = cv2.imread('image.jpg')
h, w = img.shape[:2]
overlay = img.copy()
y_start = int(h * 0.80)
cv2.rectangle(overlay, (0, y_start), (w, h), (255, 0, 0), -1)      # BGR blue
blended = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)               # src1,a,src2,b,gamma
cv2.putText(blended, 'My Caption', (30, y_start + int(h * 0.12)),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
```

`cv2.putText(img, text, (x, y), font, fontScale, color, thickness)`. The (x, y) is the **bottom-left** corner of the text.

## Practice 1.8: Global vs adaptive threshold + 45° rotation at 0.8 scale (Task 7)

```python
gray = cv2.imread('doc.jpg', cv2.IMREAD_GRAYSCALE)
_, glob = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 11, 2)

h, w = adap.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 45, 0.8)   # centre, angle (CCW +), scale
rot = cv2.warpAffine(adap, M, (w, h))

for i, (im, t) in enumerate([(gray,'Gray'),(glob,'Global'),(adap,'Adaptive'),(rot,'Rotated 45°')], 1):
    plt.subplot(1, 4, i); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
plt.show()
```

## Practice 1.9: Bitwise masking (Task 8)

```python
a = cv2.resize(cv2.imread('a.jpg'), (500, 500))
b = cv2.resize(cv2.imread('b.jpg'), (500, 500))

mask = np.zeros((500, 500), dtype=np.uint8)
cv2.circle(mask, (250, 250), 150, 255, -1)               # white disc

fg  = cv2.bitwise_and(a, a, mask=mask)                    # keep image A inside the circle
inv = cv2.bitwise_not(mask)                               # invert: background becomes white
bg  = cv2.bitwise_and(b, b, mask=inv)                     # keep image B outside the circle
out = cv2.bitwise_or(fg, bg)                              # combine

plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
```

The **mask must be single-channel uint8** and the `mask=` keyword is what limits the AND to the white region.

## Practice 1.10: Pandas image profiling (Task 9)

```python
import pandas as pd
img = cv2.imread('image.jpg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
df = pd.DataFrame({
    'Red':   rgb[:, :, 0].flatten(),
    'Green': rgb[:, :, 1].flatten(),
    'Blue':  rgb[:, :, 2].flatten(),
})
print(df.describe())          # count, mean, std, min, 25%, 50%, 75%, max
```

Alternative from the manual: `img.reshape(-1, 3)` gives (total pixels, 3), and OpenCV order is **B, G, R**. If you flatten the BGR array directly, label the columns `['B','G','R']`.

## Lab 1 extras the manual shows

```python
cv2.imread('x.jpg', cv2.IMREAD_GRAYSCALE)                  # load as gray directly
cv2.resize(img, None, fx=0.5, fy=0.5)                      # scale by factor
cv2.resize(img, (new_w, new_h))                            # explicit size, (width, height)
cv2.GaussianBlur(img, (31, 31), 0)                         # odd kernel, sigma 0 = auto
cropped = img[:, :w // 2]                                  # left half
cv2.add(a, b)                                              # SATURATING add (caps at 255)
cv2.equalizeHist(gray)                                     # histogram equalisation
```

`cv2.add` saturates (250 + 20 = 255), while NumPy `a + b` on uint8 **wraps** (250 + 20 = 14). That is a classic MCQ.

**What the examiner will probably twist:** change the angle, change the kernel size, ask for the *right* half instead of the left, ask for the ROI from the top-left instead of the centre, or ask for text at the top instead of the bottom.

---

# LAB 2: Point (photometric) transformations

The Lab 2 manual is not uploaded, but the announcement lists **Log, Gamma, Piecewise, Histogram** as examinable, and your Lab 2 tasks (chest X-ray, cardiac fusion, echo video) show the expected skills. Everything below is self-contained.

## Theory

A **point transformation** changes each pixel's *value* independently: s = T(r), where r is input intensity and s is output intensity. It does not move pixels (that would be geometric, Lab 3).

| Transform | Formula | Effect | Use when |
| --- | --- | --- | --- |
| Negative | s = 255 − r | Inverts | Bright detail on dark background in a dark image |
| Log | s = c · log(1 + r), c = 255 / log(1 + max) | Expands dark values, compresses bright | Underexposed images, wide dynamic range |
| Inverse log | exponential | Opposite of log | Rare |
| Gamma (power-law) | s = c · r^γ (normalised r in \[0,1\]) | γ < 1 brightens midtones, γ > 1 darkens | Display/monitor correction, midtone control |
| Piecewise linear / contrast stretching | linear segments between (r1,s1) and (r2,s2) | Stretches a chosen range | Low-contrast images |
| Histogram equalisation | CDF mapping | Flattens histogram, global contrast boost | Washed-out or dark images |
| CLAHE | Tile-wise equalisation with clip limit | Local contrast, limits noise amplification | Medical images, uneven contrast |

**Key facts**

- Gamma **< 1** brightens (expands dark tones). Gamma **> 1** darkens.
- Log makes dark regions brighter, so it is the right answer for "reveal faint dark structures".
- Equalisation redistributes intensities using the cumulative distribution function (CDF). It works on **single-channel** images (`equalizeHist` needs an 8-bit single-channel input).
- For colour equalisation, convert to YCrCb or HSV and equalise only the luminance/V channel.

## Practice 2.1: All point transforms on one image

```python
img = cv2.imread('xray.png', cv2.IMREAD_GRAYSCALE)
if img is None:
    print('Error: Image not found.')
else:
    # Negative
    negative = 255 - img

    # Log transform
    c = 255 / np.log(1 + np.max(img))
    log_img = (c * np.log(1 + img.astype(np.float32))).astype(np.uint8)

    # Gamma transform (gamma < 1 brightens)
    gamma = 0.5
    gamma_img = (255 * (img / 255.0) ** gamma).astype(np.uint8)

    # Or via lookup table (faster, common in exams)
    lut = np.array([255 * (i / 255.0) ** gamma for i in range(256)]).astype(np.uint8)
    gamma_img2 = cv2.LUT(img, lut)

    # Histogram equalisation
    eq = cv2.equalizeHist(img)

    titles = ['Original', 'Negative', 'Log', 'Gamma 0.5', 'Equalised']
    imgs   = [img, negative, log_img, gamma_img, eq]
    plt.figure(figsize=(15, 4))
    for i, (im, t) in enumerate(zip(imgs, titles), 1):
        plt.subplot(1, 5, i); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
    plt.show()
```

**Why `astype(np.float32)` before log?** If `img` is uint8 and `np.max(img)` is 255, `1 + img` overflows 255 to 0 for the brightest pixels. Always cast first.

## Practice 2.2: Piecewise linear contrast stretching

Stretch input range \[r1, r2\] to \[0, 255\]; clip outside.

```python
def piecewise(img, r1, s1, r2, s2):
    out = np.zeros_like(img, dtype=np.float32)
    f = img.astype(np.float32)
    # segment 1: 0..r1
    m1 = f <= r1
    out[m1] = f[m1] * (s1 / r1) if r1 > 0 else 0
    # segment 2: r1..r2
    m2 = (f > r1) & (f <= r2)
    out[m2] = s1 + (f[m2] - r1) * ((s2 - s1) / (r2 - r1))
    # segment 3: r2..255
    m3 = f > r2
    out[m3] = s2 + (f[m3] - r2) * ((255 - s2) / (255 - r2))
    return np.clip(out, 0, 255).astype(np.uint8)

stretched = piecewise(img, r1=70, s1=0, r2=180, s2=255)
```

One-line min-max stretch alternative: `cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)`.

## Practice 2.3: Histogram plotting and manual equalisation

```python
hist = cv2.calcHist([img], [0], None, [256], [0, 256])        # image list, channel, mask, bins, range
plt.plot(hist); plt.title('Histogram'); plt.xlim([0, 256]); plt.show()

# Manual equalisation (be ready to explain the steps)
h, _ = np.histogram(img.flatten(), 256, [0, 256])
cdf = h.cumsum()
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf_final = np.ma.filled(cdf_m, 0).astype(np.uint8)
manual_eq = cdf_final[img]                                    # table lookup per pixel

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img)
```

## Practice 2.4: Chest X-ray pipeline (Lab 2 Task 1)

Equalise, JET heatmap, colour balance, threshold dense tissue, log and gamma.

```python
xray = cv2.imread('data/sample_xray.png', cv2.IMREAD_GRAYSCALE)

eq      = cv2.equalizeHist(xray)
heatmap = cv2.applyColorMap(eq, cv2.COLORMAP_JET)                    # returns BGR

# Colour balance: scale each channel so the channel means match (grey-world)
b, g, r = [ch.astype(np.float32) for ch in cv2.split(heatmap)]
avg = (b.mean() + g.mean() + r.mean()) / 3
balanced = cv2.merge([np.clip(b * avg / b.mean(), 0, 255),
                      np.clip(g * avg / g.mean(), 0, 255),
                      np.clip(r * avg / r.mean(), 0, 255)]).astype(np.uint8)

# Keep only densest tissue (bone / dense fluid)
_, dense = cv2.threshold(eq, 200, 255, cv2.THRESH_BINARY)
dense_only = cv2.bitwise_and(xray, xray, mask=dense)

# Log expands dark ribcage edges
c = 255 / np.log(1 + xray.max())
log_x = (c * np.log(1 + xray.astype(np.float32))).astype(np.uint8)

# Gamma < 1 tames bone contrast and lifts lung tissue
gamma_x = (255 * (xray / 255.0) ** 0.6).astype(np.uint8)
```

Why a colour map helps: the human eye separates hues far better than 256 grey shades, so JET turns small intensity differences into clearly different colours and fluid boundaries become visible.

## Practice 2.5: CT + MRI weighted fusion (Lab 2 Task 2)

```python
ct  = cv2.imread('data/ct.png',  cv2.IMREAD_GRAYSCALE)
mri = cv2.imread('data/mri.png', cv2.IMREAD_GRAYSCALE)
mri = cv2.resize(mri, (ct.shape[1], ct.shape[0]))             # sizes MUST match

ct_eq, mri_eq = cv2.equalizeHist(ct), cv2.equalizeHist(mri)
ct_col  = cv2.applyColorMap(ct_eq,  cv2.COLORMAP_BONE)
mri_col = cv2.applyColorMap(mri_eq, cv2.COLORMAP_JET)

fused = cv2.addWeighted(ct_col, 0.7, mri_col, 0.3, 0)         # heavier CT = sharper edges

c = 255 / np.log(1 + fused.max())
fused_log = (c * np.log(1 + fused.astype(np.float32))).astype(np.uint8)
fused_final = (255 * (fused_log / 255.0) ** 0.8).astype(np.uint8)

for i, (im, t) in enumerate([(ct,'CT'),(mri,'MRI'),(fused_final,'Fused')], 1):
    plt.subplot(1, 3, i)
    plt.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap='gray' if im.ndim == 2 else None)
    plt.title(t); plt.axis('off')
plt.show()
```

`addWeighted(src1, alpha, src2, beta, gamma)` computes dst = src1·alpha + src2·beta + gamma. Alpha + beta ≈ 1 keeps brightness stable.

## Practice 2.6: Real-time echo video loop (Lab 2 Task 3)

```python
cap = cv2.VideoCapture('data/echo.mp4')                       # file, not webcam index 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    heat = cv2.applyColorMap(eq, cv2.COLORMAP_JET)
    c = 255 / np.log(1 + heat.max())
    logd = (c * np.log(1 + heat.astype(np.float32))).astype(np.uint8)
    gam  = (255 * (logd / 255.0) ** 1.5).astype(np.uint8)    # gamma > 1 suppresses bright noise

    raw = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    both = np.hstack((raw, gam))                              # same height and channels required
    cv2.imshow('Raw | Enhanced', both)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

The `ret` flag is False at end of file. `np.hstack` needs equal height and equal channel count, so convert gray frames with `COLOR_GRAY2BGR` before stacking.

**What the examiner will probably twist:** swap the transform order (log then gamma vs gamma then log), ask for γ = 2 instead of 0.5 and the *explanation* of the visual change, or ask you to equalise a **colour** image correctly (use YCrCb luminance).

```python
ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
color_eq = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
```

---

# LAB 3: Geometric transformations

## Theory

- **Geometric** transformations change *where* a pixel is: (x, y) → (x', y'). **Photometric** ones change *what* a pixel looks like: I(x, y) → I'(x, y). Memory line: *Geometry changes where the pixel is. Photometry changes what it looks like.*
- **Image coordinates**: origin top-left, x → right, y → down. `image[y, x]` but a point is `(x, y)`. For a 640×480 image the corners are (0,0), (639,0), (0,479), (639,479).
- **2D transforms** use 2×3 (affine, `warpAffine`) or 3×3 homogeneous matrices (`warpPerspective`). **3D transforms** use 4×4.

### Reading a 2×2 matrix \[\[a, b\], \[c, d\]\] by columns

| Entry | Role | Effect |
| --- | --- | --- |
| a | x of new X-axis | Horizontal scale (negative = flip) |
| c | y of new X-axis | Vertical shear |
| b | x of new Y-axis | Horizontal shear |
| d | y of new Y-axis | Vertical scale (negative = flip) |

x' = ax + by and y' = cx + dy. Rotation uses \[\[cosθ, −sinθ\], \[sinθ, cosθ\]\], scaling \[\[sx, 0\], \[0, sy\]\], reflection over the x-axis \[\[1, 0\], \[0, −1\]\], horizontal shear \[\[1, k\], \[0, 1\]\] (x' = x + ky).

### The hierarchy (the manual's tree)

```
2D Geometric Transformations
├── Affine   (p' = Ap + b; keeps collinearity AND parallelism)
│   ├── Linear (no translation; origin stays fixed: A·0 = 0)
│   │   ├── Rotation
│   │   ├── Scale → Similarity → Rigid
│   │   └── Shear
│   └── Translation
└── Projective (homography, 3×3; parallel lines may converge)
    └── Perspective
```

| Type | Preserves | Degrees of freedom | OpenCV |
| --- | --- | --- | --- |
| Rigid (rotation + translation) | Lengths, angles, shape | 3 | `getRotationMatrix2D` with scale 1 + shift |
| Similarity (rotation + translation + uniform scale) | Angles, ratios | 4 | `getRotationMatrix2D` with scale ≠ 1 |
| Affine | Parallelism, collinearity, ratios along a line | 6 (3 point pairs) | `getAffineTransform`, `warpAffine` |
| Projective / homography | Straight lines only | 8 (4 point pairs) | `getPerspectiveTransform`, `findHomography`, `warpPerspective` |

### Homogeneous coordinates

A linear 2×2 matrix cannot translate because it keeps the origin fixed (A·0 = 0). Adding a dummy 1, (x, y) → (x, y, 1), lets translation live inside a 3×3 matrix:

```
[x']   [1 0 tx] [x]
[y'] = [0 1 ty] [y]     → x' = x + tx,  y' = y + ty
[1 ]   [0 0 1 ] [1]
```

Because all transforms are then matrix multiplications, they can be **composed into one master matrix** (one multiplication per pixel). **Order matters**: for "scale then rotate then translate" you write `M = T @ R @ S` (the rightmost is applied first).

**Projective**: \[x', y', w'\] = H \[x, y, 1\], and you must divide by w'. x' = (h11x + h12y + h13) / (h31x + h32y + h33). In affine the bottom row is \[0, 0, 1\], so the denominator is 1. In projective the denominator varies with position, which creates depth/perspective.

### Key OpenCV calls

```python
M = cv2.getRotationMatrix2D(center, angle, scale)     # 2x3; +angle = counter-clockwise; center=(x, y)
out = cv2.warpAffine(img, M, (width, height))          # dsize is (W, H)
M = cv2.getAffineTransform(src3, dst3)                 # float32 arrays of shape (3, 2)
M = cv2.getPerspectiveTransform(src4, dst4)            # float32 arrays of shape (4, 2)
out = cv2.warpPerspective(img, M, (width, height))
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)   # many points, outlier-robust
```

Point arrays **must be `np.float32`**.

## Practice 3.1: Enlarge 300% and keep it centred (Task 1)

Plain scaling grows around the origin (top-left). To keep the centre fixed, scale about the centre: t = c − s·c.

```python
img = cv2.imread('fingerprint.png')
h, w = img.shape[:2]
sx = sy = 3.0
cx, cy = w / 2, h / 2

S = np.array([[sx, 0], [0, sy]], dtype=np.float32)           # manual 2x2 scaling matrix
tx = cx - sx * cx                                             # = -2*cx
ty = cy - sy * cy
M = np.hstack([S, [[tx], [ty]]]).astype(np.float32)           # 2x3 affine matrix

scaled = cv2.warpAffine(img, M, (w, h))                       # same canvas, centre preserved
# To SEE the full enlarged image instead: output size (int(w*sx), int(h*sy)) with tx=ty=0
```

## Practice 3.2: Rotate back 45° without cropping (Task 2)

Expanded canvas size for angle θ: new\_w = h·|sinθ| + w·|cosθ|, new\_h = h·|cosθ| + w·|sinθ|. Then shift the matrix so the centre lands in the middle of the new canvas.

```python
img = cv2.imread('city.jpg')
h, w = img.shape[:2]
angle = -45                                   # image was rotated +45 CCW, so undo with -45
theta = np.deg2rad(angle)
cos, sin = np.cos(theta), np.sin(theta)
cx, cy = w / 2, h / 2

# manual 2x3 rotation matrix about the centre (same as getRotationMatrix2D, scale = 1)
M = np.array([[ cos, sin, (1 - cos) * cx - sin * cy],
              [-sin, cos,  sin * cx + (1 - cos) * cy]], dtype=np.float32)

new_w = int(abs(h * sin) + abs(w * cos))
new_h = int(abs(h * cos) + abs(w * sin))
M[0, 2] += new_w / 2 - cx                     # recentre in the bigger canvas
M[1, 2] += new_h / 2 - cy

rotated = cv2.warpAffine(img, M, (new_w, new_h))
```

If the exam says the image appears rotated the opposite way, flip the sign of the angle.

## Practice 3.3: Remove right-slant from a barcode with a shear (Task 3)

A right-slanted barcode has x shifted in proportion to y. Undo with the opposite shear and add a translation so nothing leaves the frame.

```python
img = cv2.imread('barcode.png')
h, w = img.shape[:2]
k = 0.3                                              # estimate from the slant: k = dx / dy
M = np.float32([[1, -k, k * h],                      # x' = x - k*y + k*h  (keeps x >= 0)
                [0,  1, 0    ]])
new_w = int(w + k * h)
fixed = cv2.warpAffine(img, M, (new_w, h))
```

If the barcode gets *more* slanted, use `+k` and set the translation to 0. Estimate k as (horizontal offset of a bar from top to bottom) / (image height).

## Practice 3.4: Translate the map into view (Task 4)

The X is 150 px left and 80 px up, so shift the picture **right 150 and down 80**.

```python
img = cv2.imread('map.png')
h, w = img.shape[:2]
T = np.float32([[1, 0, 150],
                [0, 1,  80],
                [0, 0,   1]])                         # 3x3 homogeneous translation
shifted = cv2.warpAffine(img, T[:2], (w, h))          # warpAffine wants the top 2 rows
```

## Practice 3.5: Rigid alignment of a chip (Task 5)

Combine rotation and translation into **one** 3×3 matrix.

```python
def R3(deg, cx, cy):                                  # rotation about (cx, cy) as 3x3
    M = cv2.getRotationMatrix2D((cx, cy), deg, 1.0)   # scale = 1 → rigid
    return np.vstack([M, [0, 0, 1]]).astype(np.float32)

def T3(tx, ty):
    return np.float32([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

img = cv2.imread('chip.png'); h, w = img.shape[:2]
rigid = T3(40, -25) @ R3(-90, w / 2, h / 2)           # rotate first, then translate
aligned = cv2.warpAffine(img, rigid[:2], (w, h))
```

## Practice 3.6: Similarity fix of a blueprint (Task 6)

Scale + rotation + translation in one matrix.

```python
img = cv2.imread('blueprint.png'); h, w = img.shape[:2]
s, angle = 2.0, -20                                   # undo the random rotation, enlarge
M_rs = cv2.getRotationMatrix2D((0, 0), angle, s)      # rotate + scale about the origin
sim = np.vstack([M_rs, [0, 0, 1]])
master = T3(30, 30) @ sim                             # then move away from the corner
out = cv2.warpAffine(img, master[:2], (int(w * s), int(h * s)))
```

Similarity keeps **angles**, because scale is uniform. That is the manual's distinguishing property from general affine.

## Practice 3.7: General affine from 3 landmark pairs (Task 7)

6 unknowns need 6 equations, which is 3 points × 2 coordinates.

```python
src = np.float32([[50, 50], [200, 50], [50, 200]])    # landmarks in the glitched image
dst = np.float32([[70, 80], [210, 60], [90, 230]])    # where they SHOULD be after the glitch

# By hand: solve [x y 1] @ [a b tx ; c d ty]^T = [x' y']
A = np.hstack([src, np.ones((3, 1))])                 # 3x3
sol_x = np.linalg.solve(A, dst[:, 0])                 # a, b, tx
sol_y = np.linalg.solve(A, dst[:, 1])                 # c, d, ty
M_glitch = np.vstack([sol_x, sol_y]).astype(np.float32)

# Or: M_glitch = cv2.getAffineTransform(src, dst)

# The glitch mapped good -> bad, so FIX the image with the INVERSE mapping
M_fix = cv2.invertAffineTransform(M_glitch)           # or getAffineTransform(dst, src)
h, w = img.shape[:2]
fixed = cv2.warpAffine(img, M_fix, (w, h))
```

## Practice 3.8: Chalk-art bird's-eye view (Task 8)

```python
img = cv2.imread('chalk.jpg')
src = np.float32([[320, 210], [610, 205], [740, 520], [150, 540]])   # TL, TR, BR, BL of the art (clicked)
S = 500
dst = np.float32([[0, 0], [S, 0], [S, S], [0, S]])                    # perfect square, same order
H = cv2.getPerspectiveTransform(src, dst)
top_down = cv2.warpPerspective(img, H, (S, S))
```

Point **order must match** between `src` and `dst` (TL, TR, BR, BL). A wrong order creates a twisted/mirrored result.

## Practice 3.9: Stadium panorama via homography (Task 9)

```python
img1 = cv2.imread('left.jpg'); img2 = cv2.imread('right.jpg')
# 4 matching points picked by eye in the overlap (x, y)
pts2 = np.float32([[40, 100], [120, 90], [130, 300], [30, 310]])     # in image 2
pts1 = np.float32([[610, 105], [690, 95], [700, 305], [600, 315]])   # same scene points in image 1

H = cv2.getPerspectiveTransform(pts2, pts1)                          # map image 2 INTO image 1's space
h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
pano = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))         # wide canvas
pano[0:h1, 0:w1] = img1                                              # paste reference on top
```

Direction is the trap: `H = getPerspectiveTransform(src=image2_points, dst=image1_points)` projects image 2 **into** image 1's coordinate frame. With many noisy matches use `cv2.findHomography(..., cv2.RANSAC)`.

## Practice 3.10: Master forger, full hierarchy (Task 10)

Linear scale → rigid move → projective warp into an angled frame.

```python
paint = cv2.imread('painting.jpg'); ph, pw = paint.shape[:2]
wall  = cv2.imread('museum.jpg')

S = np.float32([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])                # 1) linear scale (shrink)
Rg = R3(0, 0, 0) @ T3(120, 80)                                       # 2) rigid: rotation 0 + translation
A = Rg @ S                                                           # combined affine (3x3)

corners = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]]).reshape(-1, 1, 2)
pre = cv2.perspectiveTransform(corners, A)                           # where the corners are after scale+move

frame = np.float32([[300, 120], [620, 160], [610, 430], [310, 400]]) # 3) 4 frame corners on the wall
P = cv2.getPerspectiveTransform(pre.reshape(4, 2), frame)            # projective step

final = P @ A                                                         # one master matrix
warped = cv2.warpPerspective(paint, final, (wall.shape[1], wall.shape[0]))

mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), final, (wall.shape[1], wall.shape[0]))
wall_bg = cv2.bitwise_and(wall, wall, mask=cv2.bitwise_not(mask))
result  = cv2.add(wall_bg, cv2.bitwise_and(warped, warped, mask=mask))
```

**What the examiner will probably twist:** sign of angle, asking for a **non-centred** rotation, a different shear direction, translating by (−x, −y), asking for the matrix to be *printed* (print `M`), or combining operations in the **opposite order** to see that matrix multiplication is not commutative.

---

# LAB 4: Feature extraction, filtering and edges

## Theory

**Feature extraction** reduces high-dimensional raw pixels to compact, meaningful descriptors. **Local features** (keypoints, corners, patches) suit matching and detection. **Global features** (colour histograms, texture descriptors, moments) describe the whole image. Technique families: histogram-based, filter-based (Gabor, Haar), deep learning (CNNs).

### Histogram-based descriptors

| Method | What it measures | Steps | Library call |
| --- | --- | --- | --- |
| **HOG** | Distribution of gradient directions | Grayscale → Sobel gradients (magnitude + angle) → split into 8×8 cells → orientation histogram per cell → normalise over 2×2 blocks → concatenate | `skimage.feature.hog(img, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)` |
| **LBP** | Local texture | Compare each pixel with its circular neighbours (≥ centre → 1, else 0) → binary code → histogram of codes | `skimage.feature.local_binary_pattern(img, P, R, method='uniform')` |
| **Colour histogram** | Colour distribution | Bin RGB/HSV and count pixels | `cv2.calcHist([img],[ch],None,[256],[0,256])` |
| **HED** (edge directions) | Dominant edge orientations | Gray → optional blur → Canny/Sobel → angle = arctan2(gy, gx) → 8 bins over 0–360° → histogram | `np.histogram(angle, bins=8, range=(0,360))` |
| **HIG** (intensity gradients) | Gradient orientation distribution | Sobel dx, dy → magnitude sqrt(dx²+dy²), angle atan2(dy, dx) → 9 bins over 0–180° → normalise | `np.histogram(orientation, bins=9, range=(0,180))` |
| **Texture energy / contrast** | Uniformity / variation in a window | Energy = Σ(pixel²) over a window, contrast = std-dev over a window; histogram each | `cv2.filter2D(img.astype(float32)**2, -1, np.ones((3,3)))` |

**LBP uniform**: with P neighbours the uniform method gives values 0 … P+1, so the histogram has **P + 2** bins (`bins=np.arange(0, P+3)`, `range=(0, P+2)`). Standard radius-1 setup is P = 8·R = 8.

**Higher texture energy = more complex, less uniform texture. Higher contrast = bigger intensity variation.**

### Filtering and convolution

A **filter/kernel** is a small matrix slid over the image. **Convolution** multiplies the kernel with the underlying patch element-wise and sums (a dot product), producing one output pixel.

| Convolution type | Behaviour | Output size |
| --- | --- | --- |
| Standard / full | Kernel centred on every pixel | Same as input (OpenCV) |
| Valid | Only where kernel fully overlaps | Smaller: (H−k+1) × (W−k+1) |
| Same | Zero padding keeps size | Same as input |
| Strided | Skips pixels | Roughly (H−k)/s + 1 |

**Exam trap:** `cv2.filter2D` always returns the **same size**, and it has **no `strides` argument**. The manual's `strides=(2, 2)` example does not exist in the real API. To get valid output, crop `k//2` pixels from each side; to get strides, slice `result[::2, ::2]`. `borderType` options: `BORDER_CONSTANT` (zero padding), `BORDER_REFLECT`, `BORDER_REPLICATE`, `BORDER_DEFAULT` (reflect-101).

```python
box   = np.ones((3, 3), np.float32) / 9                       # box blur, sums to 1
gauss = cv2.getGaussianKernel(5, 1.0)                         # 1-D (5x1); outer product gives 2-D
gauss2d = gauss @ gauss.T
sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], np.float32)
emboss  = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
out = cv2.filter2D(gray, -1, kernel)                           # ddepth -1 = same as input
```

### Edge detection techniques

- **Gradient-based**: Sobel, Prewitt, Scharr (3×3 kernels, fast, noise-sensitive). Edges are where the gradient magnitude is high.
- **Canny**: Gaussian smoothing → gradient (Sobel) → non-maximum suppression (thin edges) → hysteresis (T\_high strong, T\_low weak, weak kept only if connected to strong). Best accuracy and noise suppression.
- **LoG / Marr-Hildreth**: Gaussian then Laplacian, edges at zero-crossings of the second derivative.
- **Sobel-Feldman**: variant emphasising diagonals. **CNN-based**: learned edges.

## Practice 4.1: HOG with visualisation

```python
from skimage.feature import hog
from skimage import exposure

gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
features, hog_img = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
hog_resc = exposure.rescale_intensity(hog_img, in_range=(0, 10))

plt.subplot(1, 2, 1); plt.imshow(gray, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(hog_resc, cmap='gray'); plt.title('HOG'); plt.axis('off')
plt.show()
print('HOG vector length:', features.shape)
```

Feature length for a 64×128 window with 9 bins, 8×8 cells, 2×2 blocks is 7×15 blocks × 4 cells × 9 = **3780**. (skimage default orientations = 9.)

## Practice 4.2: LBP histogram

```python
from skimage import feature
R = 1; P = 8 * R
lbp = feature.local_binary_pattern(gray, P, R, method='uniform')
hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
hist = hist.astype('float'); hist /= (hist.sum() + 1e-6)

plt.subplot(1, 2, 1); plt.imshow(lbp, cmap='gray'); plt.title('LBP image'); plt.axis('off')
plt.subplot(1, 2, 2); plt.bar(range(P + 2), hist); plt.title('LBP histogram'); plt.xlabel('Pattern'); plt.ylabel('Frequency')
plt.show()
```

## Practice 4.3: Colour histogram (3 channels)

```python
img = cv2.imread('image.jpg')
for i, col in enumerate(('b', 'g', 'r')):                      # OpenCV channel order is B, G, R
    h = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(h, color=col)
plt.title('Colour histogram'); plt.xlim([0, 256]); plt.show()

# HSV 2-D histogram (Hue 0-179, Sat 0-255)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h2 = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
```

## Practice 4.4: Histogram of Edge Directions (HED)

```python
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
smooth = cv2.GaussianBlur(gray, (5, 5), 0)
edges  = cv2.Canny(smooth, 30, 70)

gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
angle = np.arctan2(gy, gx) * 180 / np.pi                        # -180..180
angle = np.mod(angle, 360)                                      # 0..360 so range(0,360) catches everything

hist, bin_edges = np.histogram(angle[edges > 0], bins=8, range=(0, 360))   # only edge pixels
plt.bar(bin_edges[:-1], hist, width=45, align='center')
plt.xticks(range(0, 360, 45)); plt.xlabel('Edge direction (deg)'); plt.ylabel('Frequency'); plt.show()
```

The manual histograms *all* pixels' angles with `range=(0, 360)`. `arctan2` returns −180…180, so negative angles silently fall outside that range. Adding `np.mod(angle, 360)` fixes it, and selecting `angle[edges > 0]` makes it a true *edge* histogram.

## Practice 4.5: HIG (gradient orientation histogram)

```python
dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(dx ** 2 + dy ** 2)
ori = np.mod(np.arctan2(dy, dx) * 180 / np.pi, 180)             # unsigned orientation 0..180
hist, bins = np.histogram(ori, bins=9, range=(0, 180), weights=mag)   # magnitude-weighted (HOG-like)
hist = hist / hist.sum()
plt.bar(bins[:-1], hist, width=20, align='edge'); plt.show()
```

## Practice 4.6: Texture energy and contrast histograms

```python
f = gray.astype(np.float32)                                     # avoid uint8 overflow when squaring
k = np.ones((3, 3), np.float32)
energy = cv2.filter2D(f ** 2, -1, k)                            # sum of squares in 3x3 window

mean  = cv2.blur(f, (3, 3))
sqmean = cv2.blur(f ** 2, (3, 3))
contrast = np.sqrt(np.maximum(sqmean - mean ** 2, 0))           # local standard deviation

eh, eb = np.histogram(energy, bins=256, range=(0, energy.max()))
ch, cb = np.histogram(contrast, bins=256, range=(0, contrast.max()))
eh = eh / eh.sum(); ch = ch / ch.sum()
```

The manual's `cv2.filter2D(image**2, -1, ...)` on a **uint8** image overflows (e.g. 200² wraps), and `filter2D(image, -1, ones)` computes a *sum*, not the standard deviation the text defines as contrast. The version above is the numerically correct one.

## Practice 4.7: Convolution from scratch (valid, same, stride)

```python
def conv2d(img, kernel, mode='valid', stride=1):
    kh, kw = kernel.shape
    k = np.flipud(np.fliplr(kernel))                            # true convolution flips; filter2D does correlation
    if mode == 'same':
        ph, pw = kh // 2, kw // 2
        img = np.pad(img, ((ph, ph), (pw, pw)), mode='constant')
    H, W = img.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            patch = img[i * stride:i * stride + kh, j * stride:j * stride + kw]
            out[i, j] = np.sum(patch * k)
    return out

result = conv2d(gray.astype(np.float32), sobel_x, mode='same')
```

Output size formula: **floor((N + 2P − K) / S) + 1**.

## Practice 4.8: Edge detector comparison (Sobel, Scharr, Laplacian, Canny)

```python
gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)

sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.uint8(np.clip(np.sqrt(sx ** 2 + sy ** 2), 0, 255))

scx = cv2.Scharr(blur, cv2.CV_64F, 1, 0)
scy = cv2.Scharr(blur, cv2.CV_64F, 0, 1)
scharr_mag = np.uint8(np.clip(np.sqrt(scx ** 2 + scy ** 2), 0, 255))

lap = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))
canny = cv2.Canny(blur, 100, 200)

for i, (im, t) in enumerate([(gray,'Gray'),(sobel_mag,'Sobel'),(scharr_mag,'Scharr'),(lap,'Laplacian'),(canny,'Canny')], 1):
    plt.subplot(1, 5, i); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
plt.show()
```

## Practice 4.9: Written task: texture analysis for material classification

**Two techniques you could use:**

1. **LBP** works by comparing each pixel with a ring of neighbours to form a binary code, then histogramming the codes. It is rotation-tolerant with the uniform variant and robust to lighting changes since it only uses ordering, not absolute values. Suits **fabric** (repetitive micro-patterns) and **wood grain**.
2. **Texture energy and contrast (or HOG/HIG gradient histograms)**. Energy captures how uniform a surface is, contrast how strongly intensities vary, so **polished metal** (low contrast, smooth) separates from **wood** and **fabric** (higher variation). Gradient-orientation histograms capture directional grain, such as wood stripes or brushed metal.

**Advantages.** Both are cheap, need no training data for the features themselves, and give a fixed-length vector that you can feed to an SVM or k-NN. LBP is illumination-robust, and energy/contrast are trivial to compute and interpret.

**Limitations.** LBP at one radius misses large-scale texture. Energy and contrast ignore spatial arrangement and are sensitive to lighting and scale. HOG-style descriptors are not rotation-invariant. Shiny metal creates specular highlights that corrupt every descriptor, so combining features (LBP + contrast + colour histogram) and normalising illumination first gives the best results.

**What the examiner will probably twist:** change cell size to 16×16, change LBP radius/points (R = 2 → P = 16 → 18 bins), ask the histogram bin count for a given setup, or ask which `borderType` produces "valid" vs "same".

---

# LAB 5: Image segmentation

## Theory

**Segmentation** partitions an image into distinct, non-overlapping regions that share colour, texture or intensity. It is the step that turns pixels into "objects".

| Technique | Core idea | Best when | Key call / parameters |
| --- | --- | --- | --- |
| Global threshold | One T for the whole image | Even lighting, clear object/background gap | `cv2.threshold(g, T, 255, cv2.THRESH_BINARY)` |
| Adaptive threshold | T computed per neighbourhood | Uneven illumination (documents) | `cv2.adaptiveThreshold(g, 255, METHOD, THRESH_BINARY, blockSize, C)` |
| Otsu | Picks T that maximises between-class variance | Bimodal histogram, T unknown | `cv2.threshold(g, 0, 255, THRESH_BINARY + THRESH_OTSU)` |
| Colour (HSV) | Range test per channel | Distinct coloured object | `cv2.inRange(hsv, lower, upper)` |
| Canny / edges | Boundaries via gradient + hysteresis | Need outlines not regions | `cv2.Canny(g, low, high)` |
| Region growing | Expand from a seed through similar neighbours | Contiguous uniform region (organ, tumour) | Custom: seed, threshold |
| Watershed | Flood a gradient/distance "topography" from markers | **Touching/overlapping objects** | `cv2.watershed(img, markers)` |
| K-Means | Cluster pixels by colour | Simplify colours, complex scenes | `cv2.kmeans(data, K, None, criteria, attempts, flags)` |
| Deep learning | CNN semantic/instance segmentation | Large data, hard tasks | Not allowed in lab tasks |

**Adaptive threshold parameters**

- `ADAPTIVE_THRESH_MEAN_C`: T = mean of the neighbourhood − C. `ADAPTIVE_THRESH_GAUSSIAN_C`: T = Gaussian-weighted mean − C (smoother, usually better).
- **blockSize** (odd, ≥ 3) is the neighbourhood size. Too small gives broken characters and noise; too large behaves like a global threshold and misses local variation.
- **C** is subtracted from the local mean. Larger C makes it harder for a pixel to become foreground, so you get cleaner background but thinner or lost strokes.

**Thresholding types**: `THRESH_BINARY` (> T → max else 0), `THRESH_BINARY_INV`, `THRESH_TRUNC`, `THRESH_TOZERO`, `THRESH_TOZERO_INV`.

**HSV**: OpenCV Hue is **0–179**, Saturation and Value 0–255. Red wraps around 0, so it needs two ranges (0–10 and 170–179) combined with `cv2.bitwise_or`. Yellow ≈ H 20–35, green ≈ 35–85, blue ≈ 90–130.

**Canny hysteresis**: pixel ≥ high is a strong edge, pixel < low is rejected, pixels between are kept only if connected to a strong edge. A common ratio is high ≈ 2–3 × low.

## Practice 5.1: Global thresholds on an unevenly lit document (Task 1)

```python
gray = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE)
T_values = [80, 127, 180]
globals_ = [cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)[1] for T in T_values]
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 10)

show = [gray] + globals_ + [adaptive]
titles = ['Original', 'Global T=80', 'Global T=127', 'Global T=180', 'Adaptive (Gaussian, 21, 10)']
plt.figure(figsize=(18, 4))
for i, (im, t) in enumerate(zip(show, titles), 1):
    plt.subplot(1, 5, i); plt.imshow(im, cmap='gray'); plt.title(t); plt.axis('off')
plt.show()
cv2.imwrite('task1_output.png', adaptive)                     # "save the final output" is a lab requirement
```

**Parameters and justification (write this in the answer).** T = 80 loses text in the dark side, T = 180 turns the bright side into a black block, and T = 127 sits in between and still fails on one side. Adaptive thresholding with a 21×21 Gaussian window and C = 10 adapts to local brightness.

**Why does a single threshold struggle?** One T assumes the same text/background intensity everywhere. With uneven illumination the *background of the dark side* is darker than the *text of the bright side*, so no single T separates both.

## Practice 5.2: blockSize × C × method grid (Task 2)

```python
configs = []
for method, mname in [(cv2.ADAPTIVE_THRESH_MEAN_C, 'Mean'), (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 'Gaussian')]:
    for bs in [11, 31, 91]:                                   # odd
        for C in [2, 10, 20]:
            out = cv2.adaptiveThreshold(gray, 255, method, cv2.THRESH_BINARY, bs, C)
            configs.append((out, f'{mname} bs={bs} C={C}'))

plt.figure(figsize=(18, 10))
for i, (im, t) in enumerate(configs[:12], 1):
    plt.subplot(3, 4, i); plt.imshow(im, cmap='gray'); plt.title(t, fontsize=9); plt.axis('off')
plt.tight_layout(); plt.show()
```

The task needs at least 3 blockSizes, 3 C values and both methods, so the grid covers 2 × 3 × 3 = 18 results (show a subset of ≥ 6 plus the original).

**Analysis answers**

1. *Neighbourhood too small*: mostly noise, hollow/broken strokes because the window sees only the stroke itself.
2. *Too large*: approaches global behaviour, so illumination gradients return and thin text fades.
3. *C increased*: threshold drops further below the local mean, so fewer pixels qualify as foreground. Noise vanishes but light strokes can disappear.
4. *Cleanest*: typically mid-sized blockSize (≈ 11–31) with moderate C (≈ 5–15). Verify on your image.
5. *Better method*: Gaussian usually produces smoother, less speckled results than Mean, since pixels near the centre count more.

## Practice 5.3: Otsu with histogram (Task 3)

```python
gray = cv2.imread('coins.png', cv2.IMREAD_GRAYSCALE)
T, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('Otsu threshold:', T)

# Repeat after changing contrast
eq = cv2.equalizeHist(gray)
T2, otsu2 = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
T3, otsu3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print('Thresholds (raw / equalised / blurred):', T, T2, T3)

plt.subplot(1, 3, 1); plt.imshow(gray, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(1, 3, 2); plt.hist(gray.ravel(), 256, [0, 256]); plt.axvline(T, color='r'); plt.title('Histogram')
plt.subplot(1, 3, 3); plt.imshow(otsu, cmap='gray'); plt.title(f'Otsu mask (T={T:.0f})'); plt.axis('off')
plt.show()
```

**Why Otsu?** It automatically selects the T that maximises between-class variance, so you need no manual tuning per image. It works best with a **bimodal** histogram (two peaks: object and background). When using Otsu, the `thresh` argument is ignored (pass 0), and you **add** the flags with `+`.

## Practice 5.4: HSV colour sorting, restrictive vs final mask (Task 4)

```python
img = cv2.imread('smarties.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Too restrictive (narrow band) → misses shaded parts of the object
mask_narrow = cv2.inRange(hsv, np.array([28, 200, 200]), np.array([32, 255, 255]))

# Final mask for yellow
mask_final = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # remove speckles

extracted = cv2.bitwise_and(img, img, mask=mask_final)

# Red needs two ranges because hue wraps
red = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 100), (179, 255, 255))
```

**Why the restrictive mask fails.** Real objects have shading, highlights and shadows, so their saturation and value vary widely even though hue is constant. A tight S/V band keeps only the brightest, most saturated pixels and leaves holes or drops most of the object.

## Practice 5.5: Canny with three threshold pairs (Task 5)

```python
gray = cv2.imread('smarties.png', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
pairs = [(30, 90), (80, 160), (150, 250)]
for i, (lo, hi) in enumerate(pairs, 1):
    plt.subplot(1, 4, i + 1); plt.imshow(cv2.Canny(blur, lo, hi), cmap='gray'); plt.title(f'Canny {lo}/{hi}'); plt.axis('off')
plt.subplot(1, 4, 1); plt.imshow(gray, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.show()
```

**Interpretation.** Low thresholds (30/90): all strong edges plus weak texture and noise (**unwanted edges**). Mid thresholds (80/160): clean outlines (**strong edges**), with weak connected edges retained via hysteresis. High thresholds (150/250): only the sharpest boundaries, so faint object edges are **missing**. Raising T\_high while T\_low stays fixed removes isolated weak responses but can also break contours, because fewer strong pixels remain to anchor the weak ones.

## Practice 5.6: Region growing with parameter study (Task 6)

```python
from collections import deque

def region_growing(img, seed, thresh):
    h, w = img.shape
    mask = np.zeros((h, w), np.uint8)
    seed_val = int(img[seed])                                  # seed = (row, col); int() avoids uint8 wrap-around
    q = deque([seed])
    while q:
        r, c = q.popleft()
        if r < 0 or r >= h or c < 0 or c >= w:                 # bounds check
            continue
        if mask[r, c] != 0:                                    # already visited
            continue
        if abs(int(img[r, c]) - seed_val) <= thresh:
            mask[r, c] = 255
            q.extend([(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)])   # 4-connectivity
    return mask

mri = cv2.imread('brain_mri.png', cv2.IMREAD_GRAYSCALE)
seeds = [(120, 130), (60, 200)]
thresholds = [10, 25, 50]
plt.figure(figsize=(12, 7))
for i, s in enumerate(seeds):
    for j, t in enumerate(thresholds):
        plt.subplot(2, 3, i * 3 + j + 1)
        plt.imshow(region_growing(mri, s, t), cmap='gray'); plt.title(f'seed={s}, T={t}'); plt.axis('off')
plt.show()
```

**Why does the seed matter?** Region growing compares neighbours to the **seed's intensity**, so a different seed defines a different "similar" band, and growth starts in a different anatomical area that may be separated by boundaries the criterion cannot cross. Small T under-segments, large T leaks into neighbouring tissue. The manual's version indexes `image[x, y]` with x as the row, so be consistent: **seed is (row, col)**. In the original code a missing `abs`/`int` cast on uint8 data would wrap around (e.g., 5 − 10 = 251).

## Practice 5.7: Marker-based watershed pipeline (Task 7)

```python
img  = cv2.imread('coins.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                           # 1 preprocessing
gray = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)   # 2 threshold (INV if coins are darker)

kernel  = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)           # 3 noise removal
sure_bg = cv2.dilate(opening, kernel, iterations=3)                                # 4 sure background

dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)                              # 5 distance transform
_, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)                         # 6 sure foreground
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)                                           # 7 unknown region

n_labels, markers = cv2.connectedComponents(sure_fg)                               # 8 marker labelling
markers = markers + 1                                                              # background = 1, not 0
markers[unknown == 255] = 0                                                        # unknown = 0 (watershed decides)

markers = cv2.watershed(img, markers)                                              # 9 watershed (needs 3-channel BGR)
out = img.copy()
out[markers == -1] = (0, 0, 255)                                                   # 10 boundaries in red (BGR)

steps = [('Original', cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ('Threshold', thresh), ('Sure BG', sure_bg),
         ('Distance', dist), ('Sure FG', sure_fg), ('Unknown', unknown), ('Watershed', cv2.cvtColor(out, cv2.COLOR_BGR2RGB))]
plt.figure(figsize=(20, 4))
for i, (t, im) in enumerate(steps, 1):
    plt.subplot(1, 7, i); plt.imshow(im, cmap=None if im.ndim == 3 else 'gray'); plt.title(t); plt.axis('off')
plt.show()
```

**Marker rules (exam favourite):** sure background = label 1, each sure-foreground object = labels 2, 3, …, unknown = **0**, and boundaries come back as **−1**. If you forget `markers + 1`, the background is 0 and gets treated as "unknown", which breaks the flooding. `cv2.watershed` modifies `markers` in place and requires an 8-bit **3-channel** image.

## Practice 5.8: Tuning the distance-transform threshold (Task 8)

```python
rows = []
fig = plt.figure(figsize=(16, 4))
for k, frac in enumerate([0.2, 0.4, 0.6, 0.8], 1):
    _, fg = cv2.threshold(dist, frac * dist.max(), 255, 0); fg = np.uint8(fg)
    n, mk = cv2.connectedComponents(fg)
    n_markers = n - 1                                          # label 0 is background of this map
    unk = cv2.subtract(sure_bg, fg)
    mk = mk + 1; mk[unk == 255] = 0
    res = cv2.watershed(img.copy(), mk)
    n_regions = len(np.unique(res)) - 2                        # minus boundary(-1) and background(1)
    vis = img.copy(); vis[res == -1] = (0, 0, 255)
    plt.subplot(1, 4, k); plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); plt.title(f'frac={frac}: {n_regions} regions'); plt.axis('off')
    rows.append((k, frac, n_markers, n_regions))
for r in rows: print(r)
plt.show()
```

| Experiment | Distance threshold (× max) | Foreground markers | Regions | Typical result |
| --- | --- | --- | --- | --- |
| 1 | 0.2 | few (touching coins merge) | too few | **Under-segmentation**: touching coins stay merged |
| 2 | 0.4 | near true count | close to true | Mostly separated |
| 3 | 0.6 | true count | correct | **Best** range for round, similar-size coins |
| 4 | 0.8 | may drop small coins | too few or odd | Small objects lose markers, over-erosion |

(Fill the numbers from your own run.) **Principle:** a low threshold keeps necks between touching objects, so markers merge. A very high threshold shrinks markers so much that small objects vanish. Choose the range that yields marker count = object count.

## Practice 5.9: K-Means colour segmentation (Task 9)

```python
img = cv2.imread('dog.jpeg')
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = np.float32(rgb.reshape((-1, 3)))                      # (H*W, 3), float32 REQUIRED
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1); plt.imshow(rgb); plt.title('Original'); plt.axis('off')
for i, K in enumerate([2, 4, 6], 2):
    ret, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    seg = centers[labels.flatten()].reshape(rgb.shape)         # map every pixel to its cluster centre
    plt.subplot(1, 4, i); plt.imshow(seg); plt.title(f'K={K}'); plt.axis('off')
    print(K, 'clusters, centres:\n', centers)
plt.show()
```

K = 2 gives a coarse foreground/background split. K = 4 captures major regions (sky, fur, ground, shadow). K = 6 preserves more detail with less simplification. More K means closer to the original and less colour reduction. Remember `kmeans` returns **(compactness, labels, centers)**, and the pipeline is *reshape → float32 → cluster → centres to uint8 → index by labels → reshape back*.

## Practice 5.10: Design a segmentation system (Task 10)

Pick one image (e.g., smarties.png) and run three methods: **HSV colour thresholding**, **K-Means**, **Canny**. Then fill a table like this and add the final conclusion:

| Method | Assumption about the image | Succeeds on | Fails on | Most influential parameter |
| --- | --- | --- | --- | --- |
| Otsu / global threshold | Two intensity classes | Dark vs bright objects | Objects with similar grey, different colour | Threshold T |
| HSV colour thresholding | Objects differ in hue | One colour of sweet | Neighbouring similar hues, red wrap-around | Hue range |
| K-Means | Pixel colours form K clusters | Whole-image colour regions | Spatial separation (same colour, two objects) | K |
| Canny | Boundaries = sharp intensity change | Outlines | Fills no regions, gaps in contours | low/high thresholds |
| Watershed | Objects convex, markers recoverable | Touching round objects | Noise creates over-segmentation | Distance threshold |

**Conclusion template:** "For this image, \<method> produced the most meaningful regions because \<image property, e.g., the objects differ mainly in hue while intensity overlaps>. Evidence: \<count of correctly isolated objects / visual comparison>." Base the conclusion on what you **observed**, as the sheet requires.

---

# LAB 6: Wavelets, boundary detection, Hough transform and SIFT

## Theory

### Wavelet transform

Wavelets analyse a signal in **time-frequency** (or space-frequency) at multiple scales, so they suit **non-stationary** signals and edges, unlike the Fourier transform which loses time localisation.

| Type | Idea | Families | Uses |
| --- | --- | --- | --- |
| **CWT** | Wavelet continuously dilated (scale) and translated | Morlet (oscillatory) | Seismic and time-frequency analysis |
| **DWT** | Scales/shifts on a dyadic grid; splits signal into approximation (low-pass) and detail (high-pass) | Daubechies (db), Haar, Symlet | Compression, denoising, features |
| **WPT** | Also decomposes the *detail* branches (full tree) | Same | Complex frequency content, ML features |
| **Inverse** | Reconstructs the signal from coefficients | Same | Reconstruction after thresholding |

2-D DWT yields four sub-bands: **LL** (approximation), **LH/HL** (horizontal/vertical detail) and **HH** (diagonal detail). **Denoising recipe:** decompose → threshold (soft/hard) the detail coefficients → reconstruct.

```python
# pip install PyWavelets
import pywt
cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar')                       # 2-D DWT
recon = pywt.idwt2((cA, (cH, cV, cD)), 'haar')                   # inverse
coeffs = pywt.wavedec(signal, 'db4', level=4)                    # multi-level 1-D
rebuilt = pywt.waverec(coeffs, 'db4')
```

### Boundary (edge) detection

- **Sobel**: two 3×3 kernels, Gx (detects *vertical* edges, "is right brighter than left?") and Gy (detects *horizontal* edges). Magnitude G = √(Gx² + Gy²). Use `cv2.CV_64F` so negative gradients are kept, then `np.uint8(np.clip(mag, 0, 255))` for display. Uniform area → zero response.
- **Canny**: Gaussian smoothing → gradient magnitude/direction → non-maximum suppression → hysteresis thresholding. Thin, continuous edges.
- **LoG**: Gaussian blur then Laplacian (second derivative). The **peak** of the first derivative marks an edge, while the **zero-crossing** of the second derivative locates it. Use `cv2.convertScaleAbs` to display.

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
sx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)               # note: dx=0, dy=1  (Sobel Y is dx=0, dy=1: always check this argument order)
mag = np.uint8(np.clip(np.sqrt(sx**2 + sy**2), 0, 255))
canny = cv2.Canny(blurred, 50, 150)
log   = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F))
```

### Hough transform

Detects shapes by **voting in parameter space**: each edge pixel votes for all shapes that could pass through it; peaks = shapes.

- **Lines**: manual explains with y = mx + b (m, b axes), but OpenCV uses the polar form **ρ = x·cosθ + y·sinθ** because vertical lines have infinite slope. A line in the image = a point in (ρ, θ) space, and a point in the image = a sinusoid in parameter space.
- **Circles**: (x − a)² + (y − b)² = r², so the parameter space is **3-D (a, b, r)**. `HOUGH_GRADIENT` uses gradient direction to avoid a full 3-D vote.
- Input must be an **edge map** (Canny) for lines.

```python
lines  = cv2.HoughLines(edges, 1, np.pi / 180, 150)                           # returns (rho, theta)
linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=10)   # segments (x1,y1,x2,y2)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=40, minRadius=10, maxRadius=80)  # (x, y, r)
```

`param1` = upper Canny threshold inside HoughCircles. `param2` = accumulator threshold (lower → more, possibly false, circles). `dp` = inverse accumulator resolution. `minDist` = minimum distance between centres.

### SIFT (Scale-Invariant Feature Transform)

1. **Scale-space extrema detection**: Difference of Gaussians (DoG = L(x, y, kσ) − L(x, y, σ)) over many scales.
2. **Keypoint localisation**: a candidate must be a local max/min versus **26 neighbours** (8 in its own layer + 9 above + 9 below). A 3-D quadratic fit refines it; low-contrast and edge-like points are discarded.
3. **Orientation assignment**: dominant gradient direction → rotation invariance.
4. **Descriptor**: 4×4 sub-regions × 8 orientation bins = **128-D vector**, robust to illumination.
5. **Matching**: Euclidean (L2) distance, usually with Lowe's ratio test (best < 0.75 × second best).
6. **Outlier rejection**: **RANSAC**.
7. **Homography estimation** for stitching/recognition.

Scale and rotation invariant; partially invariant to illumination and viewpoint. Descriptors have shape `(N, 128)` and dtype float32, so use `NORM_L2` (not Hamming, which is for binary descriptors like ORB).

```python
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)
vis = cv2.drawKeypoints(rgb, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(len(kp), des.shape)
```

## Practice 6.1: Anomaly detection in sensor data with wavelets

```python
import pywt
np.random.seed(42)
n = 1000
data = np.random.normal(0, 1, n)
data[[150, 420, 700]] += [6, -7, 8]                                # injected faults

coeffs = pywt.wavedec(data, 'db4', level=4)
sigma = np.median(np.abs(coeffs[-1])) / 0.6745                    # noise estimate from finest detail
uthresh = sigma * np.sqrt(2 * np.log(n))                          # universal threshold
coeffs[1:] = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs[1:]]
denoised = pywt.waverec(coeffs, 'db4')[:n]

residual = data - denoised
limit = 3 * residual.std()                                        # 3-sigma rule
anom_idx = np.where(np.abs(residual) > limit)[0]

plt.figure(figsize=(12, 7))
plt.subplot(2, 1, 1)
plt.plot(data, label='Sensor Data'); plt.plot(denoised, '--', label='Denoised Signal')
plt.title('Sensor Data and Denoised Signal'); plt.legend()
plt.subplot(2, 1, 2)
plt.plot(residual, 'r', label='Residuals'); plt.scatter(anom_idx, residual[anom_idx], c='g', label='Anomalies', zorder=3)
plt.title('Residuals and Detected Anomalies'); plt.legend()
plt.tight_layout(); plt.show()
```

The sheet's expected plot has the same layout: top panel (sensor data + denoised dashed), bottom panel (red residuals + green anomaly dots). Lower the multiplier (2σ) for more anomalies, or raise it (3σ) for fewer.

## Practice 6.2: Computer screen detection with Hough lines

```python
img = cv2.imread('lab.jpg'); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, minLineLength=80, maxLineGap=10)

line_mask = np.zeros_like(gray)
for x1, y1, x2, y2 in (lines[:, 0] if lines is not None else []):
    ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if ang < 10 or abs(ang - 90) < 10 or ang > 170:                # keep near-horizontal / near-vertical
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

line_mask = cv2.dilate(line_mask, np.ones((5, 5), np.uint8))
cnts, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
screens = 0
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 80 and h > 60 and 1.0 < w / h < 2.2:                    # monitor-like aspect ratio
        state = 'ON' if gray[y:y + h, x:x + w].mean() > 90 else 'OFF'   # brightness decides status
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, state, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        screens += 1
print('Detected screens:', screens, '| expected rows x cols to compare for missing screens')
```

Compare the detected count against the expected count per row to flag **missing screens**.

## Practice 6.3: Asset tracking / object recognition with SIFT (+ video)

```python
ref = cv2.imread('keyboard_ref.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(ref, None)
bf = cv2.BFMatcher(cv2.NORM_L2)

cap = cv2.VideoCapture('lab.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(g, None)
    if des2 is not None and len(kp2) >= 2:
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in (p for p in matches if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) >= 10:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is not None:
                h, w = ref.shape
                box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                cv2.polylines(frame, [np.int32(cv2.perspectiveTransform(box, H))], True, (0, 255, 0), 3)
                cv2.putText(frame, f'Keyboard ({len(good)} matches)', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('SIFT recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
```

For the still-image version, remove the loop and call it once per test image. Use **`kp1` from the reference once** outside the loop for speed.

## Practice 6.4: Panorama stitching with SIFT

```python
img1 = cv2.imread('left.jpg'); img2 = cv2.imread('right.jpg')
g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
k1, d1 = sift.detectAndCompute(g1, None); k2, d2 = sift.detectAndCompute(g2, None)

matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2, d1, k=2)           # query = image 2, train = image 1
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)   # points in image 2
dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)   # points in image 1
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]
pano = cv2.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
pano[0:h1, 0:w1] = img1
plt.imshow(cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
```

Shortcut (mention it if allowed): `cv2.Stitcher_create().stitch([img1, img2])`.

## Practice 6.5: Lane detection with Hough lines

```python
img = cv2.imread('road.jpg'); h, w = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

roi = np.zeros_like(edges)
poly = np.array([[(0, h), (w // 2 - 50, int(h * 0.6)), (w // 2 + 50, int(h * 0.6)), (w, h)]], np.int32)
cv2.fillPoly(roi, poly, 255)                                     # keep only the road trapezoid
masked = cv2.bitwise_and(edges, roi)

lines = cv2.HoughLinesP(masked, 1, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
out = img.copy()
for x1, y1, x2, y2 in (lines[:, 0] if lines is not None else []):
    slope = (y2 - y1) / (x2 - x1 + 1e-6)
    if abs(slope) > 0.4:                                         # drop near-horizontal clutter
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 4)
plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
```

The **ROI mask** is what turns generic Hough into lane detection. Negative slope = left lane, positive = right lane (y grows downward).

## Practice 6.6: Coin detection and counting (Hough circles)

```python
img = cv2.imread('coins.jpg'); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)                                   # reduce false circles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                           param1=100, param2=40, minRadius=15, maxRadius=80)
count = 0
if circles is not None:
    for x, y, r in np.uint16(np.around(circles[0])):
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)               # outline
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)               # centre
        count += 1
cv2.putText(img, f'Coins: {count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off'); plt.show()
print('Number of coins:', count)
```

If it finds too many circles, **raise `param2`**. Too few, lower it. Adjust `minRadius`/`maxRadius` to the coin sizes.

## Practice 6.7: Smart security zone with boundary detection (video)

```python
cap = cv2.VideoCapture('security.mp4')              # or 0 for a webcam
bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50)
zone = (200, 150, 300, 250)                          # x, y, w, h of the restricted area

while True:
    ret, frame = cap.read()
    if not ret:
        break
    fg = bg.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)       # drop shadow pixels (value 127)
    edges = cv2.Canny(fg, 50, 150)                                # object boundaries
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    zx, zy, zw, zh = zone
    cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (255, 255, 0), 2)
    alarm = False
    for c in cnts:
        if cv2.contourArea(c) < 800:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if x < zx + zw and x + w > zx and y < zy + zh and y + h > zy:   # rectangle overlap test
            alarm = True
    if alarm:
        cv2.putText(frame, 'ALARM: UNAUTHORIZED OBJECT', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    cv2.imshow('Security', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
cap.release(); cv2.destroyAllWindows()
```

**What the examiner will probably twist:** swap Hough lines for circles, ask what changes if the threshold parameter (Hough votes) is lowered, ask for the *number of bins/neighbours/descriptor length* in SIFT, or ask to draw bounding boxes instead of polylines.

---

# MCQ PRACTICE BANK (Question 1 style)

Try all 56 first, then check the key below.

## Lab 1

1. OpenCV's `cv2.imread` returns colour channels in which order? (a) RGB (b) BGR (c) GBR (d) HSV
2. What does `cv2.imread('missing.jpg')` return if the file does not exist? (a) raises FileNotFoundError (b) an empty list (c) `None` (d) a black image
3. Which slice keeps the **right half** of an image with width `w`? (a) `img[:, :w//2]` (b) `img[:, w//2:]` (c) `img[w//2:, :]` (d) `img[:w//2, :]`
4. `cv2.add(np.uint8(250), np.uint8(20))` gives: (a) 14 (b) 270 (c) 255 (d) 0
5. A Gaussian kernel size must be: (a) even (b) odd (c) a power of 2 (d) 1
6. A 640×480 (W×H) colour image has `.shape` of: (a) (640, 480, 3) (b) (480, 640, 3) (c) (3, 480, 640) (d) (480, 640)
7. In `cv2.putText`, the (x, y) point is the text's: (a) top-left (b) centre (c) bottom-left (d) top-right
8. `cv2.resize(img, (300, 200))` outputs an image of: (a) 300 rows, 200 columns (b) 200 rows, 300 columns (c) 300×300 (d) unchanged
9. The origin (0,0) of a digital image is the: (a) centre (b) bottom-left (c) top-left (d) top-right
10. `getRotationMatrix2D(center, 45, 1)` rotates: (a) 45° clockwise (b) 45° counter-clockwise (c) 45 radians (d) only if scale > 1

## Lab 2

11. A gamma value **< 1** makes the image: (a) darker (b) brighter in midtones (c) negative (d) binary
12. Which transform best reveals detail in very dark regions? (a) negative (b) log (c) gamma > 1 (d) threshold
13. `cv2.equalizeHist` requires: (a) float64 colour (b) 8-bit single-channel (c) HSV (d) binary image
14. The image negative of an 8-bit image is: (a) r − 255 (b) 255 − r (c) 1/r (d) log(r)
15. `cv2.applyColorMap(gray, cv2.COLORMAP_JET)` returns: (a) 1-channel (b) 3-channel BGR (c) 4-channel (d) float
16. `cv2.addWeighted(a, 0.7, b, 0.3, 0)` computes: (a) 0.7a × 0.3b (b) 0.7a + 0.3b + 0 (c) a + b (d) max(a, b)
17. CLAHE's `clipLimit` mainly: (a) limits noise/contrast over-amplification (b) sets tile count (c) blurs (d) sets bit depth

## Lab 3

18. A pure 2×2 linear transform cannot translate because: (a) A·0 = 0 (b) det = 0 (c) it is not invertible (d) it is 2D
19. Which property does an affine transform **always** preserve? (a) angles (b) lengths (c) parallelism (d) perspective
20. Degrees of freedom of a homography: (a) 4 (b) 6 (c) 8 (d) 9
21. `cv2.getAffineTransform` needs: (a) 2 point pairs (b) 3 (c) 4 (d) 6
22. `cv2.warpAffine` expects a matrix of shape: (a) 2×2 (b) 2×3 (c) 3×3 (d) 4×4
23. The bottom row of an affine 3×3 matrix is: (a) \[1,1,1\] (b) \[0,0,1\] (c) \[h31,h32,h33\] (d) \[0,0,0\]
24. For "scale, then rotate, then translate" you compose: (a) S @ R @ T (b) T @ R @ S (c) R @ S @ T (d) order does not matter
25. The matrix \[\[1, k\], \[0, 1\]\] gives: (a) x' = x + ky (b) y' = y + kx (c) x' = kx (d) rotation
26. A similarity transform preserves: (a) angles (b) parallel lines only (c) nothing (d) areas
27. The `dsize` argument of `warpPerspective` is: (a) (height, width) (b) (width, height) (c) (rows, cols) (d) scale factors

## Lab 4

28. HOG is built from: (a) colour counts (b) gradient orientation histograms in cells (c) pixel intensities directly (d) Fourier magnitudes
29. LBP 'uniform' with P = 8 gives how many histogram bins? (a) 8 (b) 9 (c) 10 (d) 256
30. For a 3×3 kernel, `cv2.filter2D` output size is: (a) smaller by 2 (b) same as the input (c) larger by 2 (d) halved
31. Why use `cv2.CV_64F` in Sobel? (a) speed (b) to keep negative gradient values (c) to get colour (d) to save memory
32. Correct Canny stage order: (a) gradient → blur → NMS → hysteresis (b) blur → gradient → NMS → hysteresis (c) NMS → blur → gradient → hysteresis (d) hysteresis → gradient → blur → NMS
33. In hysteresis, a pixel between T\_low and T\_high is kept if: (a) always (b) never (c) it is connected to a strong edge (d) it is the brightest
34. Valid convolution, N = 7, K = 3, stride 1, no padding gives output width: (a) 7 (b) 6 (c) 5 (d) 3
35. High texture **energy** indicates: (a) smooth, uniform texture (b) complex, less uniform texture (c) no texture (d) colour

## Lab 5

36. Otsu's method works best when the histogram is: (a) flat (b) unimodal (c) bimodal (d) random
37. For a document with uneven lighting, use: (a) a global threshold (b) adaptive threshold (c) negative transform (d) Otsu only
38. OpenCV Hue range is: (a) 0–359 (b) 0–255 (c) 0–179 (d) 0–100
39. In watershed markers, the **unknown** region is labelled: (a) 0 (b) 1 (c) −1 (d) 255
40. `cv2.kmeans` data must be: (a) uint8 (b) float32 (c) int64 (d) boolean
41. Region growing result depends strongly on the: (a) seed point (b) image file format (c) colormap (d) figure size
42. Best classic method to split **touching** coins: (a) global threshold (b) Canny (c) marker-based watershed (d) negative
43. Increasing C in `adaptiveThreshold` generally: (a) lowers the threshold so more pixels are white (b) raises the bar for foreground so fewer pixels qualify (c) changes the colour space (d) has no effect
44. `cv2.kmeans` returns: (a) labels only (b) (compactness, labels, centers) (c) centers, labels (d) image
45. For `THRESH_OTSU`, the threshold argument you pass should be: (a) the expected T (b) 0 (c) 255 (d) 127 and it is respected

## Lab 6

46. SIFT descriptor length: (a) 64 (b) 128 (c) 256 (d) 512
47. A SIFT keypoint candidate is compared with how many neighbours in scale space? (a) 8 (b) 18 (c) 26 (d) 27
48. RANSAC in SIFT matching is used to: (a) compute DoG (b) reject outlier matches (c) blur (d) assign orientation
49. The Hough circle accumulator has how many dimensions? (a) 1 (b) 2 (c) 3 (d) 4
50. `cv2.HoughLines` returns each line as: (a) (x1, y1, x2, y2) (b) (m, b) (c) (rho, theta) (d) (a, b, r)
51. In DWT, the **approximation** coefficients come from: (a) high-pass filter (b) low-pass filter (c) Laplacian (d) median
52. The Morlet wavelet is associated with: (a) DWT only (b) CWT (c) Haar family (d) WPT only
53. LoG locates edges at: (a) peaks of first derivative (b) zero-crossings of second derivative (c) histogram peaks (d) corners
54. The Sobel **Gx** kernel mainly detects: (a) horizontal edges (b) vertical edges (c) diagonals only (d) colour
55. Which matcher norm is correct for SIFT descriptors? (a) NORM\_HAMMING (b) NORM\_L2 (c) NORM\_L0 (d) none
56. Which tool suits non-stationary signals best? (a) Fourier transform only (b) Wavelet transform (c) histogram (d) median filter

## Answer key with one-line reasons

1. **b**: OpenCV is BGR; Matplotlib is RGB.
2. **c**: `imread` fails silently and returns None, which is why the safety check exists.
3. **b**: `[:, w//2:]` keeps all rows, columns from the midpoint to the end.
4. **c**: saturating arithmetic caps at 255 (NumPy `+` would wrap to 14).
5. **b**: odd so the kernel has a centre.
6. **b**: shape is (height, width, channels).
7. **c**: bottom-left corner of the text.
8. **b**: `dsize` is (width, height), so 200 rows × 300 columns.
9. **c**: top-left; y points down.
10. **b**: positive angle is counter-clockwise.
11. **b**: γ < 1 expands dark/mid tones, brightening.
12. **b**: log expands dark values.
13. **b**: 8-bit single channel.
14. **b**: s = 255 − r.
15. **b**: 3-channel BGR.
16. **b**: weighted sum plus gamma scalar.
17. **a**: clip limit stops noise over-amplification in tiles.
18. **a**: A·0 = 0, so the origin is fixed; you need homogeneous coordinates.
19. **c**: parallelism and collinearity (not angles or lengths).
20. **c**: 8 (3×3 up to scale), so 4 point pairs.
21. **b**: 3 pairs → 6 unknowns.
22. **b**: 2×3.
23. **b**: \[0, 0, 1\], which is why the denominator is 1.
24. **b**: rightmost matrix applies first.
25. **a**: horizontal shear x' = x + ky.
26. **a**: uniform scale keeps angles.
27. **b**: (width, height).
28. **b**: gradient orientation histograms per cell with block normalisation.
29. **c**: P + 2 = 10.
30. **b**: same size; borders handled by `borderType`.
31. **b**: negative gradients would be clipped in uint8.
32. **b**: smoothing, gradient, non-maximum suppression, hysteresis.
33. **c**: weak pixels survive only when connected to strong ones.
34. **c**: 7 − 3 + 1 = 5.
35. **b**: higher energy means more complex, less uniform.
36. **c**: two classes give two peaks.
37. **b**: adaptive.
38. **c**: 0–179 (it is stored in 8 bits, so degrees are halved).
39. **a**: unknown = 0, sure background = 1, objects 2+, boundaries −1.
40. **b**: float32.
41. **a**: seed defines the reference intensity and the starting area.
42. **c**: marker-based watershed with a distance transform.
43. **b**: T = local mean − C, so a larger C means fewer white pixels.
44. **b**: `ret, labels, centers`.
45. **b**: pass 0, and it is ignored and computed automatically.
46. **b**: 4×4 × 8 = 128.
47. **c**: 8 + 9 + 9 = 26.
48. **b**: robust model fitting that rejects outliers.
49. **c**: (a, b, r).
50. **c**: polar form; `HoughLinesP` returns the endpoints form in (a).
51. **b**: low-pass.
52. **b**: CWT (Morlet).
53. **b**: zero-crossings; the first derivative gives peaks.
54. **b**: Gx responds to horizontal intensity change, i.e. vertical edges.
55. **b**: SIFT descriptors are float vectors, so L2 distance.
56. **b**: wavelets localise in time and frequency.

---

# MARK-MAXIMISING STRATEGY FOR Q2 AND Q3

## Answer in this order every time

1. **Imports and load with safety check.** Easy marks that examiners look for.
2. **Core algorithm**, using the exact functions named in the question. If the question says "use `cv2.getRotationMatrix2D`", do not do it by hand.
3. **Display with subplots**: titles, `axis('off')`, `cmap='gray'` where needed.
4. **Print the key values** (matrix, threshold, count, centre coordinates).
5. **Two lines of explanation** in a comment or markdown cell, covering *what the method assumes* and *why you picked the parameters*.

Partial credit comes from a working pipeline with sensible parameters plus a written justification, so do not abandon a task because one step fails. Comment what you intended.

## Debugging checklist (60 seconds)

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Blue/orange colours swapped | BGR shown as RGB | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` |
| Grayscale shows purple/green | Missing colormap | `plt.imshow(x, cmap='gray')` |
| `AttributeError: 'NoneType' ... shape` | Wrong path | Check path, working directory |
| `cv2.error` in `addWeighted`/`bitwise_*` | Size or channel mismatch | `cv2.resize`, `cvtColor(GRAY2BGR)` |
| Mask has no effect | Mask not uint8 single-channel or `mask=` missing | `mask.astype(np.uint8)`, use `mask=mask` |
| Black corners after rotation | Same-size canvas | Compute new dimensions and shift matrix |
| Image all black/white after log or gamma | uint8 overflow, or forgot to scale back | Cast to float32, then `.astype(np.uint8)` |
| `cv2.kmeans` error | Not float32 or not 2-D | `np.float32(img.reshape(-1, 3))` |
| Watershed ignores objects | Background label 0 | Add 1 to markers, set unknown = 0 |
| Hough returns `None` | Lower the vote threshold; check Canny output | Always test `if lines is not None` |
| Warped image is empty | Point arrays not float32 or wrong order | `np.float32`, order TL, TR, BR, BL |
| Video window freezes | No `waitKey` | `cv2.waitKey(25)`; release the capture at the end |

## Function-signature drill (fill from memory, then check your cheat sheet)

```python
cv2.threshold(src, thresh, maxval, type)           -> (ret, dst)
cv2.adaptiveThreshold(src, maxval, method, type, blockSize, C)
cv2.Canny(img, threshold1, threshold2)
cv2.GaussianBlur(src, ksize, sigmaX)
cv2.warpAffine(src, M, (w, h))
cv2.warpPerspective(src, M, (w, h))
cv2.addWeighted(src1, alpha, src2, beta, gamma)
cv2.inRange(hsv, lower, upper)
cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)
cv2.HoughCircles(gray, method, dp, minDist, param1, param2, minRadius, maxRadius)
cv2.kmeans(data, K, bestLabels, criteria, attempts, flags)
cv2.circle(img, center, radius, color, thickness)
cv2.rectangle(img, pt1, pt2, color, thickness)
cv2.line(img, pt1, pt2, color, thickness)
cv2.putText(img, text, org, fontFace, fontScale, color, thickness)
```

## Final-week plan

| Day | Focus | Do this |
| --- | --- | --- |
| 1 | Labs 1 + 2 | Rewrite 1.4–1.10 and 2.1–2.6 from a blank notebook, cheat sheet closed, then open |
| 2 | Lab 3 | Redo 3.1–3.10. Print every matrix and check it by hand for one point |
| 3 | Lab 4 | HOG, LBP, convolution from scratch, edge comparison, written texture answer |
| 4 | Lab 5 | Tasks 5.1–5.5, then 5.7–5.9 (watershed and K-Means are the big ones) |
| 5 | Lab 6 | SIFT matching, HoughLinesP lane detection, HoughCircles, wavelet denoise |
| 6 | Mixed | MCQ bank under 15 minutes, then two random tasks timed at 25 minutes each |
| 7 | Polish | Finalise the printed cheat sheet, re-read all "exam trap" lines, sleep |

When the detailed syllabus breakdown is posted, tick off every bullet against the sections above and add anything missing to your cheat sheet.
