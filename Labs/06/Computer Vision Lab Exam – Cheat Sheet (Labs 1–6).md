# Python + OpenCV Lab Exam Cheat Sheet (Labs 1–6)

AI-4002 · printable reference. Every snippet is followed by a one-line description of when and why to use it. Prints well in landscape, two columns.

## 0. Imports and the universal skeleton

```python
import cv2, numpy as np, matplotlib.pyplot as plt, pandas as pd
img = cv2.imread('a.jpg')                      # BGR uint8 (H,W,3); None if path is wrong
if img is None: print('Error: Image not found.')
```

Start every answer with this. The `None` check is an easy mark.

## 1. Read, write, show, convert

| Task | Code | Description |
| --- | --- | --- |
| Read colour / gray | `cv2.imread(p)` / `cv2.imread(p, cv2.IMREAD_GRAYSCALE)` | Colour is BGR. Gray loads 2-D directly. |
| Save | `cv2.imwrite('out.png', img)` | Needs BGR or gray, uint8. Use for "save the final output". |
| BGR → RGB | `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | Required before `plt.imshow`. |
| BGR → gray | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Most algorithms need single channel. |
| gray → BGR | `cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)` | Before stacking with colour or drawing colour on it. |
| BGR → HSV | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | Colour segmentation. H is 0–179. |
| BGR → YCrCb | `cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)` | Equalise channel 0 only for colour images. |
| Show (notebook) | `plt.imshow(rgb)`; gray: `plt.imshow(g, cmap='gray')` | `cmap='gray'` or you get false colours. |
| Show (window) | `cv2.imshow('t', img); cv2.waitKey(0); cv2.destroyAllWindows()` | Scripts only; `waitKey(25)` in video loops. |
| Split / merge | `b,g,r = cv2.split(img)`; `cv2.merge([b,g,r])` | Channel work. Order is B, G, R. |
| Shape / dtype | `h, w = img.shape[:2]`; `img.dtype` | `shape` is (H, W, C). |
| Pixel | `img[y, x]` | Row first. Colour pixel is `[B, G, R]`. |
| Crop (ROI) | `img[y1:y2, x1:x2]` | NumPy slicing, no OpenCV call. Left half: `img[:, :w//2]`. Centre 300×300: `img[h//2-150:h//2+150, w//2-150:w//2+150]`. |
| Resize | `cv2.resize(img, (w2, h2))` or `cv2.resize(img, None, fx=.5, fy=.5)` | Size is (width, height). Shrink: `INTER_AREA`; enlarge: `INTER_LINEAR`/`INTER_CUBIC`. |
| Copy | `img.copy()` | Draw on a copy, not the original. |
| Blank canvas | `np.zeros((H, W, 3), np.uint8)` | Black 3-channel. `np.full((H,W,3), 255, np.uint8)` is white. |

## 2. Drawing and text

```python
cv2.circle(img, (cx, cy), r, (B,G,R), -1)         # thickness -1 = filled; draw LARGEST first
cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)  # top-left, bottom-right
cv2.line(img, (x1, y1), (x2, y2), color, 2)
cv2.polylines(img, [pts.astype(np.int32)], True, color, 3)   # True = closed
cv2.fillPoly(mask, [poly_int32], 255)             # filled polygon (ROI mask)
cv2.putText(img, 'Hi', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2, cv2.LINE_AA)  # (x,y)=bottom-left
```

Colours are **BGR** when drawing on an OpenCV image: red is `(0,0,255)`. On an image already converted to RGB, red is `(255,0,0)`.

## 3. Arithmetic, blending and masks

| Operation | Code | Description |
| --- | --- | --- |
| Saturating add | `cv2.add(a, b)` | Caps at 255. NumPy `a+b` on uint8 **wraps** (250+20=14). |
| Subtract | `cv2.subtract(a, b)` | Floors at 0. |
| Weighted blend | `cv2.addWeighted(a, α, b, β, γ)` | dst = αa + βb + γ. Same size and type required. Semi-transparent overlay: `addWeighted(overlay, .5, img, .5, 0)`. |
| AND with mask | `cv2.bitwise_and(a, a, mask=m)` | Keeps pixels where mask is white. Mask = single-channel uint8. |
| OR / NOT / XOR | `cv2.bitwise_or(a, b)`, `cv2.bitwise_not(m)`, `cv2.bitwise_xor` | Combine regions; NOT inverts the mask. |
| Cut-and-paste pattern | `fg = and(A,A,mask=m)`; `bg = and(B,B,mask=not(m))`; `out = or(fg,bg)` | Foreground from A, background from B. |
| Match sizes | `b = cv2.resize(b, (a.shape[1], a.shape[0]))` | Before any arithmetic between two images. |
| Clip + cast | `np.clip(x, 0, 255).astype(np.uint8)` | After float maths, before display. |

## 4. Point (photometric) transforms, Lab 2

| Transform | Formula | Code | Description |
| --- | --- | --- | --- |
| Negative | s = 255 − r | `255 - img` | Inverts. |
| Log | s = c·log(1+r), c = 255/log(1+max) | `c=255/np.log(1+img.max()); (c*np.log(1+img.astype(np.float32))).astype(np.uint8)` | Brightens **dark** areas, compresses bright. Cast to float first. |
| Gamma | s = 255·(r/255)^γ | `(255*(img/255.0)**g).astype(np.uint8)` | γ < 1 brighter midtones, γ > 1 darker. |
| Gamma via LUT | same | `lut=np.array([255*(i/255)**g for i in range(256)]).astype(np.uint8); cv2.LUT(img, lut)` | Fast; any custom 256-entry curve works. |
| Contrast stretch | s = (r − min)·255/(max − min) | `cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)` | Linear stretch to full range. |
| Piecewise linear | segments between (r1,s1), (r2,s2) | `np.interp(img, [0,r1,r2,255], [0,s1,s2,255]).astype(np.uint8)` | Custom contrast curve in one line. |
| Threshold | s = 255 if r > T else 0 | `cv2.threshold(g, T, 255, cv2.THRESH_BINARY)` | Binary mask, keep only dense/bright tissue. |
| Equalise | CDF mapping | `cv2.equalizeHist(gray)` | Gray 8-bit only. Flattens histogram. |
| CLAHE | tile-wise equalise with clip | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)` | Local contrast; limits noise. |
| Colour equalise | luminance only | `y=cv2.cvtColor(i,cv2.COLOR_BGR2YCrCb); y[:,:,0]=cv2.equalizeHist(y[:,:,0]); cv2.cvtColor(y,cv2.COLOR_YCrCb2BGR)` | Keeps colours natural. |
| False colour | intensity → hue | `cv2.applyColorMap(gray8, cv2.COLORMAP_JET)` | Returns BGR. Helps the eye see subtle differences. |
| Colour balance | scale channels to equal mean | `k=mean_all/ch.mean(); ch*k` | Removes colour cast (grey-world). |
| Histogram | counts per level | `cv2.calcHist([img],[0],None,[256],[0,256])` | Args: images, channels, mask, bins, ranges. |

## 5. Geometric transforms, Lab 3

**Matrices** (apply to column vector \[x, y\] or \[x, y, 1\])

| Type | Matrix | Note |
| --- | --- | --- |
| Scale | `[[sx,0],[0,sy]]` | Grows from origin; to scale about the centre use t = c − s·c |
| Rotation | `[[cosθ,−sinθ],[sinθ,cosθ]]` | OpenCV +angle is counter-clockwise |
| Reflection | `[[1,0],[0,−1]]` (about x-axis) | Negative diagonal entry flips |
| Shear (x) | `[[1,k],[0,1]]` | x' = x + ky |
| Translation (3×3) | `[[1,0,tx],[0,1,ty],[0,0,1]]` | Needs homogeneous coordinates |
| Affine (2×3) | `[[a,b,tx],[c,d,ty]]` | Input to `warpAffine` |
| Homography (3×3) | `H` with `h33` normalised | Divide by w'; input to `warpPerspective` |

```python
M = cv2.getRotationMatrix2D((cx, cy), angle, scale)        # 2x3, centre is (x, y)
out = cv2.warpAffine(img, M, (w, h))                       # dsize = (width, height)
M = cv2.getAffineTransform(src3, dst3)                     # 3 point pairs, float32 (3,2)
M = cv2.getPerspectiveTransform(src4, dst4)                # 4 point pairs, float32 (4,2)
out = cv2.warpPerspective(img, M, (w, h))
H, m = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)       # many noisy matches
cv2.perspectiveTransform(pts.reshape(-1,1,2), H)           # transform points only
Minv = cv2.invertAffineTransform(M)                        # undo an affine
M3 = np.vstack([M, [0,0,1]])                               # 2x3 → 3x3 for chaining
```

**Compose:** "scale, then rotate, then translate" = `T @ R @ S` (rightmost first). Use `M3[:2]` for `warpAffine`.

**No-crop rotation:** `new_w = h|sinθ| + w|cosθ|`, `new_h = h|cosθ| + w|sinθ|`, then `M[0,2] += new_w/2 - cx; M[1,2] += new_h/2 - cy`; output size `(int(new_w), int(new_h))`.

**Manual rotation about (cx, cy):** `[[cos, sin, (1−cos)cx − sin·cy], [−sin, cos, sin·cx + (1−cos)cy]]` (scale 1).

**Shear canvas:** `new_w = int(w + abs(k)*h)`. **Translation:** shift right/down = positive tx, ty.

**Hierarchy:** rigid (R+T, 3 DoF) ⊂ similarity (+uniform scale, 4) ⊂ affine (6) ⊂ projective (8). Affine keeps parallelism; similarity keeps angles; rigid keeps lengths; projective keeps only straight lines.

**Point order for 4-point warps:** TL, TR, BR, BL, same order in `src` and `dst`.

## 6. Filtering and convolution, Lab 4

| Goal | Code | Description |
| --- | --- | --- |
| Box blur | `cv2.blur(img,(k,k))` or `filter2D` with `np.ones((k,k))/k**2` | Plain average. |
| Gaussian blur | `cv2.GaussianBlur(img,(k,k),0)` | k odd; sigma 0 = auto. Smoother than box. |
| Median blur | `cv2.medianBlur(img, 5)` | Best against salt-and-pepper noise; k odd. |
| Bilateral | `cv2.bilateralFilter(img, 9, 75, 75)` | Smooths but keeps edges. |
| Custom kernel | `cv2.filter2D(img, -1, kernel)` | -1 = same depth. Output **same size**; no `strides` argument. |
| Gaussian kernel | `cv2.getGaussianKernel(k, sigma)` | Returns k×1; 2-D = `g @ g.T`. |
| Sharpen | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` | Centre-boosting kernel, sums to 1. |
| Emboss | `[[-2,-1,0],[-1,1,1],[0,1,2]]` | 3-D relief effect. |
| Sobel X / Y kernels | `[[1,0,-1],[2,0,-2],[1,0,-1]]` / its transpose | Horizontal change (vertical edges) / vertical change. |
| Border handling | `borderType=cv2.BORDER_CONSTANT / REFLECT / REPLICATE / DEFAULT` | Constant = zero padding. |

**Output size** = floor((N + 2P − K)/S) + 1. Valid: P=0 (smaller). Same: P=(K−1)/2 (equal at S=1). Stride: slice `result[::s, ::s]`.

**Convolution vs correlation:** true convolution flips the kernel; `filter2D` does correlation (no flip). Same result for symmetric kernels.

## 7. Feature descriptors, Lab 4

```python
from skimage.feature import hog, local_binary_pattern
feat, vis = hog(gray, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=True)   # gradient orientation histograms
lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')                           # values 0..P+1
h,_ = np.histogram(lbp.ravel(), bins=np.arange(0,P+3), range=(0,P+2)); h = h/(h.sum()+1e-6)   # P+2 bins
gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
mag = np.sqrt(gx**2 + gy**2); ang = np.arctan2(gy, gx)*180/np.pi
hist,_ = np.histogram(np.mod(ang,360), bins=8, range=(0,360))        # HED: 8 bins over 360°
hist,_ = np.histogram(np.mod(ang,180), bins=9, range=(0,180))        # HIG/HOG-style: 9 bins over 180°
energy   = cv2.filter2D(g.astype(np.float32)**2, -1, np.ones((3,3)))   # texture energy
contrast = np.sqrt(np.maximum(cv2.blur(f**2,(3,3)) - cv2.blur(f,(3,3))**2, 0))   # local std-dev
```

| Descriptor | Captures | Remember |
| --- | --- | --- |
| HOG | Shape via gradient directions per cell | 8×8 cells, 2×2 blocks, 9 bins; block normalisation for lighting |
| LBP | Micro-texture | Compare neighbours with centre (≥ → 1); 'uniform' gives P+2 bins |
| Colour hist | Colour distribution | Bins in RGB/HSV; ignores layout |
| HED | Dominant edge direction | Canny/Sobel, 8 bins of 45° |
| Texture energy | Uniformity | High = complex, less uniform |
| Texture contrast | Variation | Std-dev in a window |

## 8. Edge detection, Labs 4–6

```python
blur = cv2.GaussianBlur(gray, (5,5), 1.4)
sx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)      # Gx: responds to VERTICAL edges
sy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)      # Gy: responds to HORIZONTAL edges
mag = np.uint8(np.clip(np.sqrt(sx**2+sy**2), 0, 255))
scx = cv2.Scharr(blur, cv2.CV_64F, 1, 0)             # more rotation-accurate than Sobel 3x3
lap = cv2.convertScaleAbs(cv2.Laplacian(blur, cv2.CV_64F))   # LoG when run after Gaussian
edges = cv2.Canny(blur, 50, 150)                     # low, high
```

| Detector | Mechanism | Strength | Weakness |
| --- | --- | --- | --- |
| Sobel/Prewitt/Scharr | 1st derivative (gradient) | Simple, gives direction | Thick edges, noise-sensitive |
| Laplacian / LoG | 2nd derivative, zero-crossings | Finds edges at all directions | Very noise-sensitive without Gaussian |
| Canny | Blur → gradient → NMS → hysteresis | Thin, continuous, low noise | Needs two thresholds |

**Canny thresholds:** high ≈ 2–3× low. Above high = strong; between = weak (kept only if linked to strong); below low = rejected. Raise both to remove noise, lower both to catch faint edges.

**Why CV\_64F:** negative gradients survive. Convert for display with `np.uint8(np.clip(x,0,255))` or `cv2.convertScaleAbs`.

## 9. Segmentation, Lab 5

**Thresholding**

```python
_, b = cv2.threshold(g, T, 255, cv2.THRESH_BINARY)                  # also _INV, _TRUNC, _TOZERO, _TOZERO_INV
T, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)   # auto T (pass 0), bimodal histogram
b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)   # blockSize odd, C
mask = cv2.inRange(hsv, (20,100,100), (35,255,255))                 # colour mask
```

| Parameter | Small / low | Large / high |
| --- | --- | --- |
| Adaptive `blockSize` | Noisy, broken strokes | Behaves like a global threshold |
| Adaptive `C` | More white pixels, noise | Fewer white pixels, thinner strokes |
| Canny thresholds | Many weak/noisy edges | Missing faint edges |
| Region-growing threshold | Under-segmentation | Leaks into neighbours |
| Distance-transform fraction | Touching objects stay merged | Small objects lose markers |
| K in K-Means | Heavy colour simplification | Closer to original |

**OpenCV HSV hues (0–179):** red 0–10 and 170–179 (two ranges, OR them), orange 10–20, yellow 20–35, green 35–85, cyan 85–100, blue 100–130, purple 130–160. S and V 0–255.

**Morphology**

```python
k = np.ones((3,3), np.uint8)
cv2.erode(m, k); cv2.dilate(m, k)
cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=2)    # erode then dilate: removes small noise
cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)                 # dilate then erode: fills small holes
```

**Marker-based watershed (memorise the order)**

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
op = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=2)   # noise removal
bg = cv2.dilate(op, np.ones((3,3),np.uint8), iterations=3)                         # sure background
dist = cv2.distanceTransform(op, cv2.DIST_L2, 5)
_, fg = cv2.threshold(dist, 0.5*dist.max(), 255, 0); fg = np.uint8(fg)             # sure foreground
unk = cv2.subtract(bg, fg)                                                          # unknown
_, mk = cv2.connectedComponents(fg); mk = mk + 1; mk[unk == 255] = 0                # bg=1, unknown=0
mk = cv2.watershed(img, mk); img[mk == -1] = (0, 0, 255)                            # boundaries = -1
```

**K-Means colour segmentation**

```python
Z = np.float32(img.reshape(-1, 3))
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
ret, labels, centers = cv2.kmeans(Z, K, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
seg = np.uint8(centers)[labels.flatten()].reshape(img.shape)
```

**Region growing skeleton:** stack/queue with seed; pop pixel; skip if out of bounds or already in mask; if `abs(int(img[p]) - int(img[seed])) <= T` mark white and push 4 neighbours. Seed is `(row, col)`. Use `int()` to avoid uint8 wrap-around.

**Contours (shape analysis after a mask)**

```python
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in cnts:
    area = cv2.contourArea(c); x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
print('objects:', len(cnts))
```

## 10. Hough, SIFT and matching, Lab 6

```python
lines  = cv2.HoughLines(edges, 1, np.pi/180, 150)                  # (rho, theta); rho = x cosθ + y sinθ
linesP = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=40, maxLineGap=10)   # (x1,y1,x2,y2)
circ   = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=40, minRadius=10, maxRadius=80)
for x,y,r in np.uint16(np.around(circ[0])): cv2.circle(img,(x,y),r,(0,255,0),3)    # check `circ is not None`
```

| Hough parameter | Meaning | Tune |
| --- | --- | --- |
| `rho`, `theta` | Accumulator resolution (px, radians) | Finer = slower, more precise |
| `threshold` | Votes needed | Lower → more lines |
| `minLineLength` | Shortest segment kept | Raise to drop clutter |
| `maxLineGap` | Gap bridged between collinear pieces | Raise to join dashed lanes |
| `dp` | Inverse accumulator resolution | 1–1.5 |
| `minDist` | Min distance between circle centres | About one radius |
| `param1` | Upper Canny threshold | 100–200 |
| `param2` | Accumulator threshold | Lower → more (false) circles |

**Lane recipe:** gray → blur → Canny → ROI mask (`fillPoly` trapezoid + `bitwise_and`) → `HoughLinesP` → filter by slope → draw. **Hough idea:** edge pixels vote in parameter space (line: ρ,θ; circle: a,b,r), peaks are shapes.

**SIFT pipeline**

```python
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)                        # des shape (N,128) float32
vis = cv2.drawKeypoints(rgb, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
bf = cv2.BFMatcher(cv2.NORM_L2)
m = bf.knnMatch(des1, des2, k=2)
good = [a for a,b in m if a.distance < 0.75*b.distance]            # Lowe ratio test
src = np.float32([kp1[g.queryIdx].pt for g in good]).reshape(-1,1,2)
dst = np.float32([kp2[g.trainIdx].pt for g in good]).reshape(-1,1,2)
H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
box = cv2.perspectiveTransform(np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2), H)
cv2.polylines(frame, [np.int32(box)], True, (0,255,0), 3)
res = cv2.drawMatches(img1, kp1, img2, kp2, good[:50], None, flags=2)
```

**SIFT steps:** (1) DoG scale-space extrema → (2) keypoint localisation, 26-neighbour check (8+9+9), reject low contrast/edges → (3) orientation assignment (rotation invariance) → (4) 128-D descriptor (4×4 cells × 8 bins) → (5) match by Euclidean distance → (6) RANSAC outlier rejection → (7) homography. Scale and rotation invariant. Binary descriptors (ORB) use `NORM_HAMMING`.

**Panorama:** `H = findHomography(pts_img2, pts_img1)`; `pano = warpPerspective(img2, H, (w1+w2, max(h1,h2)))`; `pano[:h1,:w1] = img1`.

## 11. Wavelets, Lab 6

```python
import pywt   # pip install PyWavelets
cA, (cH, cV, cD) = pywt.dwt2(gray, 'haar'); rec = pywt.idwt2((cA,(cH,cV,cD)), 'haar')
co = pywt.wavedec(sig, 'db4', level=4)                                  # [cA4, cD4, cD3, cD2, cD1]
sigma = np.median(np.abs(co[-1]))/0.6745; thr = sigma*np.sqrt(2*np.log(len(sig)))
co[1:] = [pywt.threshold(c, thr, mode='soft') for c in co[1:]]
den = pywt.waverec(co, 'db4')[:len(sig)]; resid = sig - den
anom = np.where(np.abs(resid) > 3*resid.std())[0]                       # 3-sigma anomalies
```

| Transform | Key facts |
| --- | --- |
| CWT | Continuous scales and shifts; Morlet; time-frequency analysis, seismic signals |
| DWT | Dyadic scales; cA = low-pass approximation, cD = high-pass detail; Haar, Daubechies, Symlet; compression and denoising |
| WPT | Also splits the detail branches; richer sub-bands; feature extraction |
| Inverse | Rebuilds from coefficients; needed after thresholding |
| 2-D DWT | LL (approx), LH, HL, HH (details) |

## 12. Video loop

```python
cap = cv2.VideoCapture('v.mp4')                  # 0 = webcam
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break                            # end of file
    # ... process frame ...
    cv2.imshow('win', np.hstack((frame, out)))   # same height and channels
    if cv2.waitKey(25) & 0xFF == ord('q'): break
cap.release(); cv2.destroyAllWindows()
bg = cv2.createBackgroundSubtractorMOG2()        # motion mask: fg = bg.apply(frame)
```

## 13. Matplotlib patterns

```python
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(rgb); plt.title('Original', fontsize=14, color='darkred', pad=10); plt.axis('off')
plt.subplot(1,2,2); plt.imshow(g, cmap='gray'); plt.title('Result'); plt.axis('off')
plt.tight_layout(); plt.show()
plt.hist(gray.ravel(), 256, [0,256]); plt.bar(x, y); plt.plot(h); plt.axvline(T, color='r')
```

## 14. Python and Lab 1 essentials

```python
class GroceryManager:
    def __init__(self): self.items = {}
    def add_item(self, item, qty, price): self.items[item] = {'quantity': qty, 'price': price}
    def remove_item(self, item):
        if item not in self.items: print('Error: not found'); return
        del self.items[item]
    def calculate_total(self): return sum(d['quantity']*d['price'] for d in self.items.values())

best = max(students, key=lambda s: sum(students[s]['Grades'])/len(students[s]['Grades']))
flat = img.reshape(-1, 3); df = pd.DataFrame(flat, columns=['B','G','R']); print(df.describe())
```

Dictionary tools: `d.items()`, `d.get(k, default)`, `k in d`, `sorted(d, key=...)`. Wrap risky code in `try/except`.

## 15. Traps and numbers to memorise

| Trap | Truth |
| --- | --- |
| `plt.imshow` colours wrong | Convert BGR → RGB |
| `img[x, y]` | It is `img[y, x]` |
| `cv2.resize(img, (h, w))` | It is `(width, height)` |
| `warpAffine(..., (h, w))` | It is `(width, height)` |
| Even kernel size | Must be odd |
| `cv2.add` vs `+` | Saturate vs wrap |
| `equalizeHist` on colour | Gray only; use YCrCb luminance |
| `kmeans` on uint8 | Needs float32 |
| Watershed background = 0 | Add 1; unknown = 0 |
| Otsu with manual T | Pass 0 and add `THRESH_OTSU` |
| `filter2D` strides | Not supported; output same size |
| Hough returns None | Check `is not None` |
| Point arrays for warps | `np.float32`, order TL TR BR BL |
| Mask for `bitwise_and` | Single-channel uint8 with `mask=` |
| Gamma < 1 | Brightens |
| SIFT norm | `NORM_L2`, 128-D, ratio test 0.75 |

| Number | Meaning |
| --- | --- |
| 255 / 0 | White / black (uint8) |
| 179 | Max OpenCV hue |
| 128 | SIFT descriptor length |
| 26 | SIFT keypoint neighbours (8+9+9) |
| 3 / 4 | Point pairs for affine / perspective |
| 6 / 8 | Degrees of freedom: affine / homography |
| P + 2 | LBP uniform bin count |
| 9 | Default HOG orientation bins |
| 3780 | HOG length for 64×128 window (8×8 cells, 2×2 blocks, 9 bins) |
| (N+2P−K)/S + 1 | Convolution output size |
