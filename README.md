# Stylinger — Virtual Fashion Try-On (2D CG/IP Engine)

This document provides setup instructions, architectural documentation, and verification results for the **Virtual Fashion Try-On** feature in Stylinger.

---

## 1. Architectural Overview & Data Flow

The Virtual Try-On feature integrates directly with Stylinger's existing React, TypeScript, TanStack Router, Tailwind CSS, and Supabase architecture without replacing any existing services or breaking customer/merchant portals.

The feature is powered by a dedicated Python microservice built using **classical Computer Graphics and Image Processing (CG/IP)** techniques with **OpenCV** and **NumPy**. In accordance with the project proposal, no pre-trained deep-learning models or paid external AI APIs are used.

```
+---------------------------------------------------------------------------------------------------------+
|                                        End-to-End Try-On Architecture                                   |
|                                                                                                         |
|   [Stylinger Web Frontend]                                       [Flask CG/IP Service :5001]             |
|   React 19 + TanStack Router                                      Python 3.13 + OpenCV + NumPy          |
|                                                                                                         |
|   1. Upload Full-Body Photo (.jpg/.png)                                                                 |
|   2. Select Garment (Catalog / Wardrobe)                                                                |
|   3. Adjust Placement Sliders (scale, X/Y)                                                              |
|   4. Click "Process Virtual Try-On"  ====== POST /api/tryon/process =====>  1. Validate & In-Memory Decode |
|                                                                            2. Preprocessing & Denoising |
|                                                                            3. Subject Segmentation       |
|                                                                               & Torso Estimation        |
|                                                                            4. Garment Alpha Extraction   |
|                                                                            5. Affine Warp & Alignment    |
|                                                                            6. Feathering & Compositing  |
|                                                                            7. Luminance Balancing       |
|   5. Interactive Split Comparison    <===== Returns PNG base64 + Telemetry ============================ |
|   6. Compare Looks & Download PNG                                                                       |
+---------------------------------------------------------------------------------------------------------+
```

---

## 2. File Manifest (What Was Added)

| File | Type | Description |
|---|---|---|
| [`src/routes/app.tryon.tsx`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/src/routes/app.tryon.tsx) | Frontend Route | Customer-facing Virtual Try-On studio: photo upload, product/wardrobe selection, placement sliders, split comparison slider, session history, and PNG download. |
| [`src/routes/app.tsx`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/src/routes/app.tsx) | Frontend Route | Updated customer navigation to include `/app/tryon` with `Wand2` icon. |
| [`tryon-service/app.py`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/app.py) | Python Service | Flask microservice providing `/api/health`, `/api/tryon/process`, and `/api/tryon/segment-preview`. Enforces strict in-memory execution and CORS. |
| [`tryon-service/requirements.txt`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/requirements.txt) | Python Config | Service dependencies: `flask`, `flask-cors`, `opencv-python-headless`, `numpy`, `pillow`, `pytest`, `requests`. |
| [`tryon-service/.env.example`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/.env.example) | Python Config | Documented environment settings for service port and CORS origins. |
| [`tryon-service/src/preprocessing.py`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/src/preprocessing.py) | Python Module | In-memory decoding with PIL header checks, aspect-ratio normalization, Gaussian/median denoising, and CLAHE contrast correction. |
| [`tryon-service/src/segmentation.py`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/src/segmentation.py) | Python Module | Subject silhouette segmentation via GrabCut and Otsu thresholding, contour extraction, spatial centroid calculation, and 2D torso envelope estimation. |
| [`tryon-service/src/transform.py`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/src/transform.py) | Python Module | Garment alpha mask isolation (handling studio white backgrounds and RGBA transparent images), category-based target boxes, and affine transformations. |
| [`tryon-service/src/compositing.py`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/src/compositing.py) | Python Module | Edge feathering using Gaussian filters, ambient luminance matching, and linear alpha compositing. |
| [`tryon-service/src/pipeline.py`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/src/pipeline.py) | Python Module | Coordinates the complete end-to-end processing pipeline and gathers execution telemetry. |
| [`tryon-service/tests/`](file:///C:/Users/mohit/OneDrive/Desktop/Stylinger/tryon-service/tests/) | Test Suite | 23 unit tests covering validation, preprocessing, segmentation, transforms, compositing, and Flask endpoints. |

---

## 3. Installation and Setup Instructions

### Prerequisites
- Node.js v20+ / v22+
- Python 3.10+ (tested and verified on Python 3.13)

### Running the Python Try-On Service
1. Open a terminal in the project directory:
   ```powershell
   cd C:\Users\mohit\OneDrive\Desktop\notes\stylinger
   ```
2. Create and activate a virtual environment (already set up in `tryon-service/.venv`):
   ```powershell
   python -m venv tryon-service/.venv
   .\tryon-service\.venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r tryon-service/requirements.txt
   ```
4. Start the service:
   ```powershell
   python tryon-service/app.py
   ```
   The service will run on `http://127.0.0.1:5001`.

### Running the Stylinger Frontend
1. In another terminal window:
   ```powershell
   cd C:\Users\mohit\OneDrive\Desktop\notes\stylinger
   npm.cmd run dev
   ```
2. Open **`http://localhost:5173/app/tryon`** in your browser.

---

## 4. Privacy and Security Implementation

- **Strict In-Memory Processing**: Uploaded personal photographs are read directly into memory (`io.BytesIO`) and NumPy buffers. Images are **never saved to the server filesystem**, preventing photo leaks and path traversal attacks (`../`).
- **File & Magic Header Validation**: Files are checked using PIL's binary header verifier (`Image.open().verify()`) and OpenCV's safe buffer decoder (`cv2.imdecode`) rather than trusting user-submitted MIME strings.
- **Request Size Boundaries**: Enforced 16MB max upload limit on Flask (`MAX_CONTENT_LENGTH`) and 10MB limit in the frontend file picker.
- **CORS Protection**: Access to `/api/*` is restricted to authorized frontend origins (`http://localhost:5173`, `http://localhost:8080`, `http://127.0.0.1:5173`).
- **No Secret Leakage**: Supabase service keys and credentials remain untouched in environment variables; no server keys are exposed on the client.

---

## 5. Verification and Test Results

### Automated Python Test Suite (`pytest`)
Ran all 23 unit tests using the virtual environment pytest runner:
```powershell
.\tryon-service\.venv\Scripts\pytest.exe tryon-service/tests
```
**Results:**
```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.6.0
collected 23 items

tryon-service\tests\test_api.py .....                                    [ 21%]
tryon-service\tests\test_compositing.py ...                              [ 34%]
tryon-service\tests\test_preprocessing.py .....                          [ 56%]
tryon-service\tests\test_segmentation.py ..                              [ 65%]
tryon-service\tests\test_transform.py ....                               [ 82%]
tryon-service\tests\test_validation.py ....                              [100%]

============================= 23 passed in 3.51s ==============================
```

### TypeScript and Production Build Verification
- `npx.cmd tsc --noEmit`: Exited with code 0 (zero type errors).
- `npm.cmd run build`: Generated all client and server assets via Vite and Nitro cleanly in 2.33s.
- Dev server runtime check: Served HTTP 200 on `http://localhost:5173/app/tryon`.
- Flask health check: Returned HTTP 200 with status `"healthy"` on `http://127.0.0.1:5001/api/health`.

---

## 6. Known Limitations & Methodology Disclosure

- **2D Image Warping vs. 3D Simulation**: Classical image processing uses 2D geometric scaling, translation, and affine warping. It does not account for 3D body depth, fabric draping, wrinkles, or physical cloth collisions.
- **Silhouette-Based Torso Estimation**: Classical contour detection extracts the subject's external silhouette and estimates torso bounds using proportional ratios (top ~18% to ~58% of body height). It does not compute 3D skeletal landmarks.
- **Interactive Correction**: Users can use the interactive Placement & Blending sliders (Scale, X offset, Y offset, Feather Radius, and Opacity) to position clothing onto their silhouette.
