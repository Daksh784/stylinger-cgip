# Virtual Fashion Try-On: Architecture Inspection & Implementation Plan

## 1. Project Architecture Inspection

### Frontend & Routing (React + Vite + TanStack + Tailwind)
- **Framework & Router**: React 19 with TanStack Start / TanStack Router using file-based routing (`src/routes/`). Layout is managed via `DashShell` in `src/components/site/dash-shell.tsx` and route files like `src/routes/app.tsx`.
- **Navigation**: `src/routes/app.tsx` manages customer navigation tabs (`Home`, `My Wardrobe`, `AI Suggestions`, `Daily Outfit`, etc.). We will integrate a dedicated `Virtual Try-On` route (`/app/tryon`) directly into the customer navigation.
- **Product & Wardrobe Catalog**:
  - Products: `src/lib/useProducts.ts` provides TanStack Query hooks over `src/lib/db.ts` (localStorage-backed database with 30 seeded luxury products including high-resolution images across Outerwear, Shirts, T-Shirts, Jeans, Knitwear, Dresses).
  - Wardrobe: `src/lib/mock-data.ts` provides indexed wardrobe items (`wardrobeItems`) categorized by Tops, Outerwear, Bottoms, Dresses.
- **UI Components**: Existing Radix UI / Tailwind primitives in `src/components/ui/` (`Button`, `Card`, `Slider`, `Tabs`, `Dialog`, `Badge`, `Sonner`, `Skeleton`, `Popover`, `Select`) and dark-luxury styling tokens (`--color-emerald`, `--color-surface`, `glass`, `font-sans`, `font-serif`).

### Database & Supabase Integration
- **Migrations & Schemas**: Detailed in `supabase/migrations/` (23 relational tables including `profiles`, `products`, `product_images`, `wardrobes`, `wardrobe_items`, `outfits`, `outfit_items`).
- **Storage**: Configured in `008_storage.sql` with buckets `product-images`, `wardrobe-images`, `avatars`, `future-ai-assets`.
- **Constraint Compliance**: All existing Supabase schemas and RLS policies remain completely untouched; no service keys or secrets will be exposed.

---

## 2. Computer Graphics and Image Processing (CG/IP) Architecture

To adhere to the project proposal constraints:
* **No black-box deep learning or paid external AI APIs** will be used.
* The pipeline is built purely on **classical Computer Vision and Image Processing** using Python, OpenCV (`cv2`), and NumPy.
* Anatomical landmarks cannot be reliably extracted from unconstrained 2D photos via simple contours alone. Therefore, we use **contour-based subject bounding & centroid estimation** paired with **user-interactive offset/scale adjustment** to deliver an intuitive, controllable 2D fitting preview.

```
+---------------------------------------------------------------------------------+
|                               CG/IP Pipeline Flow                               |
|                                                                                 |
| [User Photo] ---> [1. Preprocessing] ---> [2. Segmentation]                     |
|                   - Aspect normalization  - Otsu / Adaptive Thresholding        |
|                   - Gaussian / Median     - Morphological Opening/Closing       |
|                   - CLAHE / Contrast      - Subject Contour & Centroid          |
|                                                      |                          |
| [Garment Img] ---> [3. Transformation] <-------------+                          |
|                   - Alpha / Chroma Cutout - Torso Bounding Region               |
|                   - Affine Scaling & Warp - Interactive X/Y/Scale offsets       |
|                                          |                                      |
|                                          v                                      |
|                                 [4. Compositing]                                |
|                                 - Edge Feathering (Gaussian)                    |
|                                 - Alpha-Mask Blending                           |
|                                 - Lighting & Color Harmony                      |
|                                          |                                      |
|                                          v                                      |
|                                [Final 2D Preview]                               |
+---------------------------------------------------------------------------------+
```

### Module Breakdown (`tryon-service/`)
1. **`src/preprocessing.py`**:
   - `decode_image_safely()`: Validates magic numbers and decodes image bytes in-memory.
   - `normalize_dimensions()`: Resizes large inputs maintaining aspect ratio (e.g., max 1280px).
   - `reduce_noise()`: Applies Gaussian Blur ($\sigma = 1.0$) or Median filter.
   - `adjust_contrast_brightness()`: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space or linear scaling.
2. **`src/segmentation.py`**:
   - `segment_subject()`: Color/intensity segmentation using Otsu thresholding + GrabCut rectangular initialization.
   - `extract_subject_contour()`: Finds external contours, identifies largest body silhouette, calculates centroid and approximate torso region ($y \in [0.20H, 0.65H]$).
3. **`src/transform.py`**:
   - `isolate_garment_mask()`: Removes plain backgrounds (white/solid) using HSV color thresholding or utilizes existing alpha channel.
   - `compute_garment_transform()`: Computes affine transformation matrix ($M$) based on target torso bounding box, user offsets ($\Delta x, \Delta y$), scale factor ($s$), and rotation ($\theta$).
   - `warp_garment()`: Warps and maps the garment and its alpha mask into user image coordinate space.
4. **`src/compositing.py`**:
   - `feather_mask()`: Applies Gaussian blurring to garment alpha borders to prevent jagged seams.
   - `blend_garment()`: Performs linear alpha blending ($I_{res} = \alpha I_{garment} + (1-\alpha) I_{user}$) with optional illumination matching.
5. **`src/pipeline.py`**:
   - `run_tryon_pipeline()`: End-to-end execution returning processed PNG/JPEG bytes, bounding coordinates, and execution telemetry.
6. **`app.py`**:
   - Lightweight Flask service on port 5001 with CORS for local development.
   - In-memory processing: personal images are never permanently written to disk or logged.
   - Endpoints: `GET /api/health`, `POST /api/tryon/process`, `POST /api/tryon/segment-preview`.

---

## 3. Frontend Implementation Plan (`Stylinger` App)

1. **Navigation Entry**:
   - Add `{ to: "/app/tryon", label: "Virtual Try-On", icon: Sparkles }` in `src/routes/app.tsx`.
2. **Customer Try-On Route (`src/routes/app.tryon.tsx`)**:
   - **Step 1: Upload User Photo**: Front-facing, full-body photo upload (drag & drop, file size check $\le 10\text{MB}$, format validation `jpg, jpeg, png`, immediate client preview).
   - **Step 2: Garment Selection**: Tabbed selection between **Stylinger Catalog** (`useProducts()`) and **My Wardrobe** (`wardrobeItems`), with category filtering (All, Tops, Outerwear, Dresses, Bottoms).
   - **Step 3: Interactive Fitting Controls**: Sliders for Vertical Position ($Y$), Horizontal Position ($X$), Garment Scale ($70\% - 150\%$), and Mask Feathering/Opacity.
   - **Step 4: Render & Preview**: Real-time or on-demand processing call to the Python service with loading indicators, execution telemetry, and fallback simulation mode if the backend is unreachable.
   - **Step 5: Compare & Download**:
     - Interactive Before/After split comparison slider.
     - Outfit Comparison tray to compare multiple garments side-by-side.
     - High-resolution download button.
   - **Step 6: Clear Limitations Disclaimer**: Educational banner noting the 2D CG/IP nature of the preview.

---

## 4. Privacy, Security & Testing

- **Privacy**: In-memory processing via `io.BytesIO`. Temporary photo data is wiped immediately after response.
- **Security**: Request body size limit (16MB), strict MIME and magic header checks with PIL and OpenCV, no disk path traversal risks.
- **Unit Testing**:
  - `tests/test_validation.py`: Invalid formats, oversize images.
  - `tests/test_preprocessing.py`: Normalization, filtering.
  - `tests/test_segmentation.py`: Contour extraction, centroid and torso bounding.
  - `tests/test_transform.py`: Affine warping, scaling.
  - `tests/test_compositing.py`: Alpha feathering and blending.
  - `tests/test_api.py`: Flask endpoint responses and error states.
