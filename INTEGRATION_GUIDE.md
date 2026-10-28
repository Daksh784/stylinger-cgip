# CGIP (Computer Graphics & Image Processing) — Virtual Fashion Try-On

This folder contains all the files developed for the **Virtual Fashion Try-On** feature based on the Computer Graphics & Image Processing project proposal.

---

## Folder Structure

```
cgip/
├── tryon-service/              # Standalone Python Classical CG/IP Microservice (Flask, OpenCV, NumPy)
│   ├── .venv/                  # Dedicated Python virtual environment
│   ├── app.py                  # API endpoints (/api/health, /api/tryon/process, /api/tryon/segment-preview)
│   ├── requirements.txt        # Python dependencies
│   ├── .env.example            # Environment template
│   ├── src/
│   │   ├── preprocessing.py    # Denoising (Gaussian), aspect normalization, CLAHE contrast
│   │   ├── segmentation.py     # GrabCut & Otsu thresholding, body contour & torso box estimation
│   │   ├── transform.py        # Garment alpha extraction & affine geometric warping
│   │   ├── compositing.py      # Edge feathering, luminance harmonization, linear alpha blending
│   │   └── pipeline.py         # End-to-end pipeline orchestrator & telemetry
│   └── tests/                  # 23 automated unit tests (100% pass)
├── app.tryon.tsx               # Customer-facing React / TanStack Router Virtual Try-On Studio component
├── README.md                   # Comprehensive feature documentation & methodology
├── TRYON_IMPLEMENTATION_PLAN.md# Original architecture inspection & technical plan
└── INTEGRATION_GUIDE.md        # This integration guide
```

---

## How to Run the CGIP Try-On Microservice

From the project root:
```powershell
cd cgip\tryon-service
.\.venv\Scripts\python.exe app.py
```
The service will start on `http://127.0.0.1:5001`.

To run the automated test suite:
```powershell
cd cgip\tryon-service
.\.venv\Scripts\pytest.exe tests
```

---

## How to Connect to the Frontend

If you wish to mount the Virtual Try-On page in the new `frontend/` directory:

1. Copy the route component:
   ```powershell
   Copy-Item cgip\app.tryon.tsx frontend\src\routes\app.tryon.tsx
   ```

2. Add the navigation item in `frontend/src/routes/app.tsx`:
   ```tsx
   import { Wand2 } from "lucide-react";

   // Inside the nav array:
   { to: "/app/tryon", label: "Virtual Try-On", icon: Wand2 },
   ```

3. Ensure `frontend/.env` has:
   ```env
   VITE_TRYON_API_URL=http://127.0.0.1:5001
   ```
