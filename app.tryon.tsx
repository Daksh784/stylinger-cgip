import { createFileRoute } from "@tanstack/react-router";
import { useState, useRef, useId } from "react";
import { PageHeader } from "@/components/site/dash-shell";
import { useProducts } from "@/lib/useProducts";
import { wardrobeItems } from "@/lib/mock-data";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import {
  Upload,
  Sparkles,
  Shirt,
  Download,
  RotateCcw,
  Sliders,
  Check,
  AlertCircle,
  Eye,
  Columns,
  Layers,
  Info,
  Loader2,
  Trash2,
  ArrowRight,
  Maximize2,
  RefreshCw,
  User,
  Search,
} from "lucide-react";
import { toast } from "sonner";

export const Route = createFileRoute("/app/tryon")({
  head: () => ({ meta: [{ title: "Virtual Fashion Try-On — Stylinger" }] }),
  component: VirtualTryOnPage,
});

export type GenderFilter = "men" | "women" | "unisex";

interface GarmentSelection {
  id: string;
  name: string;
  category: string;
  color?: string;
  price?: number;
  image: string;
  gender?: "men" | "women" | "unisex";
  source: "catalog" | "wardrobe" | "custom";
}

export interface SizingProfile {
  recommended_size?: string | null;
  confidence_score?: number;
  fit_type: string;
  shoulder_width_px: number;
  torso_height_px: number;
  head_ratio: number;
  is_estimate?: boolean;
  size_match_supported?: boolean;
  estimation_method?: string;
}

interface TryOnResult {
  id: string;
  garmentName: string;
  garmentCategory: string;
  garmentImage: string;
  resultImage: string;
  originalImage: string;
  timestamp: string;
  telemetry?: {
    total_time_ms: number;
    segmentation_ms?: number;
    transform_ms?: number;
    composite_ms?: number;
    preprocess_ms?: number;
    decode_ms?: number;
  };
  torsoBbox?: { x: number; y: number; width: number; height: number };
  headBbox?: { x: number; y: number; width: number; height: number };
  neckPoint?: { x: number; y: number };
  leftShoulder?: { x: number; y: number };
  rightShoulder?: { x: number; y: number };
  lowerBodyBbox?: { x: number; y: number; width: number; height: number };
  hasLowerBody?: boolean;
  warnings?: string[];
  resultWidth?: number;
  resultHeight?: number;
  sizingProfile?: SizingProfile;
  assetValidation?: {
    valid: boolean;
    asset_type: string;
    message: string;
    warning?: string;
  };
}

const CATEGORY_FILTERS = ["All", "Outerwear", "Shirts", "T-Shirts", "Dresses", "Jeans", "Trousers", "Knitwear"];

const DEFAULT_SETTINGS = {
  scale: 1.0,
  offsetX: 0,
  offsetY: 0,
  featherRadius: 3,
  opacity: 1.0,
  matchLighting: true,
  enhanceContrast: true,
};

const CUSTOM_GARMENT_CATEGORIES = ["T-Shirts", "Shirts", "Outerwear", "Jeans", "Trousers", "Dresses", "Knitwear"];

function VirtualTryOnPage() {
  // Step & Data states
  const [userPhoto, setUserPhoto] = useState<string | null>(null);
  const [userPhotoFile, setUserPhotoFile] = useState<File | null>(null);
  const [photoDimensions, setPhotoDimensions] = useState<{ width: number; height: number } | null>(null);

  // Gender Selection & Filtering
  const [genderFilter, setGenderFilter] = useState<GenderFilter>("unisex");
  const [searchQuery, setSearchQuery] = useState("");

  // Automatic Sizing & Body Analysis
  const [sizingProfile, setSizingProfile] = useState<SizingProfile | null>(null);
  const [isAnalyzingPhoto, setIsAnalyzingPhoto] = useState(false);

  const [selectedGarment, setSelectedGarment] = useState<GarmentSelection | null>(null);
  const [garmentSourceTab, setGarmentSourceTab] = useState<"catalog" | "wardrobe" | "custom">("catalog");
  const [categoryFilter, setCategoryFilter] = useState("All");

  // Fitting Adjustment parameters
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);

  // Processing state
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStatus, setProcessingStatus] = useState<string>("");
  const [currentResult, setCurrentResult] = useState<TryOnResult | null>(null);

  // Comparison & History
  const [history, setHistory] = useState<TryOnResult[]>([]);
  const [isCompareModalOpen, setIsCompareModalOpen] = useState(false);
  const [viewMode, setViewMode] = useState<"result" | "original" | "split">("split");
  const [splitPosition, setSplitPosition] = useState<number>(50);

  // Segmentation Preview Inspection modal
  const [isInspectingContour, setIsInspectingContour] = useState(false);
  const [segmentationOverlay, setSegmentationOverlay] = useState<string | null>(null);
  const [isLoadingContour, setIsLoadingContour] = useState(false);

  // File input refs
  const userFileInputRef = useRef<HTMLInputElement>(null);
  const customGarmentInputRef = useRef<HTMLInputElement>(null);
  const customGarmentId = useId();

  // Queries
  const { data: catalogProducts = [] } = useProducts();

  // Handle Photo Upload with Client Validation
  const handlePhotoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate type
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type.toLowerCase())) {
      toast.error("Invalid file format. Please upload a JPG, JPEG, or PNG image.");
      return;
    }

    // Validate size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      toast.error("File is too large. Please select a photo under 10MB.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string;
      const img = new Image();
      img.onload = () => {
        if (img.width < 100 || img.height < 100) {
          toast.error("Image resolution is too low for silhouette estimation.");
          return;
        }
        setPhotoDimensions({ width: img.width, height: img.height });
        setUserPhoto(dataUrl);
        setUserPhotoFile(file);
        setCurrentResult(null);
        setSegmentationOverlay(null);
        toast.success(`Photo uploaded (${img.width}×${img.height}px)`);
        autoAnalyzePhoto(file, dataUrl);
      };
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  };

  // Automatically analyze user photo for body sizing
  const autoAnalyzePhoto = async (file: File | null, base64: string) => {
    setIsAnalyzingPhoto(true);
    try {
      const apiUrl = import.meta.env.VITE_TRYON_API_URL || "http://127.0.0.1:5001";
      const formData = new FormData();
      if (file) {
        formData.append("user_image", file);
      } else {
        formData.append("user_image_base64", base64);
      }
      const res = await fetch(`${apiUrl}/api/tryon/segment-preview`, {
        method: "POST",
        body: formData,
      });
      if (res.ok) {
        const data = await res.json();
        if (data.sizing_profile) {
          setSizingProfile(data.sizing_profile);
          toast.success("Body silhouette analyzed (2D contour estimate)");
        }
        if (data.overlay_base64) {
          setSegmentationOverlay(data.overlay_base64);
        }
      }
    } catch {
      // Non-blocking background analysis
    } finally {
      setIsAnalyzingPhoto(false);
    }
  };

  // Handle Custom Garment Upload
  const handleCustomGarmentUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!["image/jpeg", "image/jpg", "image/png"].includes(file.type.toLowerCase())) {
      toast.error("Garment must be a JPG or PNG image.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string;
      setSelectedGarment({
        id: `custom-${Date.now()}`,
        name: file.name.replace(/\.[^/.]+$/, ""),
        category: "T-Shirts",
        image: dataUrl,
        source: "custom",
      });
      toast.success("Custom garment ready — choose a category if this is not a T-shirt");
    };
    reader.readAsDataURL(file);
  };

  // Inspect Body Segmentation Overlay
  const handleInspectSegmentation = async () => {
    if (!userPhoto) return;
    setIsLoadingContour(true);
    setIsInspectingContour(true);

    try {
      const apiUrl = import.meta.env.VITE_TRYON_API_URL || "http://127.0.0.1:5001";
      const formData = new FormData();
      if (userPhotoFile) {
        formData.append("user_image", userPhotoFile);
      } else {
        formData.append("user_image_base64", userPhoto);
      }

      const res = await fetch(`${apiUrl}/api/tryon/segment-preview`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.error || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setSegmentationOverlay(data.overlay_base64);
      toast.success("Contour & torso boundaries extracted");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Service connection failed";
      toast.error(`Segmentation inspection failed: ${message}`);
    } finally {
      setIsLoadingContour(false);
    }
  };

  // Execute Try-On Pipeline
  const handleRunTryOn = async () => {
    if (!userPhoto) {
      toast.error("Please upload a full-body photograph first.");
      return;
    }
    if (!selectedGarment) {
      toast.error("Please select a garment from the catalog or wardrobe.");
      return;
    }

    setIsProcessing(true);
    setProcessingStatus("Initializing Computer Graphics & Image Processing pipeline...");

    try {
      const apiUrl = import.meta.env.VITE_TRYON_API_URL || "http://127.0.0.1:5001";
      const formData = new FormData();

      if (userPhotoFile) {
        formData.append("user_image", userPhotoFile);
      } else {
        formData.append("user_image_base64", userPhoto);
      }

      // Garment image
      if (selectedGarment.image.startsWith("data:")) {
        formData.append("garment_image_base64", selectedGarment.image);
      } else {
        formData.append("garment_image_url", selectedGarment.image);
      }

      formData.append("category", selectedGarment.category);
      formData.append("scale", settings.scale.toString());
      formData.append("offset_x", settings.offsetX.toString());
      formData.append("offset_y", settings.offsetY.toString());
      formData.append("feather_radius", settings.featherRadius.toString());
      formData.append("opacity", settings.opacity.toString());
      formData.append("match_lighting", settings.matchLighting ? "true" : "false");

      setProcessingStatus("Segmenting body silhouette & transforming garment geometry...");

      const response = await fetch(`${apiUrl}/api/tryon/process`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.error || `Server returned error (${response.status})`);
      }

      setProcessingStatus("Applying alpha feathering & luminance balancing...");

      const data = await response.json();
      if (!data.success || !data.image_base64) {
        throw new Error(data.error || "Image processing returned empty result.");
      }

      if (data.sizing_profile) {
        setSizingProfile(data.sizing_profile);
      }

      const newResult: TryOnResult = {
        id: `tryon-${Date.now()}`,
        garmentName: selectedGarment.name,
        garmentCategory: selectedGarment.category,
        garmentImage: selectedGarment.image,
        resultImage: data.image_base64,
        originalImage: data.original_base64 || userPhoto,
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        telemetry: data.telemetry,
        torsoBbox: data.torso_bbox,
        headBbox: data.head_bbox,
        neckPoint: data.neck_point,
        leftShoulder: data.left_shoulder,
        rightShoulder: data.right_shoulder,
        lowerBodyBbox: data.lower_body_bbox,
        hasLowerBody: data.has_lower_body,
        assetValidation: data.asset_validation,
        warnings: data.warnings,
        resultWidth: data.dimensions?.width,
        resultHeight: data.dimensions?.height,
        sizingProfile: data.sizing_profile || sizingProfile,
      };

      if (data.preview_reliable === false) {
        toast.warning("Preview is approximate. Prefer a garment-only PNG and a front-facing photo, or use the alignment sliders.");
      } else if (data.warnings && data.warnings.length > 0) {
        toast.warning(data.warnings[0]);
      }

      setCurrentResult(newResult);
      setHistory((prev) => [newResult, ...prev]);
      setViewMode("split");
      toast.success(`Virtual Try-On generated in ${data.telemetry?.total_time_ms || 0}ms`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Service unreachable";
      toast.error(`Virtual Try-On failed: ${msg}. Verify that the Python service is running on port 5001.`);
    } finally {
      setIsProcessing(false);
      setProcessingStatus("");
    }
  };

  // Switch gender collection and auto-deselect incompatible garments
  const handleGenderChange = (newGender: GenderFilter) => {
    setGenderFilter(newGender);
    if (selectedGarment && selectedGarment.source !== "custom") {
      const g = selectedGarment.gender;
      const isCompatible =
        newGender === "men" ? (g === "men" || g === "unisex") :
        newGender === "women" ? (g === "women" || g === "unisex") :
        (g === "unisex");

      if (!isCompatible) {
        setSelectedGarment(null);
        toast.info(
          `Deselected "${selectedGarment.name}" because it is not part of the ${newGender === "men" ? "Men's" : newGender === "women" ? "Women's" : "Unisex"} collection.`
        );
      }
    }
  };

  // Download Output
  const handleDownloadResult = () => {
    if (!currentResult) return;
    const link = document.createElement("a");
    link.href = currentResult.resultImage;
    const safeName = currentResult.garmentName.toLowerCase().replace(/[^a-z0-9]/g, "-");
    link.download = `stylinger-tryon-${safeName}-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast.success("Outfit preview downloaded!");
  };

  // Filter Catalog & Wardrobe items according to gender, category, and search query
  const filteredCatalog = catalogProducts.filter((p) => {
    // 1. Gender filter:
    // Men: men + unisex
    // Women: women + unisex
    // Unisex: unisex only
    // Unknown or missing gender: do not automatically classify as men or women
    const matchesGender =
      genderFilter === "men" ? (p.gender === "men" || p.gender === "unisex") :
      genderFilter === "women" ? (p.gender === "women" || p.gender === "unisex") :
      (p.gender === "unisex");

    if (!matchesGender) return false;

    // 2. Category filter
    if (categoryFilter !== "All" && p.category.toLowerCase() !== categoryFilter.toLowerCase()) {
      return false;
    }

    // 3. Search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchesName = p.name.toLowerCase().includes(q);
      const matchesDesc = p.description.toLowerCase().includes(q);
      const matchesColor = p.color.toLowerCase().includes(q);
      if (!matchesName && !matchesDesc && !matchesColor) return false;
    }

    return true;
  });

  const filteredWardrobe = wardrobeItems.filter((w) => {
    const matchesGender =
      genderFilter === "men" ? (w.gender === "men" || w.gender === "unisex") :
      genderFilter === "women" ? (w.gender === "women" || w.gender === "unisex") :
      (w.gender === "unisex");

    if (!matchesGender) return false;

    if (categoryFilter !== "All" && w.category.toLowerCase() !== categoryFilter.toLowerCase()) {
      return false;
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchesName = w.name.toLowerCase().includes(q);
      const matchesColor = w.color.toLowerCase().includes(q);
      const matchesTag = w.tag.toLowerCase().includes(q);
      if (!matchesName && !matchesColor && !matchesTag) return false;
    }

    return true;
  });

  return (
    <>
      <PageHeader
        eyebrow="2D Computer Graphics Fitting"
        title="Virtual Fashion Try-On"
        description="Classical image-processing outfit projection with contour extraction, affine alignment, and alpha compositing."
        action={
          <div className="flex flex-wrap items-center gap-2">
            {history.length > 1 && (
              <Button
                variant="ghost"
                onClick={() => setIsCompareModalOpen(true)}
                className="rounded-full border border-hairline text-xs"
              >
                <Columns className="mr-2 h-3.5 w-3.5 text-emerald" />
                Compare Looks ({history.length})
              </Button>
            )}
            {currentResult && (
              <Button
                onClick={handleDownloadResult}
                className="rounded-full bg-emerald text-emerald-foreground hover:bg-emerald-glow text-xs"
              >
                <Download className="mr-2 h-3.5 w-3.5" />
                Download Outfit
              </Button>
            )}
          </div>
        }
      />

      {/* Limitations & Technical Disclaimer */}
      <div className="mb-6 rounded-2xl border border-hairline bg-surface/40 p-4 text-xs text-muted-foreground backdrop-blur">
        <div className="flex items-start gap-3">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-emerald" />
          <div className="space-y-1">
            <p className="font-medium text-foreground">Computer Graphics & Image Processing Methodology</p>
            <p className="leading-relaxed">
              This preview uses classical Computer Vision (Gaussian filtering, CLAHE contrast balancing, GrabCut/Otsu background segmentation, and affine geometric warping with feathered alpha compositing).
              It provides a 2D clothing visualization rather than a 3D fabric physics simulation. Use the interactive placement sliders below to fine-tune scale and torso alignment for your silhouette.
            </p>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-12">
        {/* LEFT COLUMN: Controls & Input Selection (5 Cols) */}
        <div className="space-y-6 lg:col-span-5">
          {/* STEP 1: Upload Photo Card */}
          <Card className="glass border-hairline rounded-2xl overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald/20 text-emerald text-xs font-bold">1</div>
                  <CardTitle className="text-base font-medium">Your Photo</CardTitle>
                </div>
                {userPhoto && (
                  <Badge variant="outline" className="text-[11px] border-hairline text-muted-foreground">
                    {photoDimensions?.width} × {photoDimensions?.height} px
                  </Badge>
                )}
              </div>
              <CardDescription className="text-xs">
                Front-facing, full-body portrait on a plain background for optimal contour segmentation.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <input
                ref={userFileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                className="hidden"
                onChange={handlePhotoUpload}
              />

              {!userPhoto ? (
                <div
                  onClick={() => userFileInputRef.current?.click()}
                  className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-hairline p-8 text-center transition-colors hover:border-emerald/50 hover:bg-white/[0.02] cursor-pointer"
                >
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white/5 text-muted-foreground">
                    <Upload className="h-5 w-5 text-emerald" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Click to upload photo</p>
                    <p className="text-xs text-muted-foreground mt-0.5">JPG, JPEG, PNG up to 10MB</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="relative aspect-[3/4] max-h-64 overflow-hidden rounded-xl border border-hairline bg-surface-2">
                    <img src={userPhoto} alt="Uploaded silhouette" className="h-full w-full object-contain" />
                    <div className="absolute bottom-2 right-2 flex gap-1.5">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={handleInspectSegmentation}
                        className="h-7 rounded-full text-[11px] px-2.5 bg-black/70 backdrop-blur text-white hover:bg-black"
                      >
                        <Eye className="mr-1.5 h-3 w-3 text-emerald" />
                        Inspect Mask
                      </Button>
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => {
                          setUserPhoto(null);
                          setUserPhotoFile(null);
                          setCurrentResult(null);
                        }}
                        className="h-7 w-7 rounded-full p-0 bg-black/70 backdrop-blur hover:bg-destructive"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => userFileInputRef.current?.click()}
                    className="w-full rounded-full border-hairline text-xs"
                  >
                    <RefreshCw className="mr-2 h-3.5 w-3.5" /> Change Photo
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* STEP 2: Choose Garment Card */}
          <Card className="glass border-hairline rounded-2xl overflow-hidden">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald/20 text-emerald text-xs font-bold">2</div>
                  <CardTitle className="text-base font-medium">Select Garment</CardTitle>
                </div>
                {selectedGarment && (
                  <Badge className="bg-emerald/15 text-emerald border-emerald/30 text-[11px]">
                    {selectedGarment.category}
                  </Badge>
                )}
              </div>
              <CardDescription className="text-xs">
                Pick a garment from the luxury catalog, your wardrobe, or upload a custom piece.
              </CardDescription>

              {/* Gender Selection (Inclusive, User-Controlled) */}
              <div className="mt-3 space-y-1.5">
                <div className="flex items-center justify-between text-[11px] font-medium text-foreground">
                  <span className="flex items-center gap-1.5 text-muted-foreground">
                    <User className="h-3.5 w-3.5 text-emerald" />
                    Target Collection:
                  </span>
                  <span className="text-[10px] text-muted-foreground/80">User-selected</span>
                </div>
                <div className="grid grid-cols-3 gap-1 rounded-xl border border-hairline p-1 bg-surface-2/40 text-xs">
                  {(["men", "women", "unisex"] as const).map((g) => (
                    <button
                      key={g}
                      type="button"
                      onClick={() => handleGenderChange(g)}
                      className={`rounded-lg py-1.5 text-xs font-medium capitalize transition-all ${
                        genderFilter === g
                          ? "bg-emerald text-emerald-foreground shadow"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {g === "men" ? "Men" : g === "women" ? "Women" : "Unisex"}
                    </button>
                  ))}
                </div>
              </div>

              {/* Source Tabs */}
              <div className="mt-2 flex rounded-lg border border-hairline p-1 bg-surface-2/40 text-xs">
                <button
                  type="button"
                  onClick={() => setGarmentSourceTab("catalog")}
                  className={`flex-1 rounded-md py-1 font-medium transition-all ${
                    garmentSourceTab === "catalog" ? "bg-emerald text-emerald-foreground shadow" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Catalog
                </button>
                <button
                  type="button"
                  onClick={() => setGarmentSourceTab("wardrobe")}
                  className={`flex-1 rounded-md py-1 font-medium transition-all ${
                    garmentSourceTab === "wardrobe" ? "bg-emerald text-emerald-foreground shadow" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Wardrobe
                </button>
                <button
                  type="button"
                  onClick={() => setGarmentSourceTab("custom")}
                  className={`flex-1 rounded-md py-1 font-medium transition-all ${
                    garmentSourceTab === "custom" ? "bg-emerald text-emerald-foreground shadow" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  Custom
                </button>
              </div>

              {/* Search Bar */}
              {garmentSourceTab !== "custom" && (
                <div className="relative mt-2">
                  <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder={`Search ${genderFilter === "men" ? "Men's & Unisex" : genderFilter === "women" ? "Women's & Unisex" : "Unisex"} ${garmentSourceTab}...`}
                    className="w-full rounded-xl border border-hairline bg-surface/50 pl-8 pr-7 py-1.5 text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:border-emerald"
                  />
                  {searchQuery && (
                    <button
                      type="button"
                      onClick={() => setSearchQuery("")}
                      className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground text-xs"
                    >
                      ×
                    </button>
                  )}
                </div>
              )}

              {/* Category Pills */}
              {garmentSourceTab !== "custom" && (
                <div className="flex gap-1.5 overflow-x-auto pt-2 scrollbar-none">
                  {CATEGORY_FILTERS.map((cat) => {
                    const hasItems =
                      cat === "All" ||
                      (garmentSourceTab === "catalog"
                        ? catalogProducts.some(
                            (p) =>
                              p.category.toLowerCase() === cat.toLowerCase() &&
                              (genderFilter === "men" ? (p.gender === "men" || p.gender === "unisex") :
                               genderFilter === "women" ? (p.gender === "women" || p.gender === "unisex") :
                               (p.gender === "unisex"))
                          )
                        : wardrobeItems.some(
                            (w) =>
                              w.category.toLowerCase() === cat.toLowerCase() &&
                              (genderFilter === "men" ? (w.gender === "men" || w.gender === "unisex") :
                               genderFilter === "women" ? (w.gender === "women" || w.gender === "unisex") :
                               (w.gender === "unisex"))
                          ));

                    if (!hasItems && cat !== "All") return null;

                    return (
                      <button
                        key={cat}
                        type="button"
                        onClick={() => setCategoryFilter(cat)}
                        className={`whitespace-nowrap rounded-full border px-2.5 py-0.5 text-[11px] transition-colors ${
                          categoryFilter === cat
                            ? "border-emerald bg-emerald/10 text-emerald"
                            : "border-hairline text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {cat}
                      </button>
                    );
                  })}
                </div>
              )}
            </CardHeader>

            <CardContent>
              {/* CATALOG GRID */}
              {garmentSourceTab === "catalog" && (
                <>
                  {filteredCatalog.length === 0 ? (
                    <div className="py-8 text-center text-xs text-muted-foreground">
                      No {genderFilter} garments match the selected filters.
                    </div>
                  ) : (
                    <div className="grid grid-cols-3 gap-2.5 max-h-64 overflow-y-auto pr-1">
                      {filteredCatalog.map((item) => {
                        const isSelected = selectedGarment?.id === item.id;
                        return (
                          <div
                            key={item.id}
                            onClick={() =>
                              setSelectedGarment({
                                id: item.id,
                                name: item.name,
                                category: item.category,
                                price: item.price,
                                color: item.color,
                                image: item.image,
                                gender: item.gender,
                                source: "catalog",
                              })
                            }
                            className={`group relative aspect-[3/4] cursor-pointer overflow-hidden rounded-xl border transition-all ${
                              isSelected
                                ? "border-emerald ring-2 ring-emerald/40"
                                : "border-hairline hover:border-white/30"
                            }`}
                          >
                            <img src={item.image} alt={item.name} className="h-full w-full object-cover transition-transform group-hover:scale-105" />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent opacity-90" />
                            {item.gender && (
                              <div className="absolute top-1.5 left-1.5">
                                <span className="rounded bg-black/60 backdrop-blur px-1 py-0.5 text-[8px] uppercase font-semibold text-white/80 border border-white/10">
                                  {item.gender}
                                </span>
                              </div>
                            )}
                            {isSelected && (
                              <div className="absolute top-1.5 right-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-emerald text-emerald-foreground">
                                <Check className="h-2.5 w-2.5" />
                              </div>
                            )}
                            <div className="absolute bottom-1.5 left-1.5 right-1.5">
                              <p className="truncate text-[10px] font-medium text-white">{item.name}</p>
                              <p className="text-[9px] text-emerald">₹{item.price.toLocaleString("en-IN")}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </>
              )}

              {/* WARDROBE GRID */}
              {garmentSourceTab === "wardrobe" && (
                <>
                  {filteredWardrobe.length === 0 ? (
                    <div className="py-8 text-center text-xs text-muted-foreground">
                      No {genderFilter} items found in your wardrobe.
                    </div>
                  ) : (
                    <div className="grid grid-cols-3 gap-2.5 max-h-64 overflow-y-auto pr-1">
                      {filteredWardrobe.map((item) => {
                        const isSelected = selectedGarment?.id === item.id;
                        return (
                          <div
                            key={item.id}
                            onClick={() =>
                              setSelectedGarment({
                                id: item.id,
                                name: item.name,
                                category: item.category,
                                color: item.color,
                                image:
                                  catalogProducts.find((p) => p.category.toLowerCase() === item.category.toLowerCase())?.image ||
                                  "https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=500&auto=format&fit=crop&q=60",
                                gender: item.gender,
                                source: "wardrobe",
                              })
                            }
                            className={`group relative aspect-[3/4] cursor-pointer overflow-hidden rounded-xl border transition-all ${
                              isSelected
                                ? "border-emerald ring-2 ring-emerald/40"
                                : "border-hairline hover:border-white/30"
                            }`}
                          >
                            <div className="absolute inset-0 bg-gradient-to-br from-emerald/20 via-surface-2 to-surface" />
                            <div className="absolute inset-0 flex flex-col justify-between p-2">
                              <div className="flex justify-between items-start">
                                <Badge variant="outline" className="text-[9px] px-1 py-0 border-white/20">
                                  {item.tag}
                                </Badge>
                                {isSelected && (
                                  <div className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald text-emerald-foreground">
                                    <Check className="h-2.5 w-2.5" />
                                  </div>
                                )}
                              </div>
                              <div>
                                <p className="truncate text-[10px] font-medium text-foreground">{item.name}</p>
                                <p className="text-[9px] text-muted-foreground">{item.color} · {item.gender || "unisex"}</p>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </>
              )}

              {/* CUSTOM GARMENT UPLOAD */}
              {garmentSourceTab === "custom" && (
                <div className="space-y-3">
                  <input
                    ref={customGarmentInputRef}
                    id={customGarmentId}
                    type="file"
                    accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                    className="hidden"
                    onChange={handleCustomGarmentUpload}
                  />
                  <div
                    onClick={() => customGarmentInputRef.current?.click()}
                    className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-hairline p-6 text-center transition-colors hover:border-emerald/50 cursor-pointer"
                  >
                    <Shirt className="h-6 w-6 text-emerald" />
                    <div>
                      <p className="text-xs font-medium">Upload flat-lay clothing image</p>
                      <p className="text-[10px] text-muted-foreground">Transparent PNG of the garment only, or a white flat-lay</p>
                    </div>
                  </div>
                  {selectedGarment?.source === "custom" && (
                    <div className="space-y-2">
                      <div className="flex items-center gap-3 rounded-xl border border-emerald/30 bg-emerald/5 p-2">
                        <img src={selectedGarment.image} alt="custom" className="h-10 w-10 rounded-lg object-cover" />
                        <div className="flex-1 min-w-0">
                          <p className="truncate text-xs font-medium">{selectedGarment.name}</p>
                          <p className="text-[10px] text-emerald">Custom Upload</p>
                        </div>
                      </div>
                      <label className="block text-[10px] text-muted-foreground">
                        Garment category
                        <select
                          className="mt-1 w-full rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-xs text-foreground"
                          value={selectedGarment.category}
                          onChange={(e) =>
                            setSelectedGarment((g) => (g ? { ...g, category: e.target.value } : g))
                          }
                        >
                          {CUSTOM_GARMENT_CATEGORIES.map((c) => (
                            <option key={c} value={c}>
                              {c}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* STEP 3: Automated Sizing & Fit Intelligence */}
          <Card className="glass border-hairline rounded-2xl overflow-hidden">
            <CardHeader className="pb-3 border-b border-hairline">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-emerald" />
                  <CardTitle className="text-sm font-medium">Estimated Sizing & Body Analysis</CardTitle>
                </div>
                {sizingProfile && (
                  <Badge className="bg-emerald/20 text-emerald border-emerald/40 text-[11px] font-mono">
                    {sizingProfile.shoulder_width_px}px shoulder span
                  </Badge>
                )}
              </div>
              <CardDescription className="text-xs">
                2D anthropometric estimation based on detected shoulder breadth, head-to-torso proportions, and collar anchors.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 pt-4 text-xs">
              {isAnalyzingPhoto ? (
                <div className="flex items-center justify-center p-4 rounded-xl border border-hairline bg-surface-2 gap-3 text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin text-emerald" />
                  <span>Analyzing body proportions & shoulder span...</span>
                </div>
              ) : sizingProfile ? (
                <div className="rounded-xl border border-emerald/30 bg-emerald/5 p-3.5 space-y-2.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald text-emerald-foreground text-xs font-bold">
                        ✓
                      </span>
                      <div>
                        <p className="font-semibold text-foreground text-xs">2D body envelope detected</p>
                        <p className="text-[10px] text-muted-foreground">{sizingProfile.fit_type}</p>
                      </div>
                    </div>
                    <Badge variant="outline" className="border-emerald/40 text-emerald text-[10px]">
                      2D Geometric Estimate
                    </Badge>
                  </div>

                  <div className="grid grid-cols-3 gap-2 pt-2 border-t border-emerald/20 text-[10px]">
                    <div>
                      <span className="text-muted-foreground block text-[9px] uppercase tracking-wider">Shoulders</span>
                      <span className="font-medium text-foreground">Auto-Calibrated</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block text-[9px] uppercase tracking-wider">Neckline</span>
                      <span className="font-medium text-foreground">Collar-Snapped</span>
                    </div>
                    <div>
                      <span className="text-muted-foreground block text-[9px] uppercase tracking-wider">Drape</span>
                      <span className="font-medium text-foreground">Proportional</span>
                    </div>
                  </div>

                  <p className="text-[10px] text-muted-foreground/80 italic pt-1 border-t border-hairline/40">
                    * Estimated using 2D contour geometry. Not a physical tailor measurement or 3D fabric physics simulation.
                  </p>
                </div>
              ) : userPhoto ? (
                <div className="rounded-xl border border-hairline bg-surface-2 p-3 text-center space-y-1">
                  <p className="text-[11px] font-medium text-foreground">Automatic Sizing Ready</p>
                  <p className="text-[10px] text-muted-foreground">Select a garment from Step 2 to generate an auto-scaled fit.</p>
                </div>
              ) : (
                <div className="rounded-xl border border-hairline bg-surface-2 p-3 text-center text-muted-foreground">
                  <p className="text-[11px]">Upload your photograph above to automatically detect your measurements and clothing size.</p>
                </div>
              )}

              {/* Primary Action Button */}
              <Button
                onClick={handleRunTryOn}
                disabled={isProcessing || !userPhoto || !selectedGarment}
                className="w-full rounded-full bg-emerald text-emerald-foreground hover:bg-emerald-glow font-medium text-xs py-5 shadow-emerald"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Calculating & Fitting Outfit...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-2 h-4 w-4" />
                    {sizingProfile ? "Run 2D Virtual Try-On" : "Run Smart Auto-Fit Try-On"}
                  </>
                )}
              </Button>

              {/* Optional Collapsed Fine-Tuning */}
              <details className="group pt-1">
                <summary className="flex cursor-pointer items-center justify-between text-[11px] text-muted-foreground hover:text-foreground select-none">
                  <span>Advanced Manual Adjustments (Optional)</span>
                  <span className="text-[10px] transition-transform group-open:rotate-180">▼</span>
                </summary>
                <div className="mt-3 space-y-3 pt-2 border-t border-hairline text-xs">
                  {/* Fitting Presets */}
                  <div>
                    <p className="mb-1.5 font-medium text-foreground text-[11px]">Fit Presets</p>
                    <div className="grid grid-cols-4 gap-1">
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, scale: 1.0, offsetY: -35, offsetX: 0 }))}
                        className="h-7 px-1 text-[10px] border-hairline hover:border-emerald hover:text-emerald"
                      >
                        Shoulders
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, scale: 1.0, offsetY: 0, offsetX: 0 }))}
                        className="h-7 px-1 text-[10px] border-hairline hover:border-emerald hover:text-emerald"
                      >
                        Center
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, scale: 1.15, offsetY: -20, offsetX: 0 }))}
                        className="h-7 px-1 text-[10px] border-hairline hover:border-emerald hover:text-emerald"
                      >
                        Oversized
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, scale: 1.0, offsetY: 110, offsetX: 0 }))}
                        className="h-7 px-1 text-[10px] border-hairline hover:border-emerald hover:text-emerald"
                      >
                        Waist
                      </Button>
                    </div>
                  </div>

                  {/* Directional Nudge Pad */}
                  <div className="flex items-center justify-between border-y border-hairline py-2">
                    <span className="text-muted-foreground text-[11px]">Fine Nudge (±10px)</span>
                    <div className="flex items-center gap-1">
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, offsetX: s.offsetX - 10 }))}
                        className="h-6 w-6 p-0 border-hairline text-[11px]"
                        title="Nudge Left"
                      >
                        ←
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, offsetY: s.offsetY - 10 }))}
                        className="h-6 w-6 p-0 border-hairline text-[11px]"
                        title="Nudge Up"
                      >
                        ↑
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, offsetY: s.offsetY + 10 }))}
                        className="h-6 w-6 p-0 border-hairline text-[11px]"
                        title="Nudge Down"
                      >
                        ↓
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        type="button"
                        onClick={() => setSettings((s) => ({ ...s, offsetX: s.offsetX + 10 }))}
                        className="h-6 w-6 p-0 border-hairline text-[11px]"
                        title="Nudge Right"
                      >
                        →
                      </Button>
                    </div>
                  </div>

                  {/* Garment Scale */}
                  <div>
                    <div className="mb-1 flex justify-between text-muted-foreground">
                      <span>Scale</span>
                      <span className="font-mono text-foreground">{Math.round(settings.scale * 100)}%</span>
                    </div>
                    <Slider
                      min={0.5}
                      max={2.2}
                      step={0.02}
                      value={[settings.scale]}
                      onValueChange={([val]) => setSettings((s) => ({ ...s, scale: val }))}
                    />
                  </div>

                  {/* Vertical Offset */}
                  <div>
                    <div className="mb-1 flex justify-between text-muted-foreground">
                      <span>Vertical Offset</span>
                      <span className="font-mono text-foreground">{settings.offsetY} px</span>
                    </div>
                    <Slider
                      min={-250}
                      max={250}
                      step={2}
                      value={[settings.offsetY]}
                      onValueChange={([val]) => setSettings((s) => ({ ...s, offsetY: val }))}
                    />
                  </div>

                  {/* Edge Feathering */}
                  <div>
                    <div className="mb-1 flex justify-between text-muted-foreground">
                      <span>Feather Radius</span>
                      <span className="font-mono text-foreground">{settings.featherRadius} px</span>
                    </div>
                    <Slider
                      min={0}
                      max={15}
                      step={1}
                      value={[settings.featherRadius]}
                      onValueChange={([val]) => setSettings((s) => ({ ...s, featherRadius: val }))}
                    />
                  </div>

                  {currentResult && (
                    <Button
                      type="button"
                      size="sm"
                      onClick={handleRunTryOn}
                      disabled={isProcessing}
                      className="w-full mt-2 rounded-xl bg-surface-2 border border-emerald/40 text-emerald hover:bg-emerald/10 text-xs py-2 font-medium"
                    >
                      <RotateCcw className="mr-1.5 h-3.5 w-3.5" />
                      Re-fit Outfit with Current Adjustments
                    </Button>
                  )}
                </div>
              </details>
            </CardContent>
          </Card>
        </div>

        {/* RIGHT COLUMN: Interactive Preview & Comparison (7 Cols) */}
        <div className="space-y-6 lg:col-span-7">
          <Card className="glass border-hairline rounded-2xl overflow-hidden min-h-[580px] flex flex-col">
            <CardHeader className="pb-3 border-b border-hairline">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <CardTitle className="text-base font-medium">Virtual Outfit Preview</CardTitle>
                  <CardDescription className="text-xs">
                    {currentResult ? `Fitted with ${currentResult.garmentName}` : "Awaiting processing"}
                  </CardDescription>
                </div>

                {/* View Mode Selectors */}
                {currentResult && (
                  <div className="flex rounded-lg border border-hairline p-0.5 bg-surface-2 text-xs">
                    <button
                      type="button"
                      onClick={() => setViewMode("split")}
                      className={`px-3 py-1 rounded-md font-medium transition-colors ${
                        viewMode === "split" ? "bg-emerald text-emerald-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Split Slider
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewMode("result")}
                      className={`px-3 py-1 rounded-md font-medium transition-colors ${
                        viewMode === "result" ? "bg-emerald text-emerald-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Outfit
                    </button>
                    <button
                      type="button"
                      onClick={() => setViewMode("original")}
                      className={`px-3 py-1 rounded-md font-medium transition-colors ${
                        viewMode === "original" ? "bg-emerald text-emerald-foreground" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Original
                    </button>
                  </div>
                )}
              </div>
            </CardHeader>

            <CardContent className="flex-1 flex flex-col justify-center items-center p-6">
              {/* Empty State */}
              {!currentResult && !isProcessing && (
                <div className="flex flex-col items-center justify-center text-center max-w-sm py-16 space-y-4">
                  <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-hairline bg-surface-2 text-muted-foreground">
                    <Sparkles className="h-8 w-8 text-emerald" />
                  </div>
                  <div>
                    <h3 className="text-base font-medium">No Outfit Generated Yet</h3>
                    <p className="text-xs text-muted-foreground mt-1 leading-relaxed">
                      Upload your photo on the left, pick a piece from the catalog or your wardrobe, and click "Process Virtual Try-On".
                    </p>
                  </div>
                  <div className="grid grid-cols-3 gap-2 w-full pt-4 border-t border-hairline text-[11px] text-muted-foreground">
                    <div className="space-y-1">
                      <div className="font-semibold text-foreground">1. Upload</div>
                      <div>Full-body shot</div>
                    </div>
                    <div className="space-y-1">
                      <div className="font-semibold text-foreground">2. Select</div>
                      <div>Tops, coat, etc.</div>
                    </div>
                    <div className="space-y-1">
                      <div className="font-semibold text-foreground">3. Compose</div>
                      <div>Contour blend</div>
                    </div>
                  </div>
                </div>
              )}

              {/* Loading State */}
              {isProcessing && (
                <div className="flex flex-col items-center justify-center text-center py-20 space-y-4">
                  <div className="relative">
                    <div className="h-16 w-16 rounded-full border-2 border-emerald/20 border-t-emerald animate-spin" />
                    <Sparkles className="h-6 w-6 text-emerald absolute inset-0 m-auto" />
                  </div>
                  <div className="space-y-1">
                    <p className="text-sm font-medium text-foreground">{processingStatus}</p>
                    <p className="text-xs text-muted-foreground">Applying OpenCV pipeline (in-memory execution)</p>
                  </div>
                </div>
              )}

              {/* Rendered Preview Screen */}
              {currentResult && !isProcessing && (
                <div className="w-full flex flex-col items-center space-y-4">
                  {/* Viewport Box */}
                  <div
                    className="relative w-full max-w-md overflow-hidden rounded-2xl border border-hairline bg-surface-2 shadow-luxury select-none"
                    style={{
                      aspectRatio:
                        currentResult.resultWidth && currentResult.resultHeight
                          ? `${currentResult.resultWidth} / ${currentResult.resultHeight}`
                          : "3 / 4",
                    }}
                  >
                    {/* View mode: Single Image */}
                    {viewMode === "result" && (
                      <img src={currentResult.resultImage} alt="Virtual Try-On" className="h-full w-full object-contain" />
                    )}
                    {viewMode === "original" && (
                      <img src={currentResult.originalImage} alt="Original User" className="h-full w-full object-contain" />
                    )}

                    {/* View mode: Split Comparison Slider */}
                    {viewMode === "split" && (
                      <div className="relative h-full w-full overflow-hidden">
                        {/* Background: Result Image */}
                        <img src={currentResult.resultImage} alt="Fitted Outfit" className="absolute inset-0 h-full w-full object-contain pointer-events-none" />

                        {/* Foreground: Original Image clipped by split percentage */}
                        <div
                          className="absolute inset-0 overflow-hidden pointer-events-none"
                          style={{ clipPath: `inset(0 ${100 - splitPosition}% 0 0)` }}
                        >
                          <img src={currentResult.originalImage} alt="Original" className="h-full w-full object-contain" />
                        </div>

                        {/* Dividing line */}
                        <div
                          className="absolute top-0 bottom-0 w-0.5 bg-emerald shadow-[0_0_10px_rgba(16,185,129,0.8)] z-10 pointer-events-none"
                          style={{ left: `${splitPosition}%` }}
                        >
                          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 flex h-6 w-6 items-center justify-center rounded-full bg-emerald text-emerald-foreground shadow">
                            <Columns className="h-3 w-3" />
                          </div>
                        </div>

                        {/* Split tags */}
                        <div className="absolute top-3 left-3 rounded-full bg-black/60 px-2.5 py-0.5 text-[10px] uppercase font-bold text-white backdrop-blur">
                          Original
                        </div>
                        <div className="absolute top-3 right-3 flex items-center gap-1.5 rounded-full bg-emerald/90 px-2.5 py-0.5 text-[10px] font-bold text-white backdrop-blur shadow">
                          <span>Try-On</span>
                          {currentResult.telemetry?.total_time_ms != null && (
                            <span className="border-l border-white/40 pl-1.5 text-white/90">
                              {currentResult.telemetry.total_time_ms}ms
                            </span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Split Controller Range slider */}
                  {viewMode === "split" && (
                    <div className="w-full max-w-md flex items-center gap-3 text-xs text-muted-foreground pt-1">
                      <span>Original</span>
                      <Slider
                        min={0}
                        max={100}
                        step={1}
                        value={[splitPosition]}
                        onValueChange={([v]) => setSplitPosition(v)}
                        className="flex-1"
                      />
                      <span>Try-On</span>
                    </div>
                  )}

                  {/* Warnings Banner if lower body or garment has notes */}
                  {currentResult.warnings && currentResult.warnings.length > 0 && (
                    <div className="w-full max-w-md rounded-xl border border-amber-500/30 bg-amber-500/10 p-2.5 text-xs text-amber-200 text-left flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 shrink-0 mt-0.5 text-amber-400" />
                      <span>{currentResult.warnings[0]}</span>
                    </div>
                  )}

                  {/* Telemetry & Sizing pills */}
                  <div className="flex flex-wrap items-center justify-center gap-2 text-[11px] text-muted-foreground pt-2">
                    {currentResult.sizingProfile && (
                      <span
                        className="rounded-full bg-emerald/10 border border-emerald/30 text-emerald px-2.5 py-0.5 font-medium flex items-center gap-1.5"
                        title="Contour geometry only. Not a clothing size match."
                      >
                        <Sparkles className="h-3 w-3" />
                        2D overlay · shoulders {currentResult.sizingProfile.shoulder_width_px}px
                      </span>
                    )}
                    {currentResult.assetValidation && (
                      <span
                        className={`rounded-full border px-2.5 py-0.5 font-medium flex items-center gap-1 ${
                          currentResult.assetValidation.asset_type === "model_detected"
                            ? "bg-amber-500/10 border-amber-500/30 text-amber-300"
                            : currentResult.assetValidation.asset_type === "transparent_png"
                            ? "bg-emerald/10 border-emerald/30 text-emerald"
                            : "bg-surface-2 border-hairline text-foreground"
                        }`}
                        title={currentResult.assetValidation.message}
                      >
                        {currentResult.assetValidation.asset_type === "model_detected" ? "⚠️ Model Auto-Suppression" :
                         currentResult.assetValidation.asset_type === "transparent_png" ? "🛡️ Clean Transparent Asset" :
                         currentResult.assetValidation.asset_type === "complex_background" ? "ℹ️ Studio Extracted" :
                         "✨ Studio Flat-Lay"}
                      </span>
                    )}
                    {currentResult.telemetry && (
                      <span className="rounded-full bg-surface-2 border border-hairline px-2.5 py-0.5">
                        ⏱️ Total: <strong className="text-foreground">{currentResult.telemetry.total_time_ms}ms</strong>
                      </span>
                    )}
                    {currentResult.telemetry?.segmentation_ms && (
                      <span className="rounded-full bg-surface-2 border border-hairline px-2.5 py-0.5">
                        Seg: <strong className="text-foreground">{currentResult.telemetry.segmentation_ms}ms</strong>
                      </span>
                    )}
                    {currentResult.telemetry?.transform_ms && (
                      <span className="rounded-full bg-surface-2 border border-hairline px-2.5 py-0.5">
                        Warp: <strong className="text-foreground">{currentResult.telemetry.transform_ms}ms</strong>
                      </span>
                    )}
                    {currentResult.telemetry?.composite_ms && (
                      <span className="rounded-full bg-surface-2 border border-hairline px-2.5 py-0.5">
                        Blend: <strong className="text-foreground">{currentResult.telemetry.composite_ms}ms</strong>
                      </span>
                    )}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* OUTFIT HISTORY & COMPARISON TRAY */}
          {history.length > 0 && (
            <Card className="glass border-hairline rounded-2xl overflow-hidden">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="h-4 w-4 text-emerald" />
                    <CardTitle className="text-sm font-medium">Session History ({history.length})</CardTitle>
                  </div>
                  {history.length > 1 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsCompareModalOpen(true)}
                      className="h-6 px-2 text-xs text-emerald hover:text-emerald-glow"
                    >
                      Compare All
                    </Button>
                  )}
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-none">
                  {history.map((item) => (
                    <div
                      key={item.id}
                      onClick={() => setCurrentResult(item)}
                      className={`group relative shrink-0 aspect-[3/4] w-24 cursor-pointer overflow-hidden rounded-xl border transition-all ${
                        currentResult?.id === item.id ? "border-emerald ring-2 ring-emerald/40" : "border-hairline hover:border-white/30"
                      }`}
                    >
                      <img src={item.resultImage} alt={item.garmentName} className="h-full w-full object-cover" />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />
                      <div className="absolute bottom-1.5 left-1.5 right-1.5">
                        <p className="truncate text-[9px] font-medium text-white">{item.garmentName}</p>
                        <p className="text-[8px] text-muted-foreground">{item.timestamp}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* MODAL 1: Compare Looks Side-by-Side */}
      <Dialog open={isCompareModalOpen} onOpenChange={setIsCompareModalOpen}>
        <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto glass border-hairline">
          <DialogHeader>
            <DialogTitle className="text-lg">Compare Virtual Outfits</DialogTitle>
            <DialogDescription className="text-xs">
              Side-by-side comparison of different garments fitted to your silhouette.
            </DialogDescription>
          </DialogHeader>

          <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-3 pt-2">
            {history.map((look) => (
              <div key={look.id} className="rounded-xl border border-hairline bg-surface-2/60 p-3 space-y-2">
                <div className="relative aspect-[3/4] overflow-hidden rounded-lg bg-surface">
                  <img src={look.resultImage} alt={look.garmentName} className="h-full w-full object-contain" />
                  <div className="absolute bottom-2 right-2 flex gap-1">
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => {
                        setCurrentResult(look);
                        setIsCompareModalOpen(false);
                      }}
                      className="h-6 rounded-full text-[10px] px-2 bg-black/70 text-white hover:bg-black"
                    >
                      Load in Studio
                    </Button>
                  </div>
                </div>
                <div>
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-xs truncate">{look.garmentName}</h4>
                    <span className="text-[10px] text-muted-foreground">{look.timestamp}</span>
                  </div>
                  <p className="text-[10px] text-emerald">{look.garmentCategory}</p>
                </div>
              </div>
            ))}
          </div>
        </DialogContent>
      </Dialog>

      {/* MODAL 2: Contour Segmentation Inspection */}
      <Dialog open={isInspectingContour} onOpenChange={setIsInspectingContour}>
        <DialogContent className="max-w-md glass border-hairline">
          <DialogHeader>
            <DialogTitle className="text-base">Subject Contour & Torso Estimation</DialogTitle>
            <DialogDescription className="text-xs">
              Visualizing the computer vision segmentation mask and estimated 2D torso envelope.
            </DialogDescription>
          </DialogHeader>

          <div className="flex flex-col items-center justify-center p-2">
            {isLoadingContour ? (
              <div className="py-16 text-center space-y-2">
                <Loader2 className="h-8 w-8 animate-spin text-emerald mx-auto" />
                <p className="text-xs text-muted-foreground">Running GrabCut & contour analysis...</p>
              </div>
            ) : segmentationOverlay ? (
              <div className="space-y-3 w-full">
                <div className="relative aspect-[3/4] overflow-hidden rounded-xl border border-hairline bg-surface-2">
                  <img src={segmentationOverlay} alt="Contour Overlay" className="h-full w-full object-contain" />
                </div>
                <div className="space-y-1 text-[11px] text-muted-foreground bg-surface-2/60 p-3 rounded-lg border border-hairline">
                  <p className="flex items-center gap-1.5 font-medium text-foreground">
                    <span className="inline-block h-2 w-2 rounded-full bg-emerald" />
                    Green Rectangle: Estimated Torso Region
                  </p>
                  <p className="flex items-center gap-1.5 font-medium text-foreground">
                    <span className="inline-block h-2 w-2 rounded-full bg-white/60" />
                    Grey Rectangle: Subject Silhouette Bounding Box
                  </p>
                  <p className="flex items-center gap-1.5 font-medium text-foreground">
                    <span className="inline-block h-2 w-2 rounded-full bg-red-500" />
                    Red Dot: Subject Spatial Centroid
                  </p>
                </div>
              </div>
            ) : (
              <p className="text-xs text-muted-foreground py-8">No segmentation preview available.</p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
