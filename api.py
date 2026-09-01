from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import time
import uvicorn
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import io
import base64
import numpy as np
import torchvision.transforms as transforms
import cv2

# ============================================================================
# MODEL DEFINITION
# ============================================================================
class RoverSegHead(nn.Module):

    def __init__(
        self,
        in_channels=384,
        num_classes=10,
        token_h=19,
        token_w=34
    ):

        super().__init__()

        self.token_h = token_h
        self.token_w = token_w

        self.conv1 = nn.Conv2d(
            in_channels,
            128,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            128,
            128,
            kernel_size=3,
            padding=1
        )

        self.classifier = nn.Conv2d(
            128,
            num_classes,
            kernel_size=1
        )

        self.activation = nn.GELU()

    def forward(self, x):

        # x = [B, N, C]
        B, N, C = x.shape

        expected_tokens = (
            self.token_h *
            self.token_w
        )

        if N != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} patch tokens "
                f"({self.token_h}x{self.token_w}), "
                f"but received {N}."
            )

        x = x.reshape(
            B,
            self.token_h,
            self.token_w,
            C
        )

        x = x.permute(
            0, 3, 1, 2
        )

        x = self.activation(
            self.conv1(x)
        )

        x = self.activation(
            self.conv2(x)
        )

        x = self.classifier(x)

        return x

# ============================================================================
# APP INITIALIZATION
# ============================================================================
app = FastAPI(title="AuraNav Edge API")

# Mount local static files for 100% offline frontend operation
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Input resolution: must be divisible by 14 (DINOv2 patch size) and must
# match the resolution used during training exactly to avoid patch misalignment.
w, h = 476, 266

# ============================================================================
# POST-PROCESSING CONFIGURATION
# ============================================================================
POSTPROCESS_ENABLED = True
DEBUG_POSTPROCESS = False
DEBUG_TRAVERSABILITY = True  # Keep debug logs active
DEBUG_PERCEPTION = True      # Enable second-stage visual refinement debug logs

# Logit smoothing
LOGIT_SMOOTH_KERNEL = 3  # kernel size for spatial average pooling

# Logs/Rocks refinement
LOG_ROCK_REFINEMENT_ENABLED = True
LOG_ROCK_DIFF_THRESHOLD = 0.5
LOG_ROCK_CALIBRATION_BIAS = 0.0

# Tiny component cleanup thresholds (in pixels)
MIN_COMPONENT_SIZE_DEFAULT = 15
MIN_COMPONENT_SIZE_LUSH_BUSH = 5
MIN_COMPONENT_SIZE_LOG = 5
MIN_COMPONENT_SIZE_ROCK = 5

# Substantial support threshold for removing small Logs/Rocks components
SUBSTANTIAL_SUPPORT_THRESHOLD = 0.5

# ============================================================================
# TRAVERSABILITY ESTIMATION CONFIGURATION
# ============================================================================
HAZARD_BUFFER_RADIUS = 25  # Bounding buffer zone in pixels
LANE_WIDTH = 90  # Candidate path column width
CORRIDOR_SAFE_THRESHOLD = 0.6
CORRIDOR_CAUTION_THRESHOLD = 0.35

# Refined Hazard Layer Thresholds
LOG_CONFIDENCE_THRESHOLD_HARD = 0.70
LOG_CONFIDENCE_THRESHOLD_LOW = 0.40
ROCK_CONFIDENCE_THRESHOLD_HARD = 0.70
ROCK_CONFIDENCE_THRESHOLD_LOW = 0.40

print("Loading DINOv2 backbone...")
backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
backbone.eval().to(device)

print("Loading fine-tuned segmentation head...")
head = RoverSegHead(in_channels=384, num_classes=10, token_h=h//14, token_w=w//14)
head.load_state_dict(torch.load("best_rover_v3_segmentation_head.pth", map_location=device))
head.eval().to(device)

# Preprocessing pipeline - identical to the one used in training
transform = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Color palette: one RGB color per class index (0-9)
color_palette = np.array([
    [0,   0,   0  ],  # 0 - Background
    [34,  139, 34 ],  # 1 - Trees
    [50,  205, 50 ],  # 2 - Lush Bushes
    [255, 215, 0  ],  # 3 - Dry Grass
    [255, 140, 0  ],  # 4 - Dry Bushes
    [128, 128, 128],  # 5 - Ground Clutter
    [255, 0,   255],  # 6 - Logs (hazard)
    [255, 0,   0  ],  # 7 - Rocks (hazard)
    [0,   255, 0  ],  # 8 - Landscape
    [0,   191, 255],  # 9 - Sky
], dtype=np.uint8)

# BGR Palette for traversability map overlays
trav_color_palette = np.array([
    [20, 20, 20],      # 0: Neutral/Dark (Invalid)
    [0, 180, 0],       # 1: Green (Traversable)
    [0, 180, 255],     # 2: Yellow (Caution)
    [0, 0, 200]        # 3: Red (Hazard)
], dtype=np.uint8)

# ============================================================================
# ROUTES
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the diagnostic frontend interface."""
    with open("diagnostic_interface.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded image, runs DINOv2 + SegHead inference, and returns:
    - mask_base64: raw/post-processed segmentation overlay blended with original
    - traversability_base64: traversability corridor + hazard bounding boxes HUD
    - telemetry: hazard metrics, confidence, and system status
    """
    try:
        t0 = time.time()

        # Read and decode uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess image to match training pipeline
        input_tensor = transform(image).unsqueeze(0).to(device)
        t1 = time.time()

        # Run inference
        with torch.no_grad():
            feats = backbone.forward_features(input_tensor)["x_norm_patchtokens"]
            t2 = time.time()
            logits = head(feats)
            t3 = time.time()

            # Upsample back to training resolution
            logits = F.interpolate(
                logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False
            )
            t4 = time.time()

            # Calculate confidence maps using softmax probs
            probs = F.softmax(logits, dim=1)
            confidence_map = torch.max(probs, dim=1).values.squeeze(0).cpu().numpy()
            avg_confidence = float(np.mean(confidence_map))

            # Save the RAW prediction mask before any post-processing
            raw_pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

            if POSTPROCESS_ENABLED:
                # 1. Logit-space smoothing: conservative 3x3 average pooling
                logits_smoothed = F.avg_pool2d(
                    logits,
                    kernel_size=LOGIT_SMOOTH_KERNEL,
                    stride=1,
                    padding=LOGIT_SMOOTH_KERNEL // 2
                )

                # 2. Logs/Rocks refinement (Logs = 6, Rocks = 7)
                if LOG_ROCK_REFINEMENT_ENABLED:
                    diff = torch.abs(logits[:, 6, :, :] - logits[:, 7, :, :])
                    mask_ambiguous = diff < LOG_ROCK_DIFF_THRESHOLD
                    
                    # Favor Logs if local neighborhood support is stronger
                    log_stronger = logits_smoothed[:, 6, :, :] > logits_smoothed[:, 7, :, :]
                    prefer_log = mask_ambiguous & log_stronger
                    
                    # Expose LOG_ROCK_CALIBRATION_BIAS (which is 0.0 initially)
                    logits[:, 6, :, :][prefer_log] = torch.max(
                        logits[:, 6, :, :][prefer_log],
                        logits[:, 7, :, :][prefer_log] + 0.1 + LOG_ROCK_CALIBRATION_BIAS
                    )

                # Get the post-processed prediction mask base
                postprocessed_pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

                # Extract second-best predictions for component replacement
                top2_indices = torch.topk(logits, 2, dim=1).indices.squeeze(0).cpu().numpy()
                second_best = top2_indices[1]

                logits_smoothed_np = logits_smoothed.squeeze(0).cpu().numpy()

                # 3. Tiny component cleanup using connected components
                for c in range(10):
                    # Select appropriate threshold
                    if c == 2:
                        threshold = MIN_COMPONENT_SIZE_LUSH_BUSH
                    elif c == 6:
                        threshold = MIN_COMPONENT_SIZE_LOG
                    elif c == 7:
                        threshold = MIN_COMPONENT_SIZE_ROCK
                    else:
                        threshold = MIN_COMPONENT_SIZE_DEFAULT

                    if threshold <= 0:
                        continue

                    class_mask = (postprocessed_pred_mask == c).astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

                    for label in range(1, num_labels):
                        area = stats[label, cv2.CC_STAT_AREA]
                        
                        # Lush Bushes (class 2): Never remove solely because it is small (i.e. size >= 5 is preserved)
                        if c == 2:
                            if area < threshold:
                                ys, xs = np.where(labels == label)
                                postprocessed_pred_mask[ys, xs] = second_best[ys, xs]
                        
                        # Logs (6) and Rocks (7): Only clean if below threshold AND second-best has stronger local support.
                        elif c in (6, 7):
                            if area < threshold:
                                ys, xs = np.where(labels == label)
                                orig_support = logits_smoothed_np[c, ys, xs]
                                sec_classes = second_best[ys, xs]
                                sec_support = logits_smoothed_np[sec_classes, ys, xs]

                                mean_orig = np.mean(orig_support)
                                mean_sec = np.mean(sec_support)

                                if mean_sec > mean_orig + SUBSTANTIAL_SUPPORT_THRESHOLD:
                                    postprocessed_pred_mask[ys, xs] = sec_classes
                        
                        # Default classes: replace directly if below threshold
                        else:
                            if area < threshold:
                                ys, xs = np.where(labels == label)
                                postprocessed_pred_mask[ys, xs] = second_best[ys, xs]

                pred_mask = postprocessed_pred_mask
            else:
                pred_mask = raw_pred_mask
            t5 = time.time()

        # ============================================================================
        # SECOND-STAGE VISUAL HAZARD REFINEMENT LAYER
        # ============================================================================
        start_row = int(h * 0.55)  # 146

        # Convert base image to BGR numpy array for OpenCV analysis
        img_bgr = cv2.cvtColor(np.array(image.resize((w, h))), cv2.COLOR_RGB2BGR)
        b, g, r = cv2.split(img_bgr)

        # 1. Color Filters
        # Brown color filter for logs (R > G > B, substantial R-B difference)
        brown_mask = (r.astype(np.int32) - g.astype(np.int32) >= 8) & \
                     (g.astype(np.int32) - b.astype(np.int32) >= 8) & \
                     (r > 50) & (b < 150)
        
        # Green color filter for vegetation (G > R and G > B)
        green_mask = (g.astype(np.int32) - r.astype(np.int32) >= 5) & \
                     (g.astype(np.int32) - b.astype(np.int32) >= 5)

        # Gray/sandy color filter for dirt trail likelihood
        dirt_color_mask = (np.abs(r.astype(np.int32) - g.astype(np.int32)) < 20) & \
                          (r.astype(np.int32) - b.astype(np.int32) > 10) & \
                          (r > 70)

        # 2. Local Texture/Edge Variation Analyzer
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_density = cv2.GaussianBlur(np.abs(laplacian), (15, 15), 0)
        edge_norm = np.clip(edge_density / (np.max(edge_density) + 1e-5), 0, 1)

        # Extract probability arrays from softmax
        probs_np = probs.squeeze(0).cpu().numpy()
        log_probs_map = probs_np[6]
        rock_probs_map = probs_np[7]

        # 3. Log Component Refinement
        # Combine V3 evidence and brown structures
        log_candidate_base = (np.isin(pred_mask, [3, 6, 7, 8]) & brown_mask).astype(np.uint8)
        num_l, labels_l, stats_l, centroids_l = cv2.connectedComponentsWithStats(log_candidate_base)
        
        hard_log_mask = np.zeros_like(pred_mask, dtype=bool)
        low_conf_log_mask = np.zeros_like(pred_mask, dtype=bool)

        for i in range(1, num_l):
            lx, ly, lwidth, lheight, larea = stats_l[i]
            
            if ly + lheight < start_row:
                continue
            if larea < 15:
                continue
                
            comp_mask = (labels_l == i)
            
            # Compute Log Confidence Components
            # 1. V3 Log Probability max inside component to detect actual log signals
            v3_log_prob = float(np.max(log_probs_map[comp_mask]))
            
            # Ignore log candidates with very low V3 log evidence
            if v3_log_prob < 0.15:
                continue
                
            # 2. Elongation Aspect Ratio
            aspect_ratio = float(max(lwidth, lheight) / (min(lwidth, lheight) + 1e-5))
            elongation_score = float(min(aspect_ratio / 3.0, 1.0))
            # 3. Edge Continuity
            comp_edges = edge_norm[comp_mask]
            edge_continuity = float(np.sum(comp_edges > 0.2) / len(comp_edges))
            # 4. Wood Color Score
            comp_brown = brown_mask[comp_mask]
            wood_color_score = float(np.sum(comp_brown) / len(comp_brown))
            
            # Log Confidence Score formula
            log_confidence = 0.50 * v3_log_prob + 0.20 * elongation_score + 0.15 * edge_continuity + 0.15 * wood_color_score
            
            is_hard = False
            is_low_conf = False
            
            # Classify logs strictly
            if larea >= 150 and log_confidence >= LOG_CONFIDENCE_THRESHOLD_HARD:
                is_hard = True
            elif log_confidence >= LOG_CONFIDENCE_THRESHOLD_LOW:
                is_low_conf = True

            if is_hard:
                hard_log_mask[comp_mask] = True
            elif is_low_conf:
                low_conf_log_mask[comp_mask] = True

        # 4. Rock Component Refinement
        # Analyze predicted rocks (class 7)
        rocks_mask = (pred_mask == 7).astype(np.uint8)
        num_r, labels_r, stats_r, centroids_r = cv2.connectedComponentsWithStats(rocks_mask)
        
        hard_rock_mask = np.zeros_like(pred_mask, dtype=bool)
        low_conf_rock_mask = np.zeros_like(pred_mask, dtype=bool)

        for i in range(1, num_r):
            rx, ry, rwidth, rheight, rarea = stats_r[i]
            
            if ry + rheight < start_row:
                continue
            if rarea < 15:
                continue
                
            comp_mask = (labels_r == i)
            
            # Compute Rock Confidence Components
            # 1. V3 Rock Probability max inside component to detect actual rock signals
            v3_rock_prob = float(np.max(rock_probs_map[comp_mask]))
            
            # Ignore rock candidates with very low V3 rock evidence
            if v3_rock_prob < 0.15:
                continue
                
            # 2. Solidity
            solidity = float(rarea / (rwidth * rheight + 1e-5))
            solidity_score = float(min(solidity / 0.7, 1.0))
            # 3. Edge Density
            mean_edge = float(np.mean(edge_norm[comp_mask]))
            edge_density_score = float(min(mean_edge / 0.3, 1.0))
            
            # Rock Confidence Score formula
            rock_confidence = 0.50 * v3_rock_prob + 0.25 * solidity_score + 0.25 * edge_density_score
            
            is_hard = False
            is_low_conf = False
            
            if rarea >= 150 and rock_confidence >= ROCK_CONFIDENCE_THRESHOLD_HARD:
                is_hard = True
            elif rock_confidence >= ROCK_CONFIDENCE_THRESHOLD_LOW:
                is_low_conf = True
                
            if is_hard:
                hard_rock_mask[comp_mask] = True
            elif is_low_conf:
                low_conf_rock_mask[comp_mask] = True

        # 5. Landscape Overprediction Adjustments (Dirt & Vegetation Likelihoods)
        # Create separate physical traversability score map
        physical_trav_score = np.zeros_like(pred_mask, dtype=np.float32)

        # Baseline scores for other classes
        physical_trav_score[pred_mask == 3] = 0.9  # Dry Grass
        physical_trav_score[pred_mask == 2] = 0.4  # Lush Bushes (Caution)
        physical_trav_score[pred_mask == 4] = 0.4  # Dry Bushes (Caution)
        physical_trav_score[pred_mask == 5] = 0.2  # Ground Clutter
        
        # Non-drivable regions remain 0.0
        physical_trav_score[np.isin(pred_mask, [0, 1, 9])] = 0.0

        # Adjust Landscape (class 8) traversability score
        landscape_mask = (pred_mask == 8)
        for y in range(h):
            for x in range(w):
                if landscape_mask[y, x]:
                    if y >= start_row:
                        # Vegetation likelihood check (Green color dominance)
                        if green_mask[y, x]:
                            physical_trav_score[y, x] = 0.4  # Reduced to CAUTION
                        # Dirt road likelihood (Smooth ground texture + sandy color)
                        elif dirt_color_mask[y, x] and edge_norm[y, x] < 0.25:
                            physical_trav_score[y, x] = 0.95 # Clear road (Preferred)
                        # Rocky/rough terrain likelihood
                        elif edge_norm[y, x] > 0.4:
                            physical_trav_score[y, x] = 0.5  # Rocky landscape (Caution)
                        else:
                            physical_trav_score[y, x] = 0.8  # Traversable
                    else:
                        # Above horizon
                        physical_trav_score[y, x] = 0.0

        # Enforce Hazard scores
        hard_hazard_mask = hard_rock_mask | hard_log_mask
        physical_trav_score[hard_hazard_mask] = -1.0
        physical_trav_score[low_conf_rock_mask | low_conf_log_mask] = 0.2

        # 6. Apply dilated buffer around Hard Hazards
        dist_to_hazard = cv2.distanceTransform((~hard_hazard_mask).astype(np.uint8), cv2.DIST_L2, 5)
        buffer_mask = dist_to_hazard < HAZARD_BUFFER_RADIUS
        
        # Apply 70% penalty in buffers
        physical_trav_score = np.where((physical_trav_score > 0) & buffer_mask, physical_trav_score * 0.3, physical_trav_score)
        physical_trav_score[hard_hazard_mask] = -1.0

        # ============================================================================
        # NAVIGATION CORRIDOR SCAN
        # ============================================================================
        col_scores = np.zeros(w, dtype=np.float32)
        for x in range(w):
            region_scores = physical_trav_score[start_row:, x]
            region_mask = pred_mask[start_row:, x]
            avg_score = np.mean(region_scores)
            
            # Penalize column features
            has_hard_hazard = np.any(hard_hazard_mask[start_row:, x])
            has_low_conf_hazard = np.any(low_conf_rock_mask[start_row:, x] | low_conf_log_mask[start_row:, x])
            has_nondrivable = np.any((region_mask == 0) | (region_mask == 1) | (region_mask == 9))
            close_to_hard_hazard = np.any(dist_to_hazard[start_row:, x] < 10)
            
            if has_hard_hazard:
                avg_score -= 2.0
            if has_low_conf_hazard and not has_hard_hazard:
                avg_score -= 0.3
            if has_nondrivable:
                avg_score -= 1.0
            if close_to_hard_hazard:
                avg_score -= 0.5
                
            col_scores[x] = avg_score

        # Safe Corridor Scan
        lane_scores = []
        for x in range(w - LANE_WIDTH + 1):
            lane_score = np.mean(col_scores[x : x + LANE_WIDTH])
            lane_scores.append((x, lane_score))

        best_x, best_score = max(lane_scores, key=lambda val: val[1])
        normalized_score = float(np.clip((best_score + 1.0) / 2.0, 0.0, 1.0))

        x_start = int(best_x)
        x_end = int(best_x + LANE_WIDTH)

        # Check occupancy inside the selected corridor window
        corridor_hard_occupancy = int(np.sum(hard_hazard_mask[start_row:, x_start:x_end]))
        corridor_trav_occupancy = int(np.sum(physical_trav_score[start_row:, x_start:x_end] >= 0.75))

        if corridor_hard_occupancy >= 40:
            status = "BLOCKED"
        elif normalized_score < CORRIDOR_SAFE_THRESHOLD:
            status = "CAUTION"
        else:
            status = "SAFE"

        # Compute traversability index (drivable area % in lower 45%)
        lower_region = pred_mask[start_row:, :]
        total_lower_pixels = lower_region.size
        drivable_pixels = np.sum(physical_trav_score[start_row:, :] >= 0.75)
        traversability_index = int((drivable_pixels / total_lower_pixels) * 100)

        # Structured Telemetry stats
        total_pixels = pred_mask.size
        rock_pixels = np.sum(pred_mask == 7)
        log_pixels  = np.sum(pred_mask == 6)
        rock_pct = (rock_pixels / total_pixels) * 100
        log_pct = (log_pixels / total_pixels) * 100
        hazard_pct = rock_pct + log_pct
        t6 = time.time()

        # Debug Logs
        if DEBUG_PERCEPTION:
            print(f"\n[DEBUG PERCEPTION]")
            print(f"RAW ROCK: {rock_pixels} px ({rock_pct:.2f}%) | RAW LOG: {log_pixels} px ({log_pct:.2f}%) | LANDSCAPE: {np.sum(pred_mask == 8)} px")
            print(f"HARD ROCK: {np.sum(hard_rock_mask)} px | LOW-CONF ROCK: {np.sum(low_conf_rock_mask)} px")
            print(f"HARD LOG: {np.sum(hard_log_mask)} px | LOW-CONF LOG: {np.sum(low_conf_log_mask)} px")
            print(f"REFINED TRAVERSABLE: {np.sum(physical_trav_score >= 0.75)} px ({traversability_index}%)")
            print(f"CORRIDOR: {status} | Score: {normalized_score:.3f} | x-range: {x_start}-{x_end}")

        # ============================================================================
        # HUD GENERATION & BLENDING
        # ============================================================================
        # 1. Base Segmentation Overlay (displays the exact predictions raw)
        color_mask = color_palette[pred_mask]
        base_img  = image.copy().convert("RGB")
        mask_img  = Image.fromarray(color_mask).resize(base_img.size, Image.NEAREST)
        segmentation_overlay = Image.blend(base_img, mask_img, alpha=0.5)

        buffered_seg = io.BytesIO()
        segmentation_overlay.save(buffered_seg, format="JPEG", quality=85)
        mask_base64 = base64.b64encode(buffered_seg.getvalue()).decode("utf-8")

        # 2. Traversability Overlay View
        trav_overlay_img = image.copy().resize((w, h))
        trav_cv = cv2.cvtColor(np.array(trav_overlay_img), cv2.COLOR_RGB2BGR)

        # Render Pixel-Level Traversability Map Overlay
        category_map = np.zeros_like(pred_mask, dtype=np.uint8)
        category_map[physical_trav_score >= 0.75] = 1                         # Traversable (Green)
        category_map[(physical_trav_score > 0.1) & (physical_trav_score < 0.75)] = 2 # Caution (Yellow)
        category_map[buffer_mask] = 2                                         # Buffers -> Caution
        category_map[low_conf_rock_mask | low_conf_log_mask] = 2              # Low conf -> Caution
        category_map[hard_hazard_mask] = 3                                    # Hard Hazards -> Hazard (Red)
        
        trav_mask_color = trav_color_palette[category_map]
        cv2.addWeighted(trav_mask_color, 0.20, trav_cv, 0.80, 0, trav_cv)

        # Draw substantial bounding boxes for Hard Hazards only
        hazards = []
        
        # Logs
        for i in range(1, num_l):
            lx, ly, lwidth, lheight, larea = stats_l[i]
            if larea >= 15:
                # Use base target indices to reference classification safely
                is_hard = hard_log_mask[labels_l == i][0] if len(hard_log_mask[labels_l == i]) > 0 else False
                is_low = low_conf_log_mask[labels_l == i][0] if len(low_conf_log_mask[labels_l == i]) > 0 else False
                
                hazards.append({
                    "class": "Log",
                    "area": float(larea / total_pixels * 100),
                    "bbox": [int(lx), int(ly), int(lwidth), int(lheight)],
                    "confidence": "HARD" if is_hard else ("LOW-CONF" if is_low else "NONE")
                })
                # Bounding box only for HARD hazards >= 150 px
                if is_hard and larea >= 150:
                    cv2.rectangle(trav_cv, (lx, ly), (lx+lwidth, ly+lheight), (255, 0, 255), 1)
                    cv2.putText(trav_cv, "LOG", (lx + 3, ly + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    if status == "BLOCKED" and not (lx + lwidth < x_start or lx > x_end):
                        cv2.putText(trav_cv, "LOG HAZARD", (lx, ly - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1, cv2.LINE_AA)

        # Rocks
        for i in range(1, num_r):
            rx, ry, rwidth, rheight, rarea = stats_r[i]
            if rarea >= 15:
                is_hard = hard_rock_mask[labels_r == i][0] if len(hard_rock_mask[labels_r == i]) > 0 else False
                is_low = low_conf_rock_mask[labels_r == i][0] if len(low_conf_rock_mask[labels_r == i]) > 0 else False
                
                hazards.append({
                    "class": "Rock",
                    "area": float(rarea / total_pixels * 100),
                    "bbox": [int(rx), int(ry), int(rwidth), int(rheight)],
                    "confidence": "HARD" if is_hard else ("LOW-CONF" if is_low else "NONE")
                })
                # Bounding box only for HARD hazards >= 150 px
                if is_hard and rarea >= 150:
                    cv2.rectangle(trav_cv, (rx, ry), (rx+rwidth, ry+rheight), (0, 0, 255), 1)
                    cv2.putText(trav_cv, "ROCK", (rx + 3, ry + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    if status == "BLOCKED" and not (rx + rwidth < x_start or rx > x_end):
                        cv2.putText(trav_cv, "ROCK HAZARD", (rx, ry - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw Perspective Driving Corridor
        center_x = (x_start + x_end) // 2
        lane_w_top = int(LANE_WIDTH * 0.6)
        
        x_bl = x_start
        x_br = x_end
        x_tl = center_x - lane_w_top // 2
        x_tr = center_x + lane_w_top // 2
        
        corridor_poly = np.array([[x_bl, h], [x_tl, start_row], [x_tr, start_row], [x_br, h]], dtype=np.int32)
        
        if status in ("SAFE", "CAUTION"):
            overlay_color = (0, 255, 0) if status == "SAFE" else (0, 255, 255)
            
            # Semi-translucent overlay
            overlay_mask = trav_cv.copy()
            cv2.fillPoly(overlay_mask, [corridor_poly], overlay_color)
            cv2.addWeighted(overlay_mask, 0.2, trav_cv, 0.8, 0, trav_cv)
            
            # Draw perspective borders
            cv2.line(trav_cv, (x_bl, h), (x_tl, start_row), overlay_color, 1, cv2.LINE_AA)
            cv2.line(trav_cv, (x_br, h), (x_tr, start_row), overlay_color, 1, cv2.LINE_AA)
            
            # Draw thin centerline
            for y_dash in range(start_row, h, 14):
                curr_cx = int(center_x)
                cv2.line(trav_cv, (curr_cx, y_dash), (curr_cx, min(y_dash + 8, h)), overlay_color, 1, cv2.LINE_AA)
            
            label_text = "SAFE CORRIDOR" if status == "SAFE" else "CAUTION PATH"
            cv2.putText(trav_cv, label_text, (center_x - 35, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # BLOCKED Corridor outline
            overlay_color = (0, 0, 255) # Red dashed
            for y_dash in range(start_row, h, 14):
                t_ratio = (y_dash - start_row) / (h - start_row)
                curr_xl = int(x_tl + (x_bl - x_tl) * t_ratio)
                curr_xr = int(x_tr + (x_br - x_tr) * t_ratio)
                curr_xc = int(center_x)
                
                cv2.line(trav_cv, (curr_xl, y_dash), (curr_xl, min(y_dash + 8, h)), overlay_color, 1, cv2.LINE_AA)
                cv2.line(trav_cv, (curr_xr, y_dash), (curr_xr, min(y_dash + 8, h)), overlay_color, 1, cv2.LINE_AA)
                cv2.line(trav_cv, (curr_xc, y_dash), (curr_xc, min(y_dash + 8, h)), overlay_color, 1, cv2.LINE_AA)
            
            cv2.putText(trav_cv, "CORRIDOR BLOCKED", (center_x - 45, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert BGR back to Base64
        trav_img_rgb = cv2.cvtColor(trav_cv, cv2.COLOR_BGR2RGB)
        trav_pil = Image.fromarray(trav_img_rgb).resize(image.size, Image.BILINEAR)
        
        buffered_trav = io.BytesIO()
        trav_pil.save(buffered_trav, format="JPEG", quality=85)
        traversability_base64 = base64.b64encode(buffered_trav.getvalue()).decode("utf-8")

        t7 = time.time()
        latency_ms = int((t7 - t0) * 1000)

        if DEBUG_TRAVERSABILITY:
            print(f"[DEBUG LATENCY] Preprocess: {t1-t0:.3f}s, DINOv2: {t2-t1:.3f}s, Head: {t3-t2:.3f}s, Upsample: {t4-t3:.3f}s, Postprocess: {t5-t4:.3f}s, Traversability: {t6-t5:.3f}s, HUD/Base64: {t7-t6:.3f}s")

        # Compute class count dictionaries
        raw_pct_dict = {}
        post_pct_dict = {}
        class_names = ["Background", "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter", "Logs", "Rocks", "Landscape", "Sky"]
        for c_idx, name in enumerate(class_names):
            raw_pct_dict[name] = float((np.sum(raw_pred_mask == c_idx) / total_pixels) * 100)
            post_pct_dict[name] = float((np.sum(pred_mask == c_idx) / total_pixels) * 100)

        return {
            "status": "SUCCESS",
            "mask_base64": f"data:image/jpeg;base64,{mask_base64}",
            "traversability_base64": f"data:image/jpeg;base64,{traversability_base64}",
            "telemetry": {
                "hazard_level": f"{hazard_pct:.1f}%",
                "rock_area_percent": round(rock_pct, 2),
                "log_area_percent": round(log_pct, 2),
                "hazard_area_percent": round(hazard_pct, 2),
                "confidence": f"{avg_confidence * 100:.1f}%",
                "traversability": status,
                "corridor_status": status,
                "traversability_index": traversability_index,
                "latency": f"{latency_ms}ms"
            },
            "safe_corridor": {
                "x_start": x_start,
                "x_end": x_end,
                "score": round(normalized_score, 3),
                "status": status
            },
            "hazards": hazards,
            "raw_class_percentages": raw_pct_dict,
            "postprocessed_class_percentages": post_pct_dict
        }

    except Exception as e:
        return {"status": "ERROR", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
