# ROVER: Visual Perception & Traversability System for Unmanned Ground Vehicles

ROVER is a visual perception and traversability estimation system designed to help unmanned ground vehicles (UGVs) understand unstructured outdoor terrain. It identifies natural hazards and estimates safe driving corridors using camera-based perception. 

---

## 1. Problem Statement Context

* **Problem Statement ID**: 26126
* **Problem Statement Title**: Vision Based Autonomous Navigation for Unmanned Ground Vehicle for Outdoor environment
* **Category**: Smart India Hackathon (SIH)

### The Visual Perception Challenge
Outdoor off-road environments present unpredictable terrain, changing light, and unstructured routes. Standard navigation systems fail in unstructured environments without lane lines or clear pavement. To navigate safely, a ground vehicle must rely on local camera perception to distinguish drivable terrain from natural hazards (rocks, logs, trees) in real time. ROVER implements this critical perception and traversability estimation layer.

---

## 2. System Architecture

The UGV perception pipeline processes images through the following stages:

```
      Camera Image
           │
           ▼
 DINOv2 ViT-S/14 Transformer
           │
           ▼
 ROVER V3 Segmentation Head
           │
           ▼
 10-Class Terrain Segmentation
           │
           ▼
  Log/Rock Hazard Refinement
           │
           ▼
   Traversability Map
           │
           ▼
  Morphological Dilation Buffer
           │
           ▼
  Safe Corridor Estimation
           │
           ▼
   UGV Path Recommendation
```

---

## 3. Deep Learning Architecture

ROVER V3 utilizes a frozen transformer backbone combined with a custom convolution-based segmentation head.

* **Backbone**: DINOv2 ViT-S/14 (Frozen)
* **Input Resolution**: $476 \times 266$ pixels
* **Patch Grid**: $19 \times 34 = 646$ patch tokens
* **Feature Dimension**: 384 dimensions per patch token
* **Segmentation Head**:
  * $384 \to 128$ Conv2D (Kernel size 3, padding 1) + GELU activation
  * $128 \to 128$ Conv2D (Kernel size 3, padding 1) + GELU activation
  * $128 \to 10$ Classifier Conv2D (Kernel size 1)

---

## 4. Semantic Classes

The model classifies terrain into 10 distinct classes:

0. **Background**: Neutral or non-ground elements.
1. **Trees**: Vertically growing trees and trunks.
2. **Lush Bushes**: Green foliage and dense shrubbery.
3. **Dry Grass**: Tall, yellow, or dry grassy areas.
4. **Dry Bushes**: Woody, dry, or dead bushes.
5. **Ground Clutter**: Detritus, leaves, or minor twigs on the floor.
6. **Logs**: Fallen wooden logs (hazardous).
7. **Rocks**: Large rocks, boulders, and stones (hazardous).
8. **Landscape**: Open terrain, dirt roads, and traversable pathways.
9. **Sky**: Clouds and open sky.

---

## 5. Segmentation Performance

The V3 segmentation model achieves the following validation metrics evaluated on the off-road validation dataset:

* **Validation mIoU**: 51.90%
* **Validation Dice**: 65.83%
* **Validation Accuracy**: 81.29%

### Per-Class Validation IoU Breakdown
* **Sky**: 0.9622
* **Trees**: 0.7066
* **Dry Grass**: 0.6115
* **Lush Bushes**: 0.6059
* **Landscape**: 0.5775
* **Background**: 0.5085
* **Dry Bushes**: 0.3894
* **Rocks**: 0.3320
* **Logs**: 0.2570
* **Ground Clutter**: 0.2392

*Note: These are validation metrics for the semantic segmentation network and do not represent the final navigation pathing accuracy.*

---

## 6. Traversability Reasoning

The system converts semantic segmentation masks into a physical drivability score based on predicted classes and local confidence:

* **Preferred Ground**: Landscape, Dry Grass (score $\approx 1.0$)
* **Caution Ground**: Lush Bushes, Dry Bushes (score $\approx 0.4$), Ground Clutter (score $\approx 0.2$)
* **Hazard Zones**: Logs, Rocks (score $\approx -1.0$)
* **Invalid Regions**: Sky, Trees, Background (score $= 0.0$)

---

## 7. Hazard Refinement Layer

ROVER utilizes a second-stage visual hazard refinement system at runtime to filter out false hazard detections. These checks run in the backend API at inference time. The trained model weights remain completely unchanged.

### Log Refinement
Fallen wooden structures are verified using a weighted confidence score:
$$\text{log\_confidence} = 0.50 \times \text{v3\_log\_prob} + 0.20 \times \text{elongation} + 0.15 \times \text{edge\_continuity} + 0.15 \times \text{wood\_color}$$
* **Evidence Check**: Rejects candidates if the peak V3 log probability in the component is $< 0.15$.
* **Shape Constraints**: Requires elongated geometry (aspect ratio $\ge 1.8$), thin profile, and a maximum size limit to prevent sandy dirt trails from being misclassified as logs.
* **Classification**: Promoted to **Hard Log** if component area $\ge 150$ px and confidence $\ge 0.70$. Otherwise, it is classified as **Low-Confidence Log**.

### Rock Refinement
Rocks are verified using solidity and edge density:
$$\text{rock\_confidence} = 0.50 \times \text{v3\_rock\_prob} + 0.25 \times \text{solidity} + 0.25 \times \text{edge\_density}$$
* Rejects candidates if peak Rock probability is $< 0.15$.
* Promoted to **Hard Rock** if area $\ge 150$ px and confidence $\ge 0.70$. Otherwise, it remains a **Low-Confidence Rock**.

---

## 8. Hazard Safety Buffers

To prevent the vehicle from driving too close to detected obstacles, Hard Rock and Hard Log hazards receive a spatial safety buffer. The system uses morphological dilation (`cv2.dilate` with a 15x15 ellipse kernel) to follow obstacle boundaries naturally. Entering a buffer zone reduces the drivability score by 70%.

---

## 9. Safe Corridor Estimation

The planning algorithm scans candidate lanes of width 90 pixels across the lower 45% (near-ground driving region) of the camera feed:
* **Corridor Scoring**: Evaluates candidate lanes by accumulating traversability scores and buffer penalties.
* **Corridor Status**:
  * **`SAFE`**: Clear path corridor found with high drivability and zero hard hazard occupancy.
  * **`CAUTION`**: Lane selected but contains low-confidence hazards or is close to obstacle buffers.
  * **`BLOCKED`**: No safe corridor exists. The selected lane contains a hard hazard (hazard occupancy $\ge 40$ pixels).

---

## 10. ROVER Telemetry Cockpit

The web interface serves as a robotics diagnostic dashboard:

* **Raw Prediction Mode**: Renders the exact 10-class predictions generated by the ROVER V3 model.
* **Traversability Mode**: Renders BGR-coded drivability maps (Green for Traversable, Yellow for Caution, Red for Hard Hazards, Gray for Sky/Trees) blended at 20% opacity.
* **Telemetry Statistics**: Displays UGV status (`SAFE` / `CAUTION` / `BLOCKED`), Traversability Index, obstacle count, rock/log area percentages, processing latency, and static Validation mIoU (`51.9%`).
* **Visual Annotations**: Shows perspective lane overlays, dashed blocked corridors, and bounding outlines for Hard hazards.

---

## 11. Technology Stack

* **Backend**: Python, PyTorch, DINOv2, FastAPI, OpenCV, torchvision
* **Frontend**: HTML, CSS, JavaScript

---

## 12. Repository Structure

```
ROVER/
│
├── rover_v3.ipynb                      # Training notebook & architecture source of truth
├── best_rover_v3_segmentation_head.pth # Saved weights for the custom segmentation head
│
├── api.py                              # FastAPI backend serving inference and telemetry
├── diagnostic_interface.html           # HTML/JS/CSS telemetry dashboard UI
│
├── requirements.txt                    # Python package dependencies
├── .gitignore                          # Git patterns to ignore
└── .gitattributes                       # Git attributes definition
```

---

## 13. Getting Started

### Prerequisites
Ensure Python 3.9+ is installed on your system.

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd object-segmentation
   ```

2. **Create and activate a virtual environment**:
   * Windows:
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   * Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI server**:
   ```bash
   python api.py
   ```

5. **Open the Dashboard**:
   Open your browser and navigate to `http://127.0.0.1:8000`.

6. **Upload and Scan**:
   Select an off-road image to view the raw segmentation mask and traversability overlays.

---

## 14. Demo Workflow

1. **Input**: The user uploads an off-road camera image to the telemetry cockpit.
2. **Segmentation**: The DINOv2 + ROVER V3 network outputs a pixel-level 10-class segmentation mask.
3. **Hazard Interpretation**: The refinement layer filters out noise and determines Hard vs. Low-Confidence hazard boundaries.
4. **Traversability**: The backend generates BGR category overlays representing physical drivability.
5. **Corridor Scan**: The planner searches the lower ground region and draws the recommended corridor (perspective lane or blocked dashed boundary).
6. **Telemetry**: Telemetry stats (UGV status, index %, rock/log area %, latency) are returned to the dashboard.
