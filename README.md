# 🧩 Puzzle Piece Analyzer

This project analyzes an image containing puzzle pieces, detects each piece, finds Harris corners, classifies side shapes (inward, outward), and identifies possible matches between pieces.

---

## 📂 Features

- Image preprocessing: grayscale, sharpening, blurring, binarization  
- Puzzle piece segmentation using contours  
- Harris corner detection and deduplication  
- Side shape classification: inward, outward, flat  
- Fuzzy matching of side shapes using Hausdorff distance  
- Visualization of processing stages

---

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Required packages:

- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`

---

## 🖼 Input

Place a test image (e.g. `testImage.png`) in the same directory. The image should contain multiple puzzle pieces on a contrasting background.

---

## ▶️ Usage

Run the main script:

```bash
python main.py
```

Or run it inside an IDE like PyCharm.

---

## 📸 Example Output

- Preprocessing stages (gray, sharpened, binary)
- Bounding boxes and deduplicated Harris corners
- Detected corners (4 per piece)
- Side classification:
  - ↪️ Tab (Outward)
  - ↩️ Slot (Inward)
  - ❓ Unknown
- Match scores between compatible puzzle piece sides

---

## 🧠 How It Works

### Preprocessing
Enhances contrast and detects foreground puzzle pieces.

### Corner Detection
Uses Harris corner detection + DBSCAN clustering to deduplicate.

### Side Extraction
Associates Harris points between corner pairs to extract edge shape.

### Shape Classification
Uses deviation from mean along axis to decide if edge is inward or outward.

### Matching
Uses directed Hausdorff distance to compare side profiles.

---

## 📌 Future Ideas

- Reconstruct full puzzle layout  
- Handle rotated/misaligned pieces  
- Export matches as JSON or UI overlay  
- Puzzle solving GUI  

---

## 📷 Sample Image Format

Use an image with:

- A white or light puzzle background  
- Clearly separated pieces  
- Uniform lighting (no heavy shadows)

---

## 📁 File Structure

```
project/
│
├── main.py               # Main script to run the puzzle analysis
├── testImage.png         # Input puzzle image (you supply this)
├── requirements.txt      # Python dependencies
└── README.md             # Project description
```
