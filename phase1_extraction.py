import cv2
import numpy as np
import os

# --- Configuration & Paths ---
DATASET_DIR = "."
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, "images")
TEMPLATE_IMAGES_DIR = os.path.join(DATASET_DIR, "PCB_USED")

OUTPUT_ROI_DIR = os.path.join(DATASET_DIR, "extracted_rois")
OUTPUT_DEBUG_DIR = os.path.join(DATASET_DIR, "debug_visuals")

os.makedirs(OUTPUT_ROI_DIR, exist_ok=True)
os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)

def process_images():
    defect_categories = [d for d in os.listdir(TEST_IMAGES_DIR) if os.path.isdir(os.path.join(TEST_IMAGES_DIR, d))]

    for category in defect_categories:
        print(f"Processing category: {category}...")
        category_path = os.path.join(TEST_IMAGES_DIR, category)
        
        os.makedirs(os.path.join(OUTPUT_ROI_DIR, category), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DEBUG_DIR, category), exist_ok=True)

        for img_name in os.listdir(category_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            test_img_path = os.path.join(category_path, img_name)
            
            template_prefix = img_name.split('_')[0]
            template_img_path = os.path.join(TEMPLATE_IMAGES_DIR, f"{template_prefix}.JPG")

            if not os.path.exists(template_img_path):
                print(f"  [Warning] Template {template_prefix}.JPG not found for {img_name}. Skipping.")
                continue

            test_img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
            template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
            test_img_color = cv2.imread(test_img_path)

            if test_img.shape != template_img.shape:
                test_img = cv2.resize(test_img, (template_img.shape[1], template_img.shape[0]))
                test_img_color = cv2.resize(test_img_color, (template_img.shape[1], template_img.shape[0]))

            diff_img = cv2.absdiff(template_img, test_img)
            _, thresh_img = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)
            clean_thresh = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
            clean_thresh = cv2.dilate(clean_thresh, kernel, iterations=1)

            contours, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            defect_count = 0
            for contour in contours:
                # --- CRITICAL FIX: Ignore noise smaller than 150 pixels ---
                if cv2.contourArea(contour) > 150: 
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    padding = 15
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(test_img.shape[1], x + w + padding)
                    y2 = min(test_img.shape[0], y + h + padding)

                    roi = test_img_color[y1:y2, x1:x2]
                    roi_resized = cv2.resize(roi, (128, 128))
                    
                    roi_filename = os.path.join(OUTPUT_ROI_DIR, category, f"roi_{defect_count}_{img_name}")
                    cv2.imwrite(roi_filename, roi_resized)
                    
                    cv2.rectangle(test_img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    defect_count += 1

            debug_filename = os.path.join(OUTPUT_DEBUG_DIR, category, f"debug_{img_name}")
            cv2.imwrite(debug_filename, test_img_color)

    print("Phase 1 Complete! Check 'extracted_rois' and 'debug_visuals' folders.")

if __name__ == "__main__":
    process_images()