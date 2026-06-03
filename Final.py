import cv2
import numpy as np
import math


def get_color_ratios(frame, contour):
    """Calculate color ratios within a contour region"""
    mask = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)
    masked_roi = cv2.bitwise_and(frame, frame, mask=mask)
    hsv = cv2.cvtColor(masked_roi, cv2.COLOR_BGR2HSV)
    
    
    color = {
        "red": ((158, 75, 85), (178, 255, 255)),      # Điều chỉnh red
        "blue": ((100, 85, 150), (130, 255, 255)),    # Điều chỉnh blue
        "yellow": ((15, 145, 145), (25, 255, 255)),   # Điều chỉnh yellow
        "white": ((0, 0, 125), (170, 75, 255)),       # Điều chỉnh white
    }
    
    total_pixels = cv2.countNonZero(mask)
    if total_pixels == 0:
        return {c: 0 for c in color}
    
    percentages = {}
    for c in color:
        lower = np.array(color[c][0])
        upper = np.array(color[c][1])
        mask_c = cv2.inRange(hsv, lower, upper)
        percentages[c] = (np.sum(mask_c > 0) / total_pixels) * 100  # Convert to percentage
    
    return percentages

def infer_sign_content(frame, circle_contours, triangle_contours):
    """Infer traffic sign content based on shape and color ratios"""
    detected_signs = []
    
    # Circle signs (prohibition/mandatory signs)
    for cnt in circle_contours:
        ratios = get_color_ratios(frame, cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        
        if ratios['red'] > 10 and ratios['blue'] < 20 and ratios["white"] ==0.0:
            content = "CAM DO XE"
        elif ratios['red'] <75 and 28 <ratios['blue'] < 43:
            content = "CAM DO VA DUNG XE"
        elif ratios['red'] <60 and ratios['blue'] < 10 and ratios["white"] >30:
            content = "CAM RE TRAI"
        elif ratios['red'] >70 and ratios["white"] <20:
            content = "CAM DI NGUOC CHIEU"
        elif  ratios['blue'] >50:
            content = "DI VONG SANG PHAI"
        else:
            content= "UNKNOWN"
        detected_signs.append({
            "rect": (x, y, w, h),
            "shape": "CIRCLE",
            "content": content,
            "ratios": ratios
        })
    
    # Triangle signs (warning signs)
    for cnt in triangle_contours:
        ratios = get_color_ratios(frame, cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        content = "BIEN NGUY HIEM"  # Danger sign
        
        # Warning signs
        if ratios['yellow'] < 75:
            content = "CHU Y QUA DUONG"  # Pedestrian crossing
        else:
            content = "NGUY HIEM DI CHAM"  # Danger - go slow
        
        detected_signs.append({
            "rect": (x, y, w, h),
            "shape": "TRIANGLE",
            "content": content,
            "ratios": ratios
        })
    
    return detected_signs

def draw_detected_signs(frame, detected_signs):
    """Draw bounding boxes and labels on detected signs"""
    for sign in detected_signs:
        x, y, w, h = sign["rect"]
        
        if sign["shape"] == "CIRCLE":
            cv_color = (0, 255, 0)  # Orange for circles
        else:
            cv_color = (0, 255, 0)  # Yellow for triangles
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), cv_color, 3)
        text = f"{sign['content']}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cv_color, 2)
    
    return frame

# ========== MAIN DETECTION AND PROCESSING ==========
cap = cv2.VideoCapture('task1.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    print("Cảnh báo: Không đọc được FPS, đặt mặc định là 30.")
    fps = 30  # fallback if FPS cannot be read

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('task1_output.mp4', fourcc, fps, (frame_width, frame_height))

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Xử lý video hoàn tất.")
        break
    
 
    
    # === Convert to HSV ===
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # === Color ranges ===
    # Red
    lower_red = np.array([159, 50, 70])
    upper_red = np.array([180, 255, 255])
    
    # Blue
    lower_blue = np.array([78, 158, 124])
    upper_blue = np.array([138, 255, 255])
    
    # Yellow
    lower_yellow = np.array([10, 120, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # === Create masks for each color ===
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # === Morphology operations to clean masks ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # === Combine masks ===
    mask = cv2.bitwise_or(mask_red, mask_blue)
    mask = cv2.bitwise_or(mask, mask_yellow)
    
    # === Limit to upper half of frame ===
    h, w = mask.shape
    region_mask = np.zeros_like(mask)
    region_mask[0:int(h / 2), :] = 255
    mask = cv2.bitwise_and(mask, region_mask)
    
    # === Remove small noise ===
    kernel_mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mask, iterations=1)
    
    # === Find contours ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    # Lists to store detected shapes
    circle_contours = []
    triangle_contours = []
    rectangle_contours = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(h)
        Bounding_Area = w * h
        Extent = area / Bounding_Area if Bounding_Area > 0 else 0
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        triangularity = (36 * area) / (math.sqrt(3) * (perimeter ** 2))
        
        isValid_Sign = False
        
        # === CIRCLE DETECTION ===
        if 0.8 < circularity < 1.15 and 0.8 < ratio < 1.2:
            circle_contours.append(cnt)
            isValid_Sign = True
        
        # === TRIANGLE DETECTION ===
        if not isValid_Sign and vertices == 3:
            if 0.6 < triangularity < 1.3 and 0.4 < Extent < 0.7:
                shape_mask = np.zeros_like(mask)
                cv2.drawContours(shape_mask, [cnt], 0, 255, 0)
                yellow_pixels = cv2.countNonZero(cv2.bitwise_and(mask_yellow, shape_mask))
                total_pixels = cv2.countNonZero(shape_mask)
                yellow_ratio = yellow_pixels / float(total_pixels + 1)
                
                if yellow_ratio > 0.95:
                    triangle_contours.append(cnt)
                    isValid_Sign = True
        
        # === RECTANGLE DETECTION ===
        if not isValid_Sign and vertices == 4:
            if Extent > 0.75 and ratio > 1.3:
                rectangle_contours.append(cnt)
                isValid_Sign = True
    
    # === RECOGNIZE SIGN CONTENT ===
    detected_signs = infer_sign_content(frame, circle_contours, triangle_contours)
    
    # === DRAW RESULTS ===
    frame = draw_detected_signs(frame, detected_signs)
    
    # Draw rectangles (if you want to show them separately)
    for cnt in rectangle_contours:
        approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(frame, "CHI DAN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # === DISPLAY (Optional - comment out if not needed) ===
   
    cv2.imshow('Detection Result', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break
    
    # Write frame to output video
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video processing complete! Total frames processed: {frame_count}")

