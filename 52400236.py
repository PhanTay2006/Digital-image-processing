import cv2
import numpy as np
import os
import math
#   I. TẢI VÀ XỬ LÝ TEMPLATE
def load_templates(template_folder="template"):
    """Tải tất cả ảnh mẫu (template) ở dạng Grayscale."""
    templates = {}
    # Đảm bảo thư mục 'template' tồn tại trước khi chạy
    if not os.path.exists(template_folder):
        print(f"Lỗi: Thư mục '{template_folder}' không tồn tại. Vui lòng tạo thư mục này và thêm các file .png mẫu.")
        return templates       
    for file in os.listdir(template_folder):
        if file.endswith(".png") :
            path = os.path.join(template_folder, file)
            # Tải ảnh template ở dạng Grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                 name = file.split(".")[0]
                 templates[name] = img
    return templates
def match_template(crop, templates_subset, sign_group):
    """
    Thực hiện Template Matching Đa tỉ lệ (Multi-Scale TM) 
    để nhận dạng ký hiệu bên trong biển báo.
    """
    # Chuyển ảnh ROI về Grayscale
    crop_gray_orig = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_gray_orig = cv2.equalizeHist(crop_gray_orig)
    best_score = 0
    best_name = None
    # Thử so khớp ở 3 tỉ lệ: 64x64, 80x80 (gốc), 96x96
    scales = [64, 80, 96] 
    MIN_MATCH_THRESHOLD = 0.30 # Ngưỡng so khớp tối thiểu
    for name, temp_orig in templates_subset.items():
        for size in scales:     
            # 1. Chuẩn hóa ROI về kích thước hiện tại (size x size)
            if crop_gray_orig.shape[0] < 10 or crop_gray_orig.shape[1] < 10:
                continue
            try:
                crop_scaled = cv2.resize(crop_gray_orig, (size, size))
                # Áp dụng làm mờ nhẹ để giảm nhiễu trước khi so khớp
                crop_scaled = cv2.GaussianBlur(crop_scaled, (3, 3), 0)            
                # 2. Chuẩn hóa Template về kích thước hiện tại
                temp_scaled = cv2.resize(temp_orig, (size, size))
            except cv2.error:
                continue
            # Thực hiện so khớp mẫu
            res = cv2.matchTemplate(crop_scaled, temp_scaled, cv2.TM_CCOEFF_NORMED)
            score = res.max()
            if score > best_score:
                best_score = score
                best_name = name  
    # Trả về tên nếu điểm số vượt qua ngưỡng
    return best_name if best_score > MIN_MATCH_THRESHOLD else None
#   II. PHÁT HIỆN VÀ NHẬN DẠNG TRONG VIDEO
def process_video(input_file, output_file, templates=None):
    """Xử lý từng khung hình của video."""
    if templates is None:
        templates = load_templates("template")         
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print("Cannot open video:", input_file)
        return
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break      
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # A. COLOR FILTER (GIỮ NGUYÊN)
        lower_red1, upper_red1 = np.array([0, 40, 50]), np.array([10, 255, 250])
        lower_red2, upper_red2 = np.array([127, 14, 40]), np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),cv2.inRange(hsv, lower_red2, upper_red2))
        v = hsv[:, :, 2]
        mask_bright = cv2.inRange(v, 10, 255)
        mask_red = cv2.bitwise_and(mask_red, mask_bright)
        lower_blue, upper_blue = np.array([102, 120, 80]), np.array([117, 255, 240])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        lower_yellow = np.array([18, 120, 120])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)       
        v = hsv[:, :, 2]
        mask_bright = cv2.inRange(v, 130, 255)
        mask_blue = cv2.bitwise_and(mask_blue, mask_bright)       
        mask = cv2.bitwise_or(mask_red, mask_blue, mask_yellow)
        s = hsv[:, :, 1]
        mask_s = cv2.inRange(s, 60, 255)
        mask = cv2.bitwise_and(mask, mask_s)
        # ===== Clean mask & Edge Detection (GIỮ NGUYÊN) =====
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.medianBlur(mask, 5)
        blurred = cv2.bilateralFilter(mask, 7, 50, 50)
        edges = cv2.Canny(blurred, 40, 100)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges_filled = cv2.dilate(edges_closed, kernel, iterations=1)
        final_mask = cv2.bitwise_or(mask, edges_filled)
        # B. CONTOUR VÀ LỌC HÌNH HỌC (GIỮ NGUYÊN)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 40000:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Lọc vị trí, kích thước, tỉ lệ (GIỮ NGUYÊN)
            if y > height/2 or x < (1/6) * width:
                continue
            aspect_ratio = w / float(h)
            if aspect_ratio < 0.3 or aspect_ratio > 1.8:
                continue
            if not (0.02 * width < w < 0.3 * width and 0.02 * height < h < 0.3 * height):
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.027 * peri, True)
            num_vertices = len(approx)
            circularity = 4 * np.pi * area / (peri * peri) if peri != 0 else 0
            valid_shape = False
            sign_shape_type = ""
            # --- Logic lọc hình dạng cũ của bạn ---
            if circularity > 0.8 and num_vertices > 5:
                valid_shape = True 
                sign_shape_type = "circle"
            elif num_vertices == 3 :
                valid_shape = True 
                sign_shape_type = "triangle"
            elif num_vertices == 4:
                continue # Loại bỏ hình vuông
            if not valid_shape:
                continue
            # C. PHÂN NHÓM SƠ BỘ (FIXED LOGIC)
            roi = frame[y:y+h, x:x+w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)           
            # Tính Color Ratio (Dùng để phân nhóm)
            mask_y = cv2.inRange(roi_hsv, np.array([18,70,70]), np.array([40,255,255]))
            yellow_ratio = np.sum(mask_y>0)/(w*h)
            mask_r = cv2.inRange(roi_hsv, np.array([0,70,70]), np.array([10,255,255])) | \
                     cv2.inRange(roi_hsv, np.array([170,70,70]), np.array([180,255,255]))
            red_ratio = np.sum(mask_r>0)/(w*h)
            mask_b = cv2.inRange(roi_hsv, np.array([100,120,80]), np.array([130,255,255]))
            blue_ratio = np.sum(mask_b>0)/(w*h)
            sign_group = "unknown"
            template_subset = templates
            # 1️. Biển TAM GIÁC VÀNG (Cảnh báo)
            if sign_shape_type == "triangle" and yellow_ratio > 0.05:
                sign_group = "warning"
                template_subset = {k:v for k,v in templates.items() if "CB_" in k }                
            # 2️. Biển HÌNH TRÒN
            elif sign_shape_type == "circle":
                # Biển Cấm (Có màu Đỏ rõ ràng)
                if red_ratio > 0.15: 
                    sign_group = "prohibition_red"
                    template_subset = {k:v for k,v in templates.items() if "No_" in k or "Yield_" in k}                    
                # Biển Hiệu lệnh/Chỉ dẫn (Chỉ cần có màu xanh, đỏ thấp)
                elif blue_ratio > 0.05:
                    sign_group = "regulatory_blue"
                    # Mặc định: cho phép tất cả các biển xanh
                    template_subset = {k:v for k,v in templates.items() 
                                    if "No_parking" in k or "No_stopping" in k or "Keep_Right" in k}
                    roi_center_x = x + w/2
                    if roi_center_x > (2/3) * width:
                        template_subset = {k:v for k,v in templates.items() 
                                        if "No_parking" in k or "No_stopping" in k}
                else: 
                    template_subset = templates 
            # D. TEMPLATE MATCHING (NHẬN DẠNG)
            name = match_template(roi, template_subset, sign_group)
            # Vẽ kết quả
            if name is not None:
                # Ổn định hóa tên hiển thị
                display_name = name               
                cv2.putText(frame, display_name, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,255,255), 2)            
            # Luôn vẽ khung cho biển báo được phát hiện
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        # Thông tin sinh viên
        text = "524H0068_524H0192"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3, cv2.LINE_AA)
        # Hiển thị và ghi video
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        frame_resized = cv2.resize(frame, None, fx=0.5, fy=0.5)
        #final = cv2.convertScaleAbs(final_mask, alpha=1.1, beta=10)
        #final_mask_resized = cv2.resize(final, None, fx=0.5, fy=0.5)
        cv2.imshow('Traffic Sign Detection - Final', frame_resized)
        #cv2.imshow("DEBUG - Mask (Grayscale)", final_mask_resized)
        out.write(frame)
        key = cv2.waitKey(28) & 0xFF
        if key == ord('q') or key == 27:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished:", output_file)
# III. THỰC THI CHƯƠNG TRÌNH
if __name__ == "__main__":
    templates = load_templates("template")
    # Xử lý video 1
    process_video("video1.mp4", "52400236_v1.mp4", templates)    
    # Xử lý video 2
    process_video("video2.mp4", "52400236_v2.mp4", templates)