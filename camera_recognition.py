"""
手语数字识别 - 摄像头实时识别模块（高精度版）

功能说明：
    使用训练好的 CNN 模型实时识别摄像头中的手语数字手势
    支持多种图像处理技术提高识别准确率

核心特性：
    1. 多颜色空间肤色检测融合（YCrCb + HSV + RGB）
    2. CLAHE 对比度限制自适应直方图均衡化
    3. 自适应背景减法（帧差法）动态背景更新
    4. Canny 边缘检测辅助识别
    5. 置信度加权投票机制
    6. 多帧稳定确认

使用方法：
    python camera_recognition.py
    
    快捷键：
        q - 退出程序
        s - 保存截图
        d - 切换调试模式（显示 Top-3 预测结果）
        r - 重置背景模型

依赖库：
    - cv2 (OpenCV): 图像处理和摄像头操作
    - numpy: 数值计算
    - tensorflow/keras: 加载预训练模型
    - collections: 滑动窗口数据结构
    - time: 性能统计

注意：
    需要先运行 main.py 训练模型生成 best_model.keras
"""

import cv2
import numpy as np
from tensorflow import keras
import collections
import time


def adjust_gamma(image, gamma=1.0):
    """
    Gamma 校正 - 用于光照补偿
    
    Gamma 校正是一种非线性变换，用于调整图像的整体亮度：
    - gamma < 1: 图像变亮（暗场景）
    - gamma > 1: 图像变暗（亮场景）
    
    参数：
        image (np.ndarray): 输入灰度图像
        gamma (float): Gamma 值，默认为 1.0（无校正）
        
    返回：
        np.ndarray: Gamma 校正后的图像
    """
    inv_gamma = 1.0 / gamma
    # 创建查找表，预计算每个像素值的校正结果
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # 使用 LUT（查找表）快速应用校正
    return cv2.LUT(image, table)


def apply_clahe(image):
    """
    CLAHE（对比度限制自适应直方图均衡化）
    
    与普通直方图均衡化不同，CLAHE 将图像分成多个小区域（tiles），
    对每个区域单独进行直方图均衡化，并限制对比度防止过度增强噪声。
    
    参数：
        image (np.ndarray): 输入灰度图像
        
    返回：
        np.ndarray: 对比度增强后的图像
    """
    # 创建 CLAHE 对象
    # clipLimit: 对比度限制阈值，默认为 40
    # tileGridSize: 网格大小，默认为 8x8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def detect_skin_color_multi_space(frame):
    """
    多颜色空间融合肤色检测
    
    融合 YCrCb、HSV、RGB 三种颜色空间的肤色检测结果，提高检测准确率。
    不同颜色空间对光照变化的鲁棒性不同，融合后可以互补。
    
    参数：
        frame (np.ndarray): 输入 BGR 彩色图像
        
    返回：
        np.ndarray: 肤色掩码（255 表示肤色区域，0 表示非肤色区域）
    """
    # YCrCb 颜色空间 - 对肤色检测效果较好
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # HSV 颜色空间 - 对光照变化较鲁棒
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 48, 80], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # RGB 颜色空间 - 直接使用颜色范围
    lower_rgb = np.array([95, 40, 20], dtype=np.uint8)
    upper_rgb = np.array([255, 255, 255], dtype=np.uint8)
    mask_rgb = cv2.inRange(frame, lower_rgb, upper_rgb)
    
    # 融合三个颜色空间的掩码（使用 OR 操作）
    mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
    mask = cv2.bitwise_or(mask, mask_rgb)
    
    # 形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # 开运算：先腐蚀后膨胀，去除小的噪声点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    # 闭运算：先膨胀后腐蚀，填充孔洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # 填充轮廓内部的孔洞
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.fillPoly(mask, [cnt], 255)
    
    return mask


def background_subtraction(frame, bg_model, alpha=0.01):
    """
    自适应背景减法
    
    使用累积加权平均法动态更新背景模型，适用于缓慢变化的背景。
    
    参数：
        frame (np.ndarray): 当前帧图像（BGR）
        bg_model (np.ndarray): 背景模型（浮点型）
        alpha (float): 学习率，控制背景更新速度（0-1）
        
    返回：
        fg_mask (np.ndarray): 前景掩码
        bg_model (np.ndarray): 更新后的背景模型
    """
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 初始化背景模型（如果为空）
    if bg_model is None:
        bg_model = gray.astype(np.float32)
    
    # 使用累积加权平均更新背景模型
    # bg_model = (1-alpha) * bg_model + alpha * gray
    cv2.accumulateWeighted(gray, bg_model, alpha)
    
    # 计算当前帧与背景的差分
    diff = cv2.absdiff(gray, bg_model.astype(np.uint8))
    # 二值化得到前景掩码
    _, fg_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    return fg_mask, bg_model


def preprocess_gesture_enhanced(roi, input_shape, bg_model=None):
    """
    增强版手势预处理流程
    
    包含 14 个处理步骤，确保输入到模型的图像与训练数据保持一致。
    
    参数：
        roi (np.ndarray): 手势区域图像（BGR）
        input_shape (tuple): 模型输入形状 (高度, 宽度, 通道)
        bg_model (np.ndarray): 背景模型（可选）
        
    返回：
        img_array (np.ndarray): 预处理后的图像数组（适合模型输入）
        thresh (np.ndarray): 二值化后的图像（用于显示）
        blurred (np.ndarray): 模糊后的图像（用于显示）
        edges (np.ndarray): 边缘检测结果（用于显示）
    """
    # 1. 初始灰度化 - 将彩色图像转换为灰度图像
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Gamma 校正（自适应）- 根据平均亮度动态调整
    avg_brightness = np.mean(gray)
    if avg_brightness < 70:
        gray = adjust_gamma(gray, gamma=2.0)  # 暗场景：提高亮度
    elif avg_brightness < 100:
        gray = adjust_gamma(gray, gamma=1.5)  # 稍暗：轻微提高
    elif avg_brightness > 220:
        gray = adjust_gamma(gray, gamma=0.5)  # 亮场景：降低亮度
    elif avg_brightness > 180:
        gray = adjust_gamma(gray, gamma=0.7)  # 稍亮：轻微降低
    
    # 3. CLAHE 对比度增强 - 提高局部对比度
    gray = apply_clahe(gray)
    
    # 4. 肤色检测 - 提取手部区域
    skin_mask = detect_skin_color_multi_space(roi)
    skin_result = cv2.bitwise_and(gray, gray, mask=skin_mask)
    
    # 5. 背景减法（可选）- 进一步去除背景干扰
    if bg_model is not None:
        fg_mask, _ = background_subtraction(roi, bg_model)
        combined_mask = cv2.bitwise_and(skin_mask, fg_mask)
        skin_result = cv2.bitwise_and(gray, gray, mask=combined_mask)
    else:
        combined_mask = skin_mask
    
    # 6. 高斯模糊降噪 - 去除高频噪声
    blurred = cv2.GaussianBlur(skin_result, (5, 5), 0)
    
    # 7. Canny 边缘检测 - 提取手势轮廓
    edges = cv2.Canny(blurred, 50, 150)
    
    # 8. 自适应阈值二值化 - 将灰度图转为黑白图
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 9. 形态学操作 - 去除噪声和填充孔洞
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)       # 开运算去噪
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_small, iterations=2)  # 膨胀
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large)      # 闭运算填充
    
    # 10. 边缘融合 - 将 Canny 边缘检测结果与二值化结果合并
    thresh = cv2.bitwise_or(thresh, edges)
    
    # 11. 提取最大轮廓 - 只保留最大的连通区域（假设手势是最大的物体）
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(thresh)
        cv2.drawContours(mask, [max_contour], 0, 255, -1)
        thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # 12. 调整大小 - 调整为模型输入尺寸
    target_size = (input_shape[1], input_shape[0])
    img_resized = cv2.resize(thresh, target_size, interpolation=cv2.INTER_AREA)
    
    # 13. 归一化 - 与训练数据保持一致（像素值范围 [0, 1]）
    img_array = img_resized.astype('float32') / 255.0
    
    # 14. 添加通道维度 - 适配模型输入格式
    if len(input_shape) == 3:
        img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批量维度
    
    return img_array, thresh, blurred, edges


def camera_recognition(model_path='best_model.keras'):
    """
    高精度摄像头实时识别手语数字
    
    主函数，负责：
    1. 加载预训练模型
    2. 初始化摄像头
    3. 实时获取帧并处理
    4. 使用模型进行预测
    5. 显示结果和调试信息
    
    参数：
        model_path (str): 模型文件路径，默认为 'best_model.keras'
    """
    import os
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件 '{model_path}' 不存在！")
        print("请先运行 'python main.py' 训练模型。")
        return
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    input_shape = model.input_shape[1:]  # 获取输入形状（忽略批量维度）
    print(f"模型输入形状: {input_shape}")
    print(f"模型输出类别数: {model.output_shape[1]}")
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头！")
        print("请检查摄像头是否已连接并授权。")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 设置宽度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置高度
    cap.set(cv2.CAP_PROP_FPS, 30)            # 设置帧率
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光模式
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)        # 设置曝光值（负值表示降低曝光）
    
    # 打印使用说明
    print("=" * 70)
    print("Camera started! (High Precision Version)")
    print("Put your hand in the green box")
    print("Press 'q' to quit | 's' to save screenshot")
    print("Press 'd' to toggle debug mode | 'r' to reset background")
    print("=" * 70)
    
    # 手势检测区域设置
    roi_size = 224  # 手势区域大小（224x224）
    min_confidence = 78  # 最小置信度阈值（%）
    stable_threshold = 4  # 需要连续稳定的帧数
    
    # 滑动窗口预测平滑（最近8帧）
    pred_history = collections.deque(maxlen=8)  # 存储预测概率分布
    digit_history = collections.deque(maxlen=8)  # 存储预测数字
    
    # 连续稳定计数
    stable_count = 0       # 当前稳定帧数
    last_stable_digit = -1 # 上一个稳定的数字
    confirmed_digit = -1   # 确认的数字
    
    # 背景模型
    bg_model = None          # 背景模型（浮点型）
    bg_initialized = False   # 背景模型是否已初始化
    bg_init_frame_count = 0  # 初始化帧计数
    
    # 调试模式
    debug_mode = False
    
    # 性能统计
    frame_count = 0
    start_time = time.time()
    total_pred_time = 0
    
    # 主循环
    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break
        
        frame_count += 1
        
        h, w = frame.shape[:2]
        
        # 计算手势区域位置（画面中心）
        x1 = (w - roi_size) // 2
        y1 = (h - roi_size) // 2
        x2 = x1 + roi_size
        y2 = y1 + roi_size
        
        # 绘制手势检测框
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Gesture Area', (x1, y1 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 提取手势区域
        roi = frame[y1:y2, x1:x2]
        
        # 初始化背景模型（前10帧）
        if not bg_initialized:
            if bg_model is None:
                bg_model = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                cv2.accumulateWeighted(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), bg_model, 0.1)
            bg_init_frame_count += 1
            if bg_init_frame_count >= 10:
                bg_initialized = True
                print("背景模型初始化完成")
            # 显示初始化提示
            cv2.putText(frame, 'Initializing background...', (20, h - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # 预处理手势图像
        if bg_initialized:
            img_array, processed_img, blurred_img, edge_img = preprocess_gesture_enhanced(
                roi, input_shape, bg_model
            )
        else:
            # 背景未初始化时使用简化预处理
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(gray, (input_shape[1], input_shape[0]))
            img_array = img_resized.astype('float32') / 255.0
            if len(input_shape) == 3:
                img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            processed_img = img_resized
            blurred_img = img_resized
            edge_img = img_resized
        
        # 使用模型进行预测
        pred_start = time.time()
        pred = model.predict(img_array, verbose=0)
        pred_time = time.time() - pred_start
        total_pred_time += pred_time
        
        # 获取预测结果
        digit = np.argmax(pred[0])
        confidence = pred[0][digit] * 100
        
        # 获取 Top-3 预测结果（用于调试）
        top3_indices = np.argsort(pred[0])[-3:][::-1]
        top3_probs = pred[0][top3_indices] * 100
        
        # 添加到历史记录
        pred_history.append(pred[0])
        digit_history.append(digit)
        
        # 置信度加权投票（越新的帧权重越高）
        if len(pred_history) >= 3:
            weighted_pred = np.zeros_like(pred[0])
            weights = np.linspace(0.5, 1.0, len(pred_history))  # 权重从 0.5 到 1.0
            for i, p in enumerate(pred_history):
                weighted_pred += p * weights[i]
            weighted_pred /= np.sum(weights)  # 归一化
            avg_digit = np.argmax(weighted_pred)
            avg_confidence = weighted_pred[avg_digit] * 100
        else:
            avg_digit = digit
            avg_confidence = confidence
        
        # 稳定性检测
        if avg_digit == last_stable_digit:
            stable_count += 1
        else:
            stable_count = 0
            last_stable_digit = avg_digit
        
        is_stable = stable_count >= stable_threshold
        
        # 更新确认的数字（只有稳定且置信度足够时才确认）
        if avg_confidence > min_confidence and is_stable:
            confirmed_digit = avg_digit
        
        # 确定显示状态和颜色
        if confirmed_digit != -1 and avg_confidence > min_confidence:
            text = f'Digit: {confirmed_digit}'
            confidence_text = f'Confidence: {avg_confidence:.1f}%'
            color = (0, 255, 0)      # 绿色
            bg_color = (0, 100, 0)    # 深绿色背景
            status_text = 'STABLE'     # 稳定状态
        elif avg_confidence > 60:
            text = f'Detecting: {avg_digit}'
            confidence_text = f'Confidence: {avg_confidence:.1f}%'
            color = (0, 255, 255)    # 黄色
            bg_color = (0, 100, 100)  # 深黄色背景
            status_text = 'UNSTABLE'   # 不稳定状态
        else:
            text = 'Recognizing...'
            confidence_text = f'Confidence: {avg_confidence:.1f}%'
            color = (0, 0, 255)       # 红色
            bg_color = (0, 0, 100)     # 深红色背景
            status_text = 'WAITING'    # 等待状态
        
        # 绘制结果显示区域背景
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        cv2.rectangle(frame, (10, 10), (10 + text_size[0] + 20, 10 + text_size[1] + 60), bg_color, -1)
        
        # 显示主结果
        cv2.putText(frame, text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, confidence_text, (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示稳定性指示
        stability_icon = '[' + ('=' * min(stable_count, 5)) + (' ' * (5 - min(stable_count, 5))) + ']'
        cv2.putText(frame, f'Stable: {stability_icon}', (20, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示 Top-3 结果（调试模式）
        if debug_mode:
            for i, (idx, prob) in enumerate(zip(top3_indices, top3_probs)):
                debug_text = f'{i+1}. {idx}: {prob:.1f}%'
                cv2.putText(frame, debug_text, (w - 150, 30 + i * 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示帧率和延迟信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        avg_pred_time = (total_pred_time / frame_count) * 1000 if frame_count > 0 else 0
        cv2.putText(frame, f'FPS: {int(fps)}', (w - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f'Latency: {avg_pred_time:.1f}ms', (w - 130, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示处理后的图像（小窗口）
        if bg_initialized:
            processed_img_resized = cv2.resize(processed_img, (60, 60))
            blurred_resized = cv2.resize(blurred_img, (60, 60))
            edge_resized = cv2.resize(edge_img, (60, 60))
            
            processed_img_color = cv2.cvtColor(processed_img_resized, cv2.COLOR_GRAY2BGR)
            blurred_color = cv2.cvtColor(blurred_resized, cv2.COLOR_GRAY2BGR)
            edge_color = cv2.cvtColor(edge_resized, cv2.COLOR_GRAY2BGR)
            
            # 放置在右下角
            pos_y = h - 80
            frame[pos_y:pos_y+60, w-70:w-10] = processed_img_color
            frame[pos_y:pos_y+60, w-140:w-80] = blurred_color
            frame[pos_y:pos_y+60, w-210:w-150] = edge_color
            
            cv2.putText(frame, 'Final', (w-70, pos_y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, 'Blurred', (w-140, pos_y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, 'Edges', (w-210, pos_y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 显示画面
        cv2.imshow('Sign Language Digit Recognition', frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 退出程序
            break
        elif key == ord('s'):
            # 保存截图
            screenshot = frame.copy()
            cv2.imwrite(f'screenshot_{cv2.getTickCount()}.png', screenshot)
            print("截图已保存")
        elif key == ord('d'):
            # 切换调试模式
            debug_mode = not debug_mode
            print(f"调试模式: {'开启' if debug_mode else '关闭'}")
        elif key == ord('r'):
            # 重置背景模型
            bg_model = None
            bg_initialized = False
            bg_init_frame_count = 0
            confirmed_digit = -1
            print("背景模型已重置，正在重新初始化...")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\nCamera closed")


if __name__ == '__main__':
    camera_recognition()
