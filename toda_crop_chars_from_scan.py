#!/usr/bin/env python

# MEMO:
# 1年生の原稿用紙は14文字・9行(縦書き)
# 2年生の原稿用紙は20文字・10行(横書き)
# 3年生の原稿用紙は20文字・12行(縦書き)
# 4年生の原稿用紙は20文字・11行(縦書き)
# 5, 6年生の原稿用紙は20文字・15行(縦書き)

# 2024-05-02 

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

def char_from_scan(input_img, save_dir, i):
    file_name = os.path.splitext(os.path.basename(input_img))[0] # xaaa
    writer = file_name.split('_')[0]
    
    # スキャン結果を読み込み
    scan_img = Image.open(input_img)
    width, height = scan_img.size
    
    # 剪定範囲を設定(手書き字がある部分だけ取り出す)
    
    # シート名OCR
    try:
        numberd_img = scan_img.crop((0, 0, 1000, 350))
        number = pytesseract.image_to_string(numberd_img, lang = 'eng', config= \
                                                 '-c tessedit_char_whitelist="0123456789Ss-ー" --user-patterns PATH: /home/abababam1/HandwrittenTextAlign/toda_crop_imgs/userpattern.txt')
        if number == '':
            numberd_img = scan_img.crop((0, height - 300, width, height))
            number = pytesseract.image_to_string(numberd_img, lang = 'eng', config= \
                                                 '-c tessedit_char_whitelist="0123456789Ss-ー" --user-patterns PATH: /home/abababam1/HandwrittenTextAlign/toda_crop_imgs/userpattern.txt')
    except Exception as e:
        print(f"OCR failed: {e}")
        # OCR failed -> 別途手作業で対応
        return None
    
    # OCR結果下処理
    number = re.sub(r"[\n $']", '', number)
    # '-'が抜けてしまった時に対応
    if '-' not in number:
        number = number[:2] + '-' + number[2:]
    # 先頭をSに揃える
    if number[0] != 'S' and len(number) in [4, 5, 6]:
        number = re.sub('^[0-9s]', 'S', number)
        
    print(f"OCR Result: '{number}'")
    #numberd_img.save(f'/home/abababam1/HandwrittenTextAlign/test/number/{number}.png')
    
    # 例) S1-0, grade = 1, paper_number = 0
    if len(number) not in [4,5,6] or number[1] == '-':
        # OCR failed -> 別途手作業で対応
        return None
    grade = input_img.split('/')[4].split('-')[1]# ファイル名から #number[1]
    paper_number = number[3:None]
    
    if grade not in ['1', '2', '3'] or not re.match(r'^[1-9]\d{0,2}$', paper_number):
        # OCR failed -> 別途手作業で対応
        return None
    elif grade in ['1', '2']:
        design_number = (int(paper_number) % 24) or 24
    else:
        design_number = (int(paper_number) % 9) or 9
    

    # シート名ごとの原稿配置を取得
    params = load_genko_params(f'sisha-{grade}-genko-params.json'.format(grade))[design_number - 1]

    #x_threshold,y_threshold,rotation_angle = find_rotation_angle(scan_img) # あまり作用しない
    #rotated_image = scan_img.rotate(rotation_angle, expand=True,fillcolor='white') # 画像を回転
    # crop_box作成
    
    #coordinate, crop_box = find_largest_rectangle(scan_img, params)
    cropbox, angle = find_genko_box(scan_img, params)
    if cropbox is None:
        return i
    #rotated_img = scan_img.rotate(angle, expand=True,fillcolor='white') # 画像を回転
    cropped_img = scan_img.crop(cropbox) # crop の引数は ((left, upper, right, lower))
    #cropped_img.save('/home/abababam1/cropped_result.jpg') #test
    
    # 全ての罫線の位置を取得
    lines = detect_lines(cropped_img, params)
    main_lines, sub_line_start = detect_main_line(lines, params)
    
    # 罫線を白塗り（メインラインのみ）
    image = np.array(cropped_img)
    for line in lines:
        whiteline = 3
        lineadd_img = cv2.line(image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), whiteline)
    rotated_image = Image.fromarray(lineadd_img)
    #cv2.imwrite('/home/abababam1/out.jpg', lineadd_img) #test
    
    # 文字ごとに切り分け
    chars = detect_chars_from_cropbox(scan_img, params, cropbox)
    #chars = detect_chars_crop_box(cropped_img, main_lines, sub_line_start, params, coordinate)
    #chars = detect_chars_crop_box(scan_img, main_lines, sub_line_start, params, coordinate)
    
    # 個々の文字の画像を保存する
    save_path = os.path.join(save_dir, f'{file_name}_{number}')
    os.makedirs(save_path, exist_ok=True)
    for im in chars:
        im_clean = detect_line_in_char(im)
        im_clean.save((save_path)+'/'+'{:05d}.png'.format(i), 'PNG')
        i += 1
    return i

# 原稿配置のパラメータが入ったjsonファイルの読み込み
def load_genko_params(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# 原稿用紙部分の位置を返す関数
def make_crop_box(scan_img, params, grade):
    width, height = scan_img.size

    muki = params['muki']
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']

    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    k = 3 if grade == 3 else 0

    if muki == 'tate':
        upper = (nchars + 3 + k) * char_size_px
        lower = height - k * (char_size_px + sep_size_px)
    elif muki == 'yoko':
        upper = (ncols + 3 + k) * (char_size_px + sep_size_px)
        lower = height - k * (char_size_px + sep_size_px)

    return (0, height - upper, width, lower)

def find_largest_rectangle(pillow_img, params):
    # 読み込み
    width, _ = pillow_img.size
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    #原稿用紙部分の面積
    best_area = (ncols * (char_size_px + sep_size_px)) * (nchars * char_size_px)
    
    # Pillowの画像をNumPy配列に変換（RGB）
    img = np.array(pillow_img)

    # RGBをBGRに変換
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edged = cv2.Canny(gray, 50, 150)

    # 輪郭を見つける
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大面積の四角形を見つける
    min_distance = 2500
    best_rect = None
    best_angle = 0
    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 最小の回転された矩形を取得
        box = cv2.boxPoints(rect)  # 回転された矩形の四角を取得
        #box = np.int0(box)  # 座標を整数に変換
        area = cv2.contourArea(box)
        #angle = 90+rect[2]  # 回転角を設定
        #if area > max_area:
            #max_area = area
            #best_rect = box
            #best_angle = angle
        distance = abs(area - best_area)
        if distance < min_distance:
            min_distance = distance
            best_rect = box
            
    
    if best_rect is not None:
        # バウンディングボックスの座標でクロップ
        x, y, w, h = cv2.boundingRect(best_rect)
        return (x+10, y+10, x+w-10, y+h-10), (0, y - 40, width, y + h + 40)
    
def calculate_angle_yoko(x1, y1, x2, y2):
    if x2 - x1 == 0: # 垂直の場合
        return 0
    slope = (y2 - y1) / (x2 - x1)
    
    # 角度を計算（ラジアン）
    angle_radians = math.atan(slope)
    # 角度を度に変換
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees

def calculate_angle_tate(x1, y1, x2, y2):
    if y2 - y1 == 0: # 平行線の場合
        return 0
    slope = (y2 - y1) / (x2 - x1)
    
    # 角度を計算（ラジアン）
    angle_radians = math.atan(slope)
    # 角度を度に変換
    angle_degrees = math.degrees(angle_radians)
    
    return angle_degrees
    
def check_genko_size_and_angle(hull, genko_1, genko_2):  
    quad = set()
    n = len(hull)
    x_max, x_min = 0, float('inf')
    y_max, y_min = 0, float('inf')
    genko_angle_list = []
    for i in range(n):
        for j in range(i + 1, n):
            ij_width = abs(hull[i][0][0] - hull[j][0][0])
            ij_height = abs(hull[i][0][1] - hull[j][0][1])
            # 横
            if any(abs(ij_width - henn) <= 20 for henn in [genko_1, genko_2]) and ij_height <= 5:
                quad.add(tuple(hull[i][0]))
                quad.add(tuple(hull[j][0]))
                x_max = max(x_max, hull[i][0][0], hull[j][0][0])
                x_min = min(x_min, hull[i][0][0], hull[j][0][0])
                genko_angle = calculate_angle_yoko(hull[i][0][0], hull[j][0][0], hull[i][0][0], hull[j][0][0])
                genko_angle_list.append(genko_angle)
            # 縦
            elif ij_width <= 5 and any(abs(ij_height - henn) <= 20 for henn in [genko_1, genko_2]):
                quad.add(tuple(hull[i][0]))
                quad.add(tuple(hull[j][0]))
                y_max = max(y_max, hull[i][0][1], hull[j][0][1])
                y_min = min(y_min, hull[i][0][1], hull[j][0][1])
                genko_angle = calculate_angle_tate(hull[i][0][0], hull[j][0][0], hull[i][0][0], hull[j][0][0])
                genko_angle_list.append(genko_angle)
    if len(quad) >= 4:
        # 四頂点と右上の座標
        #cropbox = (max(0, x_min + 5), max(0, y_min + 5), min(x_max - 5, 5000), min(y_max - 5, 6000))
        cropbox = (x_min + 5, y_min + 5, x_max - 5, y_max - 5)
        return quad, cropbox, genko_angle_list
    else:
        return quad, None, genko_angle

def find_genko_box(pillow_img, params):
    width, height = pillow_img.size
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    print(char_size_px, sep_size_px)
    
    #原稿用紙部分の面積
    best_area = (ncols * (char_size_px + sep_size_px) + sep_size_px) * (nchars * char_size_px)
    
    # Pillowの画像をNumPy配列に変換（RGB）
    img = np.array(pillow_img)

    # RGBをBGRに変換
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edged = cv2.Canny(gray, 50, 150)

    # 輪郭を見つける
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大面積の四角形を見つける
    min_distance = float('inf')#width * height - best_area
    best_rect = None
    for contour in contours:
        
        # 凸包近似
        hull = cv2.convexHull(contour)
        area = cv2.contourArea(hull)
        
        distance = abs(area - best_area)
        if distance < min_distance:
            min_distance = distance
            best_rect = hull
        
    if best_rect is not None:
        cv2.drawContours(img, [best_rect], -1, (0, 255, 0), 2)  # 緑色の線で描画
        genko_1 = ncols * (char_size_px + sep_size_px) + sep_size_px
        genko_2 = nchars * char_size_px
        quad, cropbox, angle = check_genko_size_and_angle(best_rect, genko_1, genko_2)
        for i in range(len(quad)):
            cv2.circle(img, list(quad)[i], 10, (0, 0, 255), thickness=5, lineType=cv2.LINE_8, shift=0)
        for i in range(len(best_rect)):
            cv2.circle(img, tuple(best_rect[i][0]), 5, (255, 0, 0), thickness=5, lineType=cv2.LINE_8, shift=0)
        cv2.imwrite('/home/abababam1/result_image_.jpg', img)

        if len(quad) >= 4:
            return cropbox, angle
        else:
            print('quadrilaterals Detection error')
            return None, angle
    else:
        print('hull Detection error')
        return None, angle


# 連続した長さのリストを返す関数
def find_continuous_lengths(input_list):
    continuous_lengths = [] 
    current_length = 0

    for num in input_list:
        if num > 0:
            current_length += 1
        else:
            if current_length > 0:
                continuous_lengths.append(current_length)
            current_length = 0

    if current_length > 0:
        continuous_lengths.append(current_length)

    return continuous_lengths

# ヒストグラムから閾値を見つける関数=線と判定する下限を見つける
def find_threshold(hist,n,threshold_0=0.3):
    hist_max = np.max(hist)
    cont = find_continuous_lengths((hist > hist_max * threshold_0) * hist_max) # 連続した正の数値の長さを見つける
    threshold_list = [threshold_0]
    i = 0
    threshold = threshold_0
    
    while threshold >= 0.05 and len(cont) < n :
        threshold = threshold - 0.01
        cont = find_continuous_lengths((hist > hist_max * threshold) * hist_max)
        threshold_list.append(threshold)
        i += 1
        
    threshold = threshold_list[i-1]
    
    return threshold,(hist > hist_max * threshold) * hist_max

# 角度を見つける
def find_rotation_angle(cropped_img):
    threshold_list = []
    rotation_angle_list = []
    
    for a in range(31):
        rotation_angle = -0.5+0.05*a # 時計回り0.5度から反時計回り1度の範囲内で
        rotated_image = cropped_img.rotate(rotation_angle, expand=True,fillcolor='white') # 画像を回転

        scan = np.asarray(rotated_image)

        x_hist = np.mean(255 - scan, axis=0)
        y_hist = np.mean(255 - scan, axis=1)

        #x_threshold = find_threshold(x_hist,20)[0] # 1年
        #y_threshold = find_threshold(y_hist,16)[0] # 1年
        #x_threshold = find_threshold(x_hist,22)[0] # 2年
        #y_threshold = find_threshold(y_hist,22)[0] # 2年
        #x_threshold = find_threshold(x_hist,27)[0] # 3年
        #y_threshold = find_threshold(y_hist,22)[0] # 3年
        #x_threshold = find_threshold(x_hist,25)[0] # 4年
        #y_threshold = find_threshold(y_hist,22)[0] # 4年
        x_threshold = find_threshold(x_hist,17)[0] # 5,6年
        y_threshold = find_threshold(y_hist,22)[0] # 5,6年
        
        if abs(x_threshold-y_threshold) < 0.1:
            threshold_list.append((x_threshold,y_threshold)) # 閾値を見つける
            rotation_angle_list.append(rotation_angle)
    
    sums = [sum(pair) for pair in threshold_list]
    min_index = sums.index(min(sums))
    
    x_threshold,y_threshold = threshold_list[min_index]
    rotation_angle = rotation_angle_list[min_index]
    return x_threshold,y_threshold,rotation_angle


def detect_lines(cropped_img, params):
    img = np.array(cropped_img)
    #img = cropped_img[:,:,0]
    height, width = img.shape
    judge_img = cv2.bitwise_not(img)
    
    char_width = params['char_width']
    sep_width = params['sep_width']
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    minlength = char_size_px * 5#0.8 * 5
    gap = 20
    
    # 検出しやすくするために二値化
    th, judge_img = cv2.threshold(judge_img, 1, 255, cv2.THRESH_BINARY)
    
    lines = []
    lines = cv2.HoughLinesP(judge_img, rho=1, theta=np.pi/360, threshold=100, minLineLength=minlength, maxLineGap=gap)
    
    return lines

def detect_main_line(lines, params):
    muki = params['muki']
    
    col_ranges = []
    row_ranges = []
    sub_line_start = []
    
    x1_pre = 0
    y1_pre = 0
    
    if muki == 'tate':# and x1 == x2:
        
        # 縦の場合、x座標を比較して近い線を除外
        lines = sorted(lines, key=lambda x: x[0][0])  # x座標に基づいて線をソート
        previous_x1 = None
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if previous_x1 is None or abs(previous_x1 - x1) > 50: # ほぼ同じ縦線消去
                col_ranges.append(x1)
                sub_line_start.append(min(y1, y2)) # 上から(右から)
            previous_x1 = x1
        col_ranges.sort(reverse = True) # 大きい順
        sub_line_start.sort()
        return col_ranges, sub_line_start[0]
    
    elif muki == 'yoko':# and y1 == y2:
        
        # 横の場合、y座標を比較して近い線を除外
        lines = sorted(lines, key=lambda x: x[0][1])  # y座標に基づいて線をソート
        previous_y1 = None
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if previous_y1 is None or abs(previous_y1 - y1) > 50: # ほぼ同じ横線消去
                row_ranges.append(y1)
                sub_line_start.append(min(x1, x2)) # 左から
            previous_y1 = y1
        row_ranges.sort()
        sub_line_start.sort()
        return row_ranges, sub_line_start[0]
    
def detect_chars_crop_box(rotated_image, main_lines, sub_line_start, params, coordinate):
    try:
        print(main_lines[0])
    except ZeroDivisionError as e:
        print(e)
        print(type(e))
    
    muki = params['muki']
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    chars = []
    
    if muki == 'tate': # 右から
        for cols in range(int(ncols)):
            for n in range(int(nchars)):
                #x_right = main_lines[1] - cols * (char_size_px + sep_size_px)
                x_right = coordinate[2] - sep_size_px - cols * (char_size_px + sep_size_px)
                #y_upper = sub_line_start + n * char_size_px
                y_upper = coordinate[1] + n * char_size_px
                
                x_left = x_right - char_size_px
                y_lower = y_upper + char_size_px
                char = rotated_image.crop((x_left, y_upper, x_right, y_lower))
                chars.append(char)
    elif muki == 'yoko': # 左から
        for cols in range(int(ncols)):
            for n in range(int(nchars)):
                #x_left = sub_line_start + n * char_size_px
                x_left = coordinate[0] + n * char_size_px
                #y_upper = main_lines[1] + cols * (char_size_px + sep_size_px)
                y_upper = coordinate[1] + sep_size_px + cols * (char_size_px + sep_size_px)
                
                x_right = x_left + char_size_px
                y_lower = y_upper + char_size_px
                char = rotated_image.crop((x_left, y_upper, x_right, y_lower))
                chars.append(char)
    return chars

def detect_chars_from_cropbox(img, params, cropbox):
    
    muki = params['muki']
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    chars = []
    
    if muki == 'tate': # 右から
        for cols in range(int(ncols)):
            for n in range(int(nchars)):
                x_right = cropbox[2] - sep_size_px - cols * (char_size_px + sep_size_px)
                y_upper = cropbox[1] + n * char_size_px
                
                x_left = x_right - char_size_px
                y_lower = y_upper + char_size_px
                char = img.crop((x_left+2, y_upper+2, x_right-2, y_lower-2))
                chars.append(char)
    elif muki == 'yoko': # 左から
        for cols in range(int(ncols)):
            for n in range(int(nchars)):
                x_left = cropbox[0] + n * char_size_px
                y_upper = cropbox[1] + sep_size_px + cols * (char_size_px + sep_size_px)
                
                x_right = x_left + char_size_px
                y_lower = y_upper + char_size_px
                char = img.crop((x_left+2, y_upper+2, x_right-2, y_lower-2))
                chars.append(char)
    return chars

def detect_line_in_char(char_img):
    img = np.array(char_img)
    height, width = img.shape
    judge_img = cv2.bitwise_not(img)

    minlength = width * 0.95
    gap = 5

    # 検出しやすくするために二値化
    th, judge_img = cv2.threshold(judge_img, 1, 255, cv2.THRESH_BINARY)

    lines = []
    lines = cv2.HoughLinesP(judge_img, rho=1, theta=np.pi/360, threshold=100, minLineLength=minlength, maxLineGap=gap)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y1 - y2) < 10 or abs(x1 - x2) < 10:
                whiteline = 3
                img = cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), whiteline)
    return Image.fromarray(img)
    

#---
# main
#---

'''
#'/data/matuzaki/sisha-split/小学校-学年-クラス.tif/画像ファイル.tif'
draft_dir = '/data/matuzaki/sisha-orgs'
save_dir = '/data2/abababam1/HandwrittenTextAlign/toda_char_imgs/'
for path in glob(f"{draft_dir}/**/**"):
    grade = path.split('/')[-2]
    grade = grade.split('-')[0]
    char_from_scan(path, save_dir, grade)
'''
if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <img-file> <grade> <save-dir>", file=sys.stderr)
    sys.exit(1)

if sys.argv[2] not in "123456":
    print("grade must be in 1, ..., 6", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
grade = int(sys.argv[2])
save_dir = sys.argv[3]

#char_from_scan(path, save_dir, i = 1)


