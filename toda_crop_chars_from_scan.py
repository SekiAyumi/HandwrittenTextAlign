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
    #print(writer)
    
    # スキャン結果を読み込み
    scan_img = Image.open(input_img)
    width, height = scan_img.size
    # scan_img = scan_img.resize((width*6, height*6)) # 拡大
    
    # 剪定範囲を設定(手書き字がある部分だけ取り出す)
    
    # シート名OCR
    tools = pyocr.get_available_tools()
    tool = tools[0]
    
    try:
        numberd_img = scan_img.crop((0, 0, 1000, 350))
        #number = tool.image_to_string(numberd_img, lang='eng', builder=pyocr.builders.DigitBuilder(tesseract_layout=8))
        #numberd_img = ImageOps.invert(numberd_img.convert('L'))
        number = pytesseract.image_to_string(numberd_img, lang = 'eng', config= \
                                                 '-c tessedit_char_whitelist="0123456789Ss-ー" --user-patterns PATH: /home/abababam1/HandwrittenTextAlign/toda_crop_imgs/userpattern.txt')
        if number == '':
            numberd_img = scan_img.crop((0, height - 300, width, height))
            #number = tool.image_to_string(numberd_img, lang='eng', builder=pyocr.builders.DigitBuilder(tesseract_layout=8))
            #numberd_img = ImageOps.invert(numberd_img.convert('L'))
            number = pytesseract.image_to_string(numberd_img, lang = 'eng', config= \
                                                 '-c tessedit_char_whitelist="0123456789Ss-ー" --user-patterns PATH: /home/abababam1/HandwrittenTextAlign/toda_crop_imgs/userpattern.txt')
    except Exception as e:
        print(f"OCR failed: {e}")
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
    grade = number[1]
    paper_number = number[3:-1]
    
    if grade not in ['1', '2', '3'] or len(number) not in [4,5,6] or not re.match(r'^\d+$', paper_number):
        # OCR failed -> 別途手作業で対応
        return None
    elif grade in ['1', '2']:
        design_number = (int(paper_number) % 24) or 24
    else:
        design_number = (int(paper_number) % 9) or 9
    

    # シート名ごとの原稿配置を取得
    params = load_genko_params(f'sisha-{grade}-genko-params.json'.format(grade))[design_number - 1]

    # crop_box作成
    
    crop_box = make_crop_box(scan_img, params, grade)
    cropped_img = scan_img.crop(crop_box) # crop の引数は ((left, upper, right, lower))
    
    x_threshold,y_threshold,rotation_angle = find_rotation_angle(cropped_img)
    
    rotated_image = cropped_img.rotate(rotation_angle, expand=True,fillcolor='white') # 画像を回転
    
    scan = np.asarray(rotated_image)
    x_hist = np.mean(255 - scan, axis=0)
    y_hist = np.mean(255 - scan, axis=1)
    
    # 全ての罫線の位置を取得
    lines = detect_lines(rotated_image, params)
    main_lines, sub_line_start = detect_main_line(lines, params)
    print(main_lines[0:2])
    
    # 罫線を白塗り（縦のみ、横は無理だった）
    image = np.array(rotated_image)
    for line in lines:
        whiteline = 3
        lineadd_img = cv2.line(image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), whiteline)
    rotated_image = Image.fromarray(lineadd_img)
    #cv2.imwrite('/home/abababam1/out.jpg', lineadd_img) #test
    
    # 文字ごとに切り分け
    chars = detect_chars_cropbox(rotated_image, main_lines, sub_line_start, params)
    
    # 個々の文字の画像を保存する
    save_path = os.path.join(save_dir, f'{file_name}_{number}')
    os.makedirs(save_path, exist_ok=True)
    for im in chars:
        im = detect_line_in_char(im)
        im.save((save_path)+'/'+'{:05d}.png'.format(i), 'PNG')
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
        upper = (nchars + 2 + k) * char_size_px
        lower = height - k * (char_size_px + sep_size_px)
    elif muki == 'yoko':
        upper = (ncols + 2 + k) * (char_size_px + sep_size_px)
        lower = height - k * (char_size_px + sep_size_px)

    return (0, height - upper, width, lower)

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
    
    minlength = char_size_px * 0.8 * 5
    gap = 10
    
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
            if previous_x1 is None or abs(previous_x1 - x1) > 10: # ほぼ同じ縦線消去
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
            if previous_y1 is None or abs(previous_y1 - y1) > 10: # ほぼ同じ横線消去
                row_ranges.append(y1)
                sub_line_start.append(min(x1, x2)) # 左から
            previous_y1 = y1
        row_ranges.sort()
        sub_line_start.sort()
        return row_ranges, sub_line_start[0]
    
def detect_chars_cropbox(rotated_image, main_lines, sub_line_start, params):
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
                x_right = main_lines[1] - cols * (char_size_px + sep_size_px)
                y_upper = sub_line_start + n * char_size_px
                
                x_left = x_right - char_size_px
                y_lower = y_upper + char_size_px
                char = rotated_image.crop((x_left+2, y_upper+2, x_right-2, y_lower-2))
                chars.append(char)
    elif muki == 'yoko': # 左から
        for cols in range(int(ncols)):
            for n in range(int(nchars)):
                x_left = sub_line_start + n * char_size_px
                y_upper = main_lines[1] + cols * (char_size_px + sep_size_px)
                
                x_right = x_left + char_size_px
                y_lower = y_upper + char_size_px
                char = rotated_image.crop((x_left+5, y_upper+5, x_right-5, y_lower-5))
                chars.append(char)
    return chars

def detect_line_in_char(char_img):
    img = np.array(char_img)
    height, width = img.shape
    judge_img = cv2.bitwise_not(img)

    minlength = width * 0.95
    gap = 10

    # 検出しやすくするために二値化
    th, judge_img = cv2.threshold(judge_img, 1, 255, cv2.THRESH_BINARY)

    lines = []
    lines = cv2.HoughLinesP(judge_img, rho=1, theta=np.pi/360, threshold=100, minLineLength=minlength, maxLineGap=gap)
    
    lineadd_img = np.copy(img)  # 元の画像データを直接変更しないようにコピーを作成
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 傾きが threshold_slope px以内の線をyoko線と判断
            if abs(y1 - y2) < 3:
                whiteline = 3
                lineadd_img = cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), whiteline)
    return Image.fromarray(lineadd_img)
    

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


