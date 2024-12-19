import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from operator import itemgetter
from PIL import Image, ImageOps
import pytesseract
import pyocr
import pyocr.builders
import json
import sys
import os
import glob
import re
import math
import itertools
import sympy as sy
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Polygon
from shapely.validation import explain_validity
# --------------------------------------------------------------------------------------------
def main():
    sheet_path = '/data/matuzaki/sisha-split'
    save_dir = '/data2/abababam1/HandwrittenTextAlign/toda_char_imgs_new'
    
    OCRfailed = [['S1-125', 'sasame-1-2', 'xaah'], ['S3-5', 'sasame-3-1', 'xaae'], ['S2-230', 'bijogi-2-4', 'xaad'], ['S2-115', 'bijogi-2-1', 'xaba'], ['S1-14', 'miyamoto-1-1', 'xaan'], ['S3-30', 'sasame-3-1', 'xaad'], ['S1-17', 'miyamoto-1-1', 'xaaq'], ['S1-7', 'miyamoto-1-1', 'xaaa'], ['S1-155', 'bijogi-1-1', 'xaaw'], ['S1-159', 'bijogi-1-1', 'xabd'], ['S1-115', 'sasame-1-2', 'xaap'], ['S1-172', 'bijogi-1-1', 'xaad'], ['S1-77', 'sasame-1-1', 'xaap'], ['S1-95', 'sasame-1-1', 'xaac'], ['S1-227', 'bijogi-1-3', 'xaaf'], ['S2-70', 'sasame-2-3', 'xaar'], ['S2-56', 'sasame-2-3', 'xabf'], ['S2-57', 'sasame-2-3', 'xaam'], ['S2-58', 'sasame-2-3', 'xabc'], ['S2-71', 'sasame-2-3', 'xaai'], ['S2-145', 'bijogi-2-2', 'xaal'], ['S2-159', 'bijogi-2-3', 'xaax'], ['S2-211', 'bijogi-2-4', 'xaaw'], ['S2-65', 'sasame-2-3', 'xaaq'], ['S2-218', 'bijogi-2-4', 'xaac'], ['S2-223', 'bijogi-2-4', 'xaaa'], ['S2-5', 'sasame-2-3', 'xaad'], ['S2-50', 'sasame-2-3', 'xaaf'], ['S2-52', 'sasame-2-3', 'xaan'], ['S2-55', 'sasame-2-3', 'xabb'], ['S3-54', 'sasame-3-3', 'xaak'], ['S3-56', 'sasame-3-3', 'xaaq'], ['S3-58', 'sasame-3-3', 'xaat'], ['S3-17', 'sasame-3-1', 'xaam'], ['S3-52', 'sasame-3-3', 'xaad'], ['S3-53', 'sasame-3-3', 'xaac'], ['S3-55', 'sasame-3-3', 'xaal'], ['S3-59', 'sasame-3-3', 'xaas'], ['S3-7','sasame-3-1', 'xaag'], ['S2-207', 'bijogi-2-4', 'xaaq']]
    OCRfailed_path = remove_OCRfailed(OCRfailed)
    
    for ID, school, filename in OCRfailed:
        path = '/data/matuzaki/sisha-split/'+f'{school}.tif'+'/'+f'{filename}.tif'
        save_dir_imgs = os.path.join(save_dir, school)
        os.makedirs(save_dir_imgs, exist_ok=True)
        
        i = char_from_scan_OCRfailed(path, save_dir_imgs, ID, i = 1)
    
    for path in glob.glob(f'{sheet_path}/*.tif/*.tif', recursive=True):
        if path in OCRfailed_path:
            continue
        school_class = os.path.splitext(path.split('/')[-2])[0]
        save_dir_imgs = os.path.join(save_dir, school_class)
        os.makedirs(save_dir_imgs, exist_ok=True)
        
        i = char_from_scan(path, save_dir_imgs, i = 1)
        
        #if i == None:
            #OCRfailed.append([path, save_dir_imgs])
            
    print(f"OCR Result: '{100 * len(OCRfailed)/len(glob.glob(f'{sheet_path}/*.tif/*.tif'))}'%")
    
    #for path, save_dir_imgs in OCRfailed:
    #    print(f'path: {path},\n save_dir: {save_dir_imgs}')
    return 0
# --------------------------------------------------------------------------------------------
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
            numberd_img = scan_img.crop((0, height - 300, 1000, height))
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

    # crop_box作成
    cropbox = find_genko_box(scan_img, params)
    if cropbox is None:
        cropbox = find_largest_rectangle(scan_img, params)
        # return 0
    cropped_img = scan_img.crop(cropbox) # crop の引数は ((left, upper, right, lower))
    
    # pillow to cv2
    image = np.array(cropped_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # line、crosspoint検出のための前処理
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    canny_edges = cv2.Canny(gray, 1, 100, apertureSize = 3)
    kernel = np.ones((3, 3), np.uint8)
    outLineImage = cv2.dilate(canny_edges, kernel, iterations=1)
    
    houghPList, mainlines = hough_lines_p(image, outLineImage, params)  # 直線抽出
    cross_points = draw_cross_points(image, houghPList, params)   # 直線リストから交点を描画
    cv2.imwrite("/home/abababam1/result_houghP_cross.png", image)  # ファイル保存
    ''' 旧    
    # 正方形の頂点をピックアップ
    squares = find_squares(cross_points, params, cropbox)

    # 正方形でクロップ
    cropped_images = crop_squares(image, squares)
    '''    

    ''' 旧
    # 個々の文字の画像を保存する
    save_path = os.path.join(save_dir, f'{file_name}_{number}')
    os.makedirs(save_path, exist_ok=True)
    for i, cropped_image in enumerate(cropped_images.values()):
        image = detect_line_in_char(cropped_image)
        
        #cropped_image.save((save_path)+'/'+'{:05d}.png'.format(i), 'PNG')
        output_path = f'{save_path}'+'/'+'{:05d}.png'.format(i)
        cv2.imwrite(output_path, image)
    '''
    # cross_pointsに位置付け
    sorted_cross_points = sort_crosspoints(cross_points, params)
    
    # cross_pointsノイズ除去、補間
    sorted_cross_points, minmax = compliment_cross_points(cross_points, params, sorted_cross_points)
    
    # cross_pointsからマス目の頂点を検出
    cropboxes = cropboxes_from_cross_points(sorted_cross_points, params, minmax)
    # crop
    if not cropboxes:
        return 0
    cropped_images = crop_squares(image, cropboxes)
    
    # 個々の文字の画像を保存する
    save_path = os.path.join(save_dir, f'{file_name}_{number}')
    os.makedirs(save_path, exist_ok=True)
    for i, image in cropped_images.items():
        # 罫線を白塗り
        image = detect_line_in_char(image)
        output_path = f'{save_path}'+'/'+'{:05d}.png'.format(i)
        cv2.imwrite(output_path, image)
        #print(f"正方形でクロップされた画像が {output_path} に保存されました")
    return i
# --------------------------------------------------------------------------------------------
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
            numberd_img = scan_img.crop((0, height - 300, 1000, height))
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

    # crop_box作成
    cropbox = find_genko_box(scan_img, params)
    if cropbox is None:
        cropbox = find_largest_rectangle(scan_img, params)
        # return 0
    cropped_img = scan_img.crop(cropbox) # crop の引数は ((left, upper, right, lower))
    
    # pillow to cv2
    image = np.array(cropped_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # line、crosspoint検出のための前処理
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    canny_edges = cv2.Canny(gray, 1, 100, apertureSize = 3)
    kernel = np.ones((3, 3), np.uint8)
    outLineImage = cv2.dilate(canny_edges, kernel, iterations=1)
    
    houghPList, mainlines = hough_lines_p(image, outLineImage, params)  # 直線抽出
    cross_points = draw_cross_points(image, houghPList, params)   # 直線リストから交点を描画
    cv2.imwrite("/home/abababam1/result_houghP_cross.png", image)  # ファイル保存
    ''' 旧    
    # 正方形の頂点をピックアップ
    squares = find_squares(cross_points, params, cropbox)

    # 正方形でクロップ
    cropped_images = crop_squares(image, squares)
    '''    

    ''' 旧
    # 個々の文字の画像を保存する
    save_path = os.path.join(save_dir, f'{file_name}_{number}')
    os.makedirs(save_path, exist_ok=True)
    for i, cropped_image in enumerate(cropped_images.values()):
        image = detect_line_in_char(cropped_image)
        
        #cropped_image.save((save_path)+'/'+'{:05d}.png'.format(i), 'PNG')
        output_path = f'{save_path}'+'/'+'{:05d}.png'.format(i)
        cv2.imwrite(output_path, image)
    '''
    # cross_pointsに位置付け
    sorted_cross_points = sort_crosspoints(cross_points, params)
    
    # cross_pointsノイズ除去、補間
    sorted_cross_points, minmax = compliment_cross_points(cross_points, params, sorted_cross_points)
    
    # cross_pointsからマス目の頂点を検出
    cropboxes = cropboxes_from_cross_points(sorted_cross_points, params, minmax)
    # crop
    if not cropboxes:
        return 0
    cropped_images = crop_squares(image, cropboxes)
    
    # 個々の文字の画像を保存する
    save_path = os.path.join(save_dir, f'{file_name}_{number}')
    os.makedirs(save_path, exist_ok=True)
    for i, image in cropped_images.items():
        # 罫線を白塗り
        image = detect_line_in_char(image)
        output_path = f'{save_path}'+'/'+'{:05d}.png'.format(i)
        cv2.imwrite(output_path, image)
        #print(f"正方形でクロップされた画像が {output_path} に保存されました")
    return i
# --------------------------------------------------------------------------------------------
def remove_OCRfailed(OCRfailed):
    OCRfailed_path = []
    for ID, school, filename in OCRfailed:
        path = '/data/matuzaki/sisha-split/'+f'{school}.tif'+'/'+f'{filename}.tif'
        OCRfailed_path.append(path)
    return OCRfailed_path

# 原稿配置のパラメータが入ったjsonファイルの読み込み
def load_genko_params(filename):
    with open(filename, 'r') as file:
        return json.load(file)
    
def check_genko_size_and_angle(hull, genko_1, genko_2):  
    quad = set()
    n = len(hull)
    x_max, x_min = 0, float('inf')
    y_max, y_min = 0, float('inf')
    genko_angle_list = []
    tate = 0
    yoko = 0
    for i in range(n):
        for j in range(i + 1, n):
            ij_distance = np.linalg.norm(hull[i][0] - hull[j][0])
            ij_width = abs(hull[i][0][0] - hull[j][0][0])
            ij_height = abs(hull[i][0][1] - hull[j][0][1])
            # 横
            if any(abs(ij_distance - henn) <= 50 for henn in [genko_1, genko_2]) and ij_height <= 20:
                yoko = 1
                quad.add(tuple(hull[i][0]))
                quad.add(tuple(hull[j][0]))
                x_max = max(x_max, hull[i][0][0], hull[j][0][0])
                x_min = min(x_min, hull[i][0][0], hull[j][0][0])
                y_max = max(y_max, hull[i][0][1], hull[j][0][1])
                y_min = min(y_min, hull[i][0][1], hull[j][0][1])
                #genko_angle = calculate_angle_yoko(hull[i][0][0], hull[j][0][0], hull[i][0][0], hull[j][0][0])
                #genko_angle_list.append(genko_angle)
            # 縦
            elif ij_width <= 20 and any(abs(ij_distance - henn) <= 50 for henn in [genko_1, genko_2]):
                tate = 1
                quad.add(tuple(hull[i][0]))
                quad.add(tuple(hull[j][0]))
                x_max = max(x_max, hull[i][0][0], hull[j][0][0])
                x_min = min(x_min, hull[i][0][0], hull[j][0][0])
                y_max = max(y_max, hull[i][0][1], hull[j][0][1])
                y_min = min(y_min, hull[i][0][1], hull[j][0][1])
                #genko_angle = calculate_angle_tate(hull[i][0][0], hull[j][0][0], hull[i][0][0], hull[j][0][0])
                #genko_angle_list.append(genko_angle)
    if yoko == 1 and tate == 1:
        # 四頂点と右上の座標
        #cropbox = (max(0, x_min + 5), max(0, y_min + 5), min(x_max - 5, 5000), min(y_max - 5, 6000))
        cropbox = (x_min + 0, y_min + 0, x_max - 0, y_max - 0)
        return quad, cropbox
    else:
        return quad, None

def find_genko_box(pillow_img, params):
    width, height = pillow_img.size
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi) # 236~
    sep_size_px = int((sep_width / 2.54) * dpi) # 47~
    
    
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
        quad, cropbox = check_genko_size_and_angle(best_rect, genko_1, genko_2)
        for i in range(len(quad)):
            cv2.circle(img, list(quad)[i], 10, (0, 0, 255), thickness=5, lineType=cv2.LINE_8, shift=0)
        for i in range(len(best_rect)):
            cv2.circle(img, tuple(best_rect[i][0]), 5, (255, 0, 0), thickness=5, lineType=cv2.LINE_8, shift=0)
        #cv2.imwrite('/home/abababam1/result_image_.jpg', img)

        if cropbox is not None:
            return cropbox
        else:
            #print('quadrilaterals Detection error')
            #cv2.imwrite(f'/home/abababam1/HandwrittenTextAlign/toda_crop_imgs/error_image/quad_error.jpg', img)
            return None
    else:
        print('hull Detection error')
        cv2.imwrite('/home/abababam1/HandwrittenTextAlign/toda_crop_imgs/error_image/hull_error.jpg', img)
        return None
    
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
    min_distance = float('inf')
    best_rect = None
    best_angle = 0
    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 最小の回転された矩形を取得
        box = cv2.boxPoints(rect)  # 回転された矩形の四角を取得
        box = np.int0(box)  # 座標を整数に変換
        area = cv2.contourArea(box)
        distance = abs(area - best_area)
        if distance < min_distance:
            min_distance = distance
            best_rect = box
    if best_rect is not None:
        # バウンディングボックスの座標でクロップ
        x, y, w, h = cv2.boundingRect(best_rect)
        cropbox = (x, y, x+w, y+h)
        return cropbox
    else:
        return None
# --------------------------------------------------------------------------------------------
# ⚠️
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

# --------------------------------------------------------------------------------------------
def detect_line_in_char(char_img): # cleaner
    image = np.array(char_img)
    img = None

    # 検出しやすくするために二値化
    #th, judge_img = cv2.threshold(judge_img, 1, 255, cv2.THRESH_BINARY)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    canny_edges = cv2.Canny(gray, 1, 100, apertureSize = 3)
    kernel = np.ones((3, 3), np.uint8)
    outLineImage = cv2.dilate(canny_edges, kernel, iterations=1)
    
    height, width = gray.shape
    
    minlength = width * 0.90
    gap = 30

    lines = []
    lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/360, threshold=100, minLineLength=minlength, maxLineGap=gap)
    
    detect_length = width * 0.05
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if (abs(y1 - y2) < 5 and (y1 <= detect_length or y2 >= height-detect_length)) or (abs(x1 - x2) < 5 and (x1 <= detect_length or x2 >= width-detect_length)):
                whiteline = 2
                img = cv2.line(image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), whiteline)
        if img is not None:
            return img#Image.fromarray(img)
        else:
            return char_img
    else:
        return char_img

# 確率的ハフ変換で直線を抽出する関数
def hough_lines_p(image, outLineImage, params):
    char_width = params['char_width']
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    
    lineList = []
    main_lines = None
    # 確率的ハフ変換で直線を抽出
    lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/360, threshold=150, minLineLength=char_size_px*0.9, maxLineGap=5)
    if lines is not None:
        print("hough_lines_p: ", len(lines))

        main_lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/360, threshold=150, minLineLength=char_size_px*5, maxLineGap=5)
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lineList.append((x1, y1, x2, y2))
            #cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2) # 緑色で直線を引く

        return lineList, main_lines
    else:
        return lineList, main_lines

# --------------------------------------------------------------------------------------------
# 交点を描画する関数
def draw_cross_points(image, lineList, params):
    sep_width = params['sep_width']
    dpi = 600
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    size = len(lineList)
    cross_points = []
    cnt = 0
    for i in range(size-1):
        for j in range(i+1, size):
            pointA = (lineList[i][0], lineList[i][1])
            pointB = (lineList[i][2], lineList[i][3])
            pointC = (lineList[j][0], lineList[j][1])
            pointD = (lineList[j][2], lineList[j][3])
            ret, cross_point = calc_cross_point(pointA, pointB, pointC, pointD) # 交点を計算
            if ret:
                # 交点が取得できた場合でも画像の範囲外のものは除外
                # if (cross_point[0] >= 0 + 0.5*sep_size_px) and (cross_point[0] <= image.shape[1] - 0.5*sep_size_px) and (cross_point[1] >= 0) and (cross_point[1] <= image.shape[0]):
                if (cross_point[0] >= 0) and (cross_point[0] <= image.shape[1]) and (cross_point[1] >= 0) and (cross_point[1] <= image.shape[0]):
                    #cv2.circle(image, (cross_point[0],cross_point[1]), 2, (255,0,0), 3) # 交点を青色で描画
                    cnt = cnt + 1
                    cross_points.append(cross_point)
    # 近くの座標をクラスタリングしてまとめる
    cross_points = cluster_and_round_points(cross_points, eps=sep_size_px*0.1, min_samples=1)
    
    for cross_point in cross_points:
        #if (cross_point[0] >= 0 + 0.5*sep_size_px) and (cross_point[0] <= image.shape[1] - 0.5*sep_size_px) and (cross_point[1] >= 0) and (cross_point[1] <= image.shape[0]):
        if (cross_point[0] >= 0) and (cross_point[0] <= image.shape[1]) and (cross_point[1] >= 0) and (cross_point[1] <= image.shape[0]):    
            cv2.circle(image, (cross_point[0],cross_point[1]), 2, (255,0,0), 3) # 交点を青色で描画
    #print("draw_cross_points:", cnt)
    print("draw_cross_points:", len(cross_points))
    return cross_points, image

# 線分ABと線分CDの交点を求める関数
def calc_cross_point(pointA, pointB, pointC, pointD):
    cross_points = [0,0]
    bunbo = (pointB[0] - pointA[0]) * (pointD[1] - pointC[1]) - (pointB[1] - pointA[1]) * (pointD[0] - pointC[0])

    # 直線が平行な場合
    if (bunbo == 0):
        return False, cross_points

    vectorAC = ((pointC[0] - pointA[0]), (pointC[1] - pointA[1]))
    r = ((pointD[1] - pointC[1]) * vectorAC[0] - (pointD[0] - pointC[0]) * vectorAC[1]) / bunbo
    s = ((pointB[1] - pointA[1]) * vectorAC[0] - (pointB[0] - pointA[0]) * vectorAC[1]) / bunbo

    # 短い線分がある場合、拡張された範囲でチェックする
    if abs(pointA[1] - pointB[1]) <= 5:
        if (r < -0.5*10000) or (r > 1.5*10000) or (s < 0) or (s > 1):
            return False, cross_points
    elif abs(pointC[1] - pointD[1]) <= 5:
        if (r < 0) or (r > 1) or (s < -0.5*10000) or (s > 1.5*10000):
            return False, cross_points
    else:
        if (r < 0) or (r > 1) or (s < 0) or (s > 1):
            return False, cross_points
    
    # 線分AB、線分AC上に存在しない場合
    #if (r <= -0.5) or (1.5 <= r) or (s <= -0.5) or (1.5 <= s):
    #    return False, cross_points
    
    # 直角に近いかどうかを判定するための内積計算
    vectorAB = (pointB[0] - pointA[0], pointB[1] - pointA[1])
    vectorCD = (pointD[0] - pointC[0], pointD[1] - pointC[1])
    dot_product = vectorAB[0] * vectorCD[0] + vectorAB[1] * vectorCD[1]
    magAB = np.sqrt(vectorAB[0]**2 + vectorAB[1]**2)
    magCD = np.sqrt(vectorCD[0]**2 + vectorCD[1]**2)
    cos_theta = dot_product / (magAB * magCD)

    # 角度が85度から95度の範囲にあるかチェック
    if not (-0.05 <= cos_theta <= 0.05):
        return False, cross_points

    # rを使った計算の場合
    distance = ((pointB[0] - pointA[0]) * r, (pointB[1] - pointA[1]) * r)
    cross_points = (int(pointA[0] + distance[0]), int(pointA[1] + distance[1]))

    # sを使った計算の場合
    # distance = ((pointD[0] - pointC[0]) * s, (pointD[1] - pointC[1]) * s)
    # cross_points = (int(pointC[0] + distance[0]), int(pointC[1] + distance[1]))

    return True, cross_points
# --------------------------------------------------------------------------------------------
def cluster_and_round_points(points, eps=3, min_samples=1):
    if len(points) == 0:
        return np.array(points)
    points_array = np.array(points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points_array)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    clustered_points = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster = points_array[labels == label]
        centroid = np.round(cluster.mean(axis=0)).astype(int)
        clustered_points.append(centroid)
    
    return np.array(clustered_points)

def cluster_x_coordinates(points, eps=2, min_samples=1):
    if len(points) == 0:
        return np.array(points)
    # x座標のみを取り出してクラスタリング # ⚠️ 同じyの値に対して
    x_coords = points[:, 0].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x_coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    clustered_points = points.copy()
    for label in unique_labels:
        if label == -1:
            continue
        # クラスター内のx座標をまとめ、その平均値を新しいx座標とする
        cluster_indices = np.where(labels == label)[0]
        mean_x = np.round(np.mean(points[cluster_indices, 0])).astype(int)
        clustered_points[cluster_indices, 0] = mean_x
    
    return clustered_points

def cluster_y_coordinates(points, eps=2, min_samples=1):
    if len(points) == 0:
        return np.array(points)
    # y座標のみを取り出してクラスタリング
    y_coords = points[:, 1].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    clustered_points = points.copy()
    for label in unique_labels:
        if label == -1:
            continue
        # クラスター内のy座標をまとめ、その平均値を新しいy座標とする # ⚠️ 同じxの値に対して
        cluster_indices = np.where(labels == label)[0]
        mean_y = np.round(np.mean(points[cluster_indices, 1])).astype(int)
        clustered_points[cluster_indices, 1] = mean_y
    
    return clustered_points
# --------------------------------------------------------------------------------------------
def crop_squares(image, squares):
    cropped_images = dict()
    image_np = np.array(image)
    for i, square in squares.items():
        if square is None:
            continue
        x_coords = sorted([int(p[0]) for p in square])
        y_coords = sorted([int(p[1]) for p in square])
        #min_x, max_x = min(x_coords), max(x_coords)
        #min_y, max_y = min(y_coords), max(y_coords)
        
        min_x, max_x = x_coords[1], x_coords[-2]
        min_y, max_y = y_coords[1], y_coords[-2]
        
        cropped_image = image_np[min_y:max_y, min_x:max_x]
        #cropped_image = Image.fromarray(cropped_image)

        cropped_images[i] = cropped_image
    return cropped_images

def PerspectiveTransform(img, square, square_size):
    left_down, right_down, right_up, left_up = square
    # 台形の4点
    perspective_base = np.float32([left_down, right_down, right_up, left_up])

    # 変換先の座標（正方形の4点）
    perspective = np.float32([
        [0, square_size], 
        [square_size, square_size], 
        [square_size, 0], 
        [0, 0]
    ])
    psp_matrix = cv2.getPerspectiveTransform(perspective_base, perspective)
    square_img = cv2.warpPerspective(img, psp_matrix, (square_size, square_size))
    
    return square_img

# -----------------------------------------------------------------------------------------

def sort_crosspoints(cross_points, params):
    muki = params['muki']
    sep_width = params['sep_width']
    
    dpi = 600
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    if type(cross_points) == tuple and len(cross_points) == 2:
        cross_points = cross_points[0]
    
    if muki == 'tate':
        numbered_cluster_dic = find_j_cluster_sort_i(cross_points, eps=sep_size_px*0.7, min_samples=1)
    elif muki == 'yoko':
        numbered_cluster_dic = find_i_cluster_sort_j(cross_points, eps=sep_size_px*0.7, min_samples=1)

    return numbered_cluster_dic

def compliment_cross_points(cross_points, params, numbered_cluster_dic):
    muki = params['muki']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    if type(cross_points) == tuple and len(cross_points) == 2:
        cross_points = cross_points[0]
    
    min_i, max_i = minmax_i_cluster(cross_points, eps=sep_size_px*0.7, min_samples=1)
    min_j, max_j = minmax_j_cluster(cross_points, eps=sep_size_px*0.7, min_samples=1)
    minmax = (min_i, max_i, min_j, max_j)
    
    for k in numbered_cluster_dic.keys():
        # ノイズを除去
        numbered_cluster_dic[k] = Rid_noise(numbered_cluster_dic[k], muki, char_size_px, sep_size_px)
        print(numbered_cluster_dic[k])
        # 欠けてる点を補間
        numbered_cluster_dic[k] = Interpolate_lack(numbered_cluster_dic[k], muki, char_size_px, sep_size_px, minmax)
    
    return numbered_cluster_dic, minmax

def find_i_cluster_sort_j(points, eps, min_samples=1):
    if len(points) == 0:
        return np.array(points)
    
    # 縦書き: ソート: x は降順、y は昇順
    
    # y座標のみを取り出してクラスタリングを実行
    y_coords = np.array(points[:, 1]).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    # クラスターごとのx座標に基づいて昇順に番号を付ける
    numbered_cluster_dic = dict()
    for label in unique_labels:
        cluster_points = points[labels == label]
        # ノイズ除去 ⚠️　不完全だが、罫線以外の交点を取るため
        if len(cluster_points) <= 5:
            continue
        
        # x座標でソート
        sorted_cluster_points = cluster_points[np.argsort(cluster_points[:, 0])]
        numbered_cluster_dic[label] = sorted_cluster_points
        
    numbered_cluster_list = sorted(numbered_cluster_dic.items(), key=lambda item: np.mean(item[1][:, 1]))
    numbered_cluster_dic.clear()
    numbered_cluster_dic.update(numbered_cluster_list)
    
    return numbered_cluster_dic

def find_j_cluster_sort_i(points, eps, min_samples=1):
    if len(points) == 0:
        return np.array(points)
    
    # 横書き: ソート: y は昇順、x は昇順
    
    # x座標のみを取り出してクラスタリングを実行
    x_coords = np.array(points[:, 0]).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x_coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    # クラスターごとのy座標に基づいて昇順に番号を付ける
    numbered_cluster_dic = dict()
    for label in unique_labels:
        cluster_points = points[labels == label]
        # ノイズ除去 ⚠️　罫線以外の交点を取るため
        if len(cluster_points) <= 5:
            continue
        
        # y座標でソート
        sorted_cluster_points = cluster_points[np.argsort(cluster_points[:, 1])]
        numbered_cluster_dic[label] = sorted_cluster_points
        
    numbered_cluster_list = sorted(numbered_cluster_dic.items(), key=lambda item: np.mean(item[1][:, 0]), reverse=True)
    numbered_cluster_dic.clear()
    numbered_cluster_dic.update(numbered_cluster_list)
    
    return numbered_cluster_dic
    

def minmax_j_cluster(points, eps=1, min_samples=1): # x: 横書きの時
    if len(points) == 0:
        return np.array(points)
    
    # x座標(j)のみを取り出してクラスタリングを実行
    x_coords = points[:, 0].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x_coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    mean_x_values = []
    # 各クラスタのx座標の平均値を計算
    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        mean_x = np.round(np.mean(points[cluster_indices, 0])).astype(int)
        mean_x_values.append(mean_x)
    
    min_mean_x_values = min(mean_x_values)
    max_mean_x_values = max(mean_x_values)

    return min_mean_x_values, max_mean_x_values

def minmax_i_cluster(points, eps=1, min_samples=1): # y: 縦書きの時
    if len(points) == 0:
        return np.array(points)
    
    # y座標(i)のみを取り出してクラスタリングを実行
    y_coords = points[:, 1].reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(y_coords)
    labels = clustering.labels_
    unique_labels = set(labels)
    
    mean_y_values = []
    # 各クラスタのy座標の平均値を計算
    for label in unique_labels:
        if label == -1:
            continue
        cluster_indices = np.where(labels == label)[0]
        mean_y = np.round(np.mean(points[cluster_indices, 1])).astype(int)
        mean_y_values.append(mean_y)
    
    min_mean_y_values = min(mean_y_values)
    max_mean_y_values = max(mean_y_values)

    return min_mean_y_values, max_mean_y_values

# クラスター内points
def Rid_noise(points, muki, char_size_px, sep_size_px):
    distances = []
    clusters = set()
    combinations = list(itertools.combinations(points, 3))
    for combo in combinations:
        if (is_distance(combo, 0, 1, char_size_px, sep_size_px) and is_distance(combo, 0, 2, char_size_px, sep_size_px)) or \
        (is_distance(combo, 0, 1, char_size_px, sep_size_px) and is_distance(combo, 1, 2, char_size_px, sep_size_px)) or \
        (is_distance(combo, 0, 2, char_size_px, sep_size_px) and is_distance(combo, 1, 2, char_size_px, sep_size_px)):
            clusters.add(tuple(combo[0]))
            clusters.add(tuple(combo[1]))
            clusters.add(tuple(combo[2]))
    if muki == 'tate':
        return sorted(list(clusters), key=lambda point: point[1])
    else:
        return sorted(list(clusters), key=lambda point: point[0])
    
def Interpolate_lack(points, muki, char_size_px, sep_size_px, minmax):
    if len(points) == 0:
        return False
    
    if type(points) == np.ndarray:
        points = list(points)
    
    # 等間隔に点を用意（理想）
    min_i, max_i, min_j, max_j = minmax
    if muki == 'tate':
        gold_points = np.arange(min_i, max_i, char_size_px)
    else:
        gold_points = np.arange(min_j, max_j, char_size_px)
    
    # 欠けている点を見つける
    i = 0
    for j, gold_point in enumerate(gold_points):
        
        gold_point = (points[i][0], gold_point) if muki == 'tate' else (gold_point, points[i][1])
        
        is_exist = np.array([np.linalg.norm(np.array(point) - np.array(gold_point)) <= sep_size_px*0.5 for point in points])
        
        if ~is_exist.any():
            points.append(gold_point) # 座標を補完
            i -= 1
            
        i += 1
        if i >= len(points):
            i = len(points) - 1

    if muki == 'tate':
        return sorted(list(points), key=lambda point: point[1])
    else:
        return sorted(list(points), key=lambda point: point[0])
# -----------------------------------------------------------------------------------------

# cropboxの4点を収集
def cropboxes_from_cross_points(numbered_cluster_dic, params, minmax):
    keys = list(numbered_cluster_dic.keys())
    cropboxes = dict()
    number = 1
    pre_cropbox = None
    centerlines = centerline(params, minmax)
    
    for k in range(len(keys)-1):
        if not numbered_cluster_dic[keys[k]]:
            return False
        for l in range(len(numbered_cluster_dic[keys[k]])-1):
            
            cropbox = search_4points(numbered_cluster_dic, keys, params, k, l)
            
            if cropbox and is_out_of_sync(cropbox, pre_cropbox, centerlines, params):
                cropboxes[number] = cropbox
                if number == 101:
                    p1, p2, p3, p4 = cropbox
                    #abs_ = abs(np.linalg.norm(np.array(p1)-np.array(p2)) - np.linalg.norm(np.array(p1)-np.array(p3)))
                    #print(abs_, np.linalg.norm(np.array(p1)-np.array(p2)), np.linalg.norm(np.array(p1)-np.array(p3)))
                    #print(cropbox)
                number += 1
                pre_cropbox = cropbox
            else:
                continue
    return cropboxes
        
def search_4points(numbered_cluster_dic, keys, params, k, l):
    muki = params['muki']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
        
    
    #if not numbered_cluster_dic[keys[k]]:
    #    return False
    
    p1 = numbered_cluster_dic[keys[k]][l]
    #print(f'p1: {p1}')
    
    i = 0
    p2 = None
    while i < 3 and k+i < len(keys):
        
        if not numbered_cluster_dic[keys[k+i]]:
            return False
        
        indice2, cood2 = search_another_point(p1, numbered_cluster_dic[keys[k+i]][l+1:], char_size_px, sep_size_px)
        #indice2 += l+1
        if indice2 != -1:
            #print(f'indice2, cood2: {indice2}, {cood2}')
            p2 = cood2
            break
        i += 1
    else:
        if p2 is None:
            return False
    
    i = 1
    p3 = None
    while i < 5 and k+i < len(keys):
        
        if not numbered_cluster_dic[keys[k+i]]:
            return False
        
        indice3, cood3 = search_another_point(p1, numbered_cluster_dic[keys[k+i]][:], char_size_px, sep_size_px)
        if indice3 != -1:
            #print(f'indice3, cood3: {indice3}, {cood3}')
            indice, _ = search_another_point(p2, [cood3], char_size_px*np.sqrt(2), sep_size_px)
            if indice != -1:
                p3 = cood3
                break
        i += 1
    else:
        if p3 is None:
            return False
    
    j = 0
    p4 = None
    while j < 3 and k+i+j < len(keys):
        
        if not numbered_cluster_dic[keys[k+i+j]]:
            return False
        
        if indice3+1 < len(numbered_cluster_dic[keys[k+i+j]]):
        
            indice4, cood4 = search_another_point(p2, numbered_cluster_dic[keys[k+i+j]][indice3+1:], char_size_px, sep_size_px)
            #indice4 += indice3+1
            if indice4 != -1:
                #print(f'indice4, cood4: {indice4}, {cood4}')
                indice, _ = search_another_point(p3, [cood4], char_size_px, sep_size_px)
                if indice != -1:
                    p4 = cood4
                    return [p1, p2, p3, p4]
        j += 1
    else:
        if p4 is None:
            return False
#------------------------------------------------------------------------------------------------

# 候補の中で一番誤差小さいもの
def search_another_point(p1, p2s, char_size_px, sep_size_px):
    points = [(p, p2s[p], delta(p1, p2s[p], char_size_px)) for p in range(len(p2s)) \
              if delta(p1, p2s[p], char_size_px) <= sep_size_px*0.5]
    if len(points) > 0:
        points = sorted(points, key=lambda point: point[2])
        return points[0][0], points[0][1] # 0: indice, 1: coord
    else:
        return -1, (-1, -1)
    
def delta(p1, p2, char_size_px):
    return abs(np.linalg.norm(np.array(p1)-np.array(p2)) - char_size_px)

#------------------------------------------------------------------------------------------------

def centerline(params, minmax):
    muki = params['muki']
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    min_i, max_i, min_j, max_j = minmax
    if muki == 'tate':
        top = max_j
        bottom = min_j
        main_centerline = np.arange(max_j - sep_size_px - 0.5*char_size_px, min_j, - (char_size_px + sep_size_px))
        sub_centerline = np.arange(min_i + 0.5*char_size_px, max_i, char_size_px)
    else:
        top = min_i
        bottom = max_i
        main_centerline = np.arange(min_i + sep_size_px + 0.5*char_size_px, max_i, char_size_px + sep_size_px)
        sub_centerline = np.arange(min_j + 0.5*char_size_px, max_j, char_size_px)
    return (main_centerline, sub_centerline) 

def is_out_of_sync(cropbox, pre_cropbox, centerlines, params):
    return is_slip(cropbox, centerlines, params) and is_square(cropbox, params) and is_overlap(cropbox, pre_cropbox)

def is_slip(cropbox, centerlines, params):
    p1, p2, p3, p4 = cropbox
    main_centerline, sub_centerline = centerlines
    
    muki = params['muki']
    m = 0 if muki == 'tate' else 1
    boxcenter = [((p1[z]+p2[z])/2 + (p3[z]+p4[z])/2)/2 for z in [0, 1]]
    
    sep_width = params['sep_width']
    dpi = 600
    sep_size_px = int((sep_width / 2.54) * dpi)
    eps = sep_size_px*0.7
    
    is_center_main = np.array([abs(boxcenter[m] - center) < eps for center in main_centerline])
    is_center_sub = np.array([abs(boxcenter[abs(m-1)] - center) < eps for center in sub_centerline])
    
    return is_center_main.any() and is_center_sub.any()

def is_square(cropbox, params):
    p1, p2, p3, p4 = cropbox
    
    char_width = params['char_width']
    sep_width = params['sep_width']
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    eps = sep_size_px*0.1
    
    dist_12 = np.linalg.norm(np.array(p1)-np.array(p2))
    dist_34 = np.linalg.norm(np.array(p3)-np.array(p4))
    dist_13 = np.linalg.norm(np.array(p1)-np.array(p3))
    dist_24 = np.linalg.norm(np.array(p2)-np.array(p4))
    
    if abs(dist_12-char_size_px) < eps and abs(dist_13-char_size_px) < eps:
        return True
    elif abs(dist_12-char_size_px) < eps and abs(dist_24-char_size_px) < eps:
        return True
    elif abs(dist_13-char_size_px) < eps and abs(dist_34-char_size_px) < eps:
        return True
    elif abs(dist_24-char_size_px) < eps and abs(dist_34-char_size_px) < eps:
        return True
    else:
        return False
    
    #return abs_<eps

def is_overlap(cropbox, pre_cropbox):
    if pre_cropbox == None:
        return True

    # 四角形の座標を定義
    rect1_coords = cropbox
    rect2_coords = pre_cropbox
    rect1_coords[2], rect1_coords[3] = rect1_coords[3], rect1_coords[2]
    rect2_coords[2], rect2_coords[3] = rect2_coords[3], rect2_coords[2]

    # ShapelyのPolygonオブジェクトを作成
    rect1 = create_valid_polygon(rect1_coords)
    rect2 = create_valid_polygon(rect2_coords)

    # 重なり部分のポリゴンを計算
    intersection = rect1.intersection(rect2)

    # 重なり面積を取得
    overlap_area = intersection.area
    
    return overlap_area < rect1.area*0.1
                              
def create_valid_polygon(coords):
    polygon = Polygon(coords)
    if not polygon.is_valid:
        #print(f"Invalid polygon: {explain_validity(polygon)}")
        # buffer(0)を使用して自己交差を修正
        polygon = polygon.buffer(0)
        if not polygon.is_valid:
            raise ValueError("Polygon cannot be fixed automatically.")
    return polygon

