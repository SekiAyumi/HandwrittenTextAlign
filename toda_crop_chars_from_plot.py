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

def main():
    sheet_path = '/data/matuzaki/sisha-split'
    save_dir = '/data2/abababam1/HandwrittenTextAlign/toda_char_imgs_new'
    
    OCRfailed = []
    
    for path in glob.glob(f'{sheet_path}/*.tif/*.tif', recursive=True):
        school_class = os.path.splitext(path.split('/')[-2])[0]
        save_dir_imgs = os.path.join(save_dir, school_class)
        os.makedirs(save_dir_imgs, exist_ok=True)
        
        i = char_from_scan(path, save_dir_imgs, i = 1)
        
        if i == None:
            OCRfailed.append([path, save_dir_imgs])
            
    print(f"OCR Result: '{100 * len(OCRfailed)/len(glob.glob(f'{sheet_path}/*.tif/*.tif'))}'%")
    
    for path, save_dir_imgs in OCRfailed:
        print(f'path: {path},\n save_dir: {save_dir_imgs}')
    return OCRfailed

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

    # crop_box作成
    cropbox, angle = find_genko_box(scan_img, params)
    if cropbox is None:
        return i
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
    
    houghPList, mainlines = hough_lines_p(image, outLineImage, char_size_px)  # 直線抽出
    cross_points = draw_cross_points(image, houghPList, sep_size_px)   # 直線リストから交点を描画
    cv2.imwrite("/home/abababam1/result_houghP_cross.png", image)  # ファイル保存
    
    # 正方形の頂点をピックアップ
    squares = find_squares(cross_points, params, cropbox)

    # 正方形でクロップ
    cropped_images = crop_squares(image, squares)
    
    if mainlines is not None:
        # 罫線を白塗り
        for line in mainlines:
            whiteline = 2
            lined_img = cv2.line(image, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (255, 255, 255), whiteline)
        image = Image.fromarray(lined_img)

    
    # 個々の文字の画像を保存する
    save_path = os.path.join(save_dir, f'{file_name}_{number}')
    os.makedirs(save_path, exist_ok=True)
    for i, cropped_image in enumerate(cropped_images.values()):
        #im_clean = detect_line_in_char(im)
        #cropped_image.save((save_path)+'/'+'{:05d}.png'.format(i), 'PNG')
        output_path = f'{save_path}'+'/'+'{:05d}.png'.format(i)
        cv2.imwrite(output_path, cropped_image)
    return i

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
    for i in range(n):
        for j in range(i + 1, n):
            ij_width = abs(hull[i][0][0] - hull[j][0][0])
            ij_height = abs(hull[i][0][1] - hull[j][0][1])
            # 横
            if any(abs(ij_width - henn) <= 50 for henn in [genko_1, genko_2]) and ij_height <= 5:
                quad.add(tuple(hull[i][0]))
                quad.add(tuple(hull[j][0]))
                x_max = max(x_max, hull[i][0][0], hull[j][0][0])
                x_min = min(x_min, hull[i][0][0], hull[j][0][0])
                y_max = max(y_max, hull[i][0][1], hull[j][0][1])
                y_min = min(y_min, hull[i][0][1], hull[j][0][1])
                genko_angle = calculate_angle_yoko(hull[i][0][0], hull[j][0][0], hull[i][0][0], hull[j][0][0])
                genko_angle_list.append(genko_angle)
            # 縦
            elif ij_width <= 5 and any(abs(ij_height - henn) <= 50 for henn in [genko_1, genko_2]):
                quad.add(tuple(hull[i][0]))
                quad.add(tuple(hull[j][0]))
                x_max = max(x_max, hull[i][0][0], hull[j][0][0])
                x_min = min(x_min, hull[i][0][0], hull[j][0][0])
                y_max = max(y_max, hull[i][0][1], hull[j][0][1])
                y_min = min(y_min, hull[i][0][1], hull[j][0][1])
                genko_angle = calculate_angle_tate(hull[i][0][0], hull[j][0][0], hull[i][0][0], hull[j][0][0])
                genko_angle_list.append(genko_angle)
    if len(quad) >= 4:
        # 四頂点と右上の座標
        #cropbox = (max(0, x_min + 5), max(0, y_min + 5), min(x_max - 5, 5000), min(y_max - 5, 6000))
        cropbox = (x_min + 0, y_min + 0, x_max - 0, y_max - 0)
        return quad, cropbox, genko_angle_list
    else:
        return quad, None, genko_angle_list

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
    
    print(char_size_px)
    
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


def detect_line_in_char(char_img): # cleaner
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

# 確率的ハフ変換で直線を抽出する関数
def hough_lines_p(image, outLineImage, char_size_px):
    lineList = []
    # 確率的ハフ変換で直線を抽出
    lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/360, threshold=150, minLineLength=char_size_px-5, maxLineGap=5)
    print("hough_lines_p: ", len(lines))
    
    main_lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/360, threshold=150, minLineLength=char_size_px*5, maxLineGap=5)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        lineList.append((x1, y1, x2, y2))
        #cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2) # 緑色で直線を引く

    return lineList, main_lines


# 交点を描画する関数
def draw_cross_points(image, lineList, sep_size_px):
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
                if (cross_point[0] >= 0 + 0.5*sep_size_px) and (cross_point[0] <= image.shape[1] - 0.5*sep_size_px) and (cross_point[1] >= 0) and (cross_point[1] <= image.shape[0]) :
                    #cv2.circle(image, (cross_point[0],cross_point[1]), 2, (255,0,0), 3) # 交点を青色で描画
                    cnt = cnt + 1
                    cross_points.append(cross_point)
    # 近くの座標をクラスタリングしてまとめる
    cross_points = cluster_and_round_points(cross_points)
    # x,y軸の座標をクラスタリングしてまとめる
    cross_points = cluster_x_coordinates(cross_points, eps=3)
    cross_points = cluster_y_coordinates(cross_points, eps=3)
    #for cross_point in cross_points:
        #if (cross_point[0] >= 0 + 0.5*sep_size_px) and (cross_point[0] <= image.shape[1] - 0.5*sep_size_px) and (cross_point[1] >= 0) and (cross_point[1] <= image.shape[0]) :
            #cv2.circle(image, (cross_point[0],cross_point[1]), 2, (255,0,0), 3) # 交点を青色で描画
    #print("draw_cross_points:", cnt)
    print("draw_cross_points:", len(cross_points))
    return cross_points

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

    # 水平な線分がある場合、拡張された範囲でチェックする
    if abs(pointA[1] - pointB[1]) <= 1:
        if (r < -0.5*10) or (r > 1.5*10) or (s < 0) or (s > 1):
            return False, cross_points
    elif abs(pointC[1] - pointD[1]) <= 1:
        if (r < 0) or (r > 1) or (s < -0.5*10) or (s > 1.5*10):
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

def cluster_x_coordinates(points, eps=3, min_samples=1):
    if len(points) == 0:
        return np.array(points)
    # x座標のみを取り出してクラスタリング
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

def cluster_y_coordinates(points, eps=3, min_samples=1):
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
        # クラスター内のy座標をまとめ、その平均値を新しいy座標とする
        cluster_indices = np.where(labels == label)[0]
        mean_y = np.round(np.mean(points[cluster_indices, 1])).astype(int)
        clustered_points[cluster_indices, 1] = mean_y
    
    return clustered_points 

# 一点固定後に他三点を探す
def is_square(cross_points, i, n, square, char_size_px, cropbox, tolerance):
    p1 = cross_points[i]
    square.append(p1)
    for j in range(i+1, n): # 下/左
        p2 = cross_points[j]
        d2 = np.linalg.norm(np.array(p1) - np.array(p2))
        if not abs(d2 - char_size_px) <= tolerance:
            continue
        else:
            square.append(p2)
        for k in range(j+1, n): # 左/右
            p3 = cross_points[k]
            d3 = np.linalg.norm(np.array(p1) - np.array(p3))
            if not abs(d3 - char_size_px) <= tolerance:
                continue
            else:
                square.append(p3)
            for l in range(k+1, n): # 左下/右下
                p4 = cross_points[l]
                d4 = np.linalg.norm(np.array(p1) - np.array(p4))
                if not abs(d4 - np.sqrt(2) * char_size_px) <= tolerance:
                    continue
                else:
                    square.append(p4)
                    if len(square) >= 3:
                        return square

def find_squares(cross_points, params, cropbox):
    squares = dict()
    n = len(cross_points)
    muki = params['muki']
    nchars = params['nchars']
    ncols = params['ncols']
    char_width = params['char_width']
    sep_width = params['sep_width']
    
    # ピクセルに変換する
    dpi = 600
    char_size_px = int((char_width / 2.54) * dpi)
    sep_size_px = int((sep_width / 2.54) * dpi)
    
    tolerance = 10
            
    if muki == 'tate':
        # ソート: x は降順、y は昇順
        cross_points = sorted(cross_points, key=lambda point: (-point[0], point[1]))
        # p1: 一点目
        for i in range(n):
            p1 = cross_points[i]
            # cropboxの左端or下限
            if abs(p1[0] - cropbox[0]) <= tolerance or abs(p1[1] - cropbox[3]) <= tolerance:
                continue
            else:
                square = is_square(cross_points, i, n, [], char_size_px, cropbox, tolerance)
                if square:
                    squares[i] = square   
    else:
        # ソート: y を優先して昇順、その次に x を昇順にソート
        cross_points = sorted(cross_points, key=lambda point: (point[1], point[0]))
        # p1: 一点目
        for i in range(n):
            p1 = cross_points[i]
            # cropboxの左端or下限
            if abs(p1[0] - cropbox[0]) <= tolerance or abs(p1[1] - cropbox[3]) <= tolerance:
                continue
            else:
                square = is_square(cross_points, i, n, [], char_size_px, cropbox, tolerance)
                if square:
                    squares[i] = square 
    return squares

def crop_squares(image, squares):
    cropped_images = dict()
    for i, square in squares.items():
        x_coords = [int(p[0]) for p in square]
        y_coords = [int(p[1]) for p in square]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        #if (max_x - min_x) == side_length and (max_y - min_y) == side_length:
        cropped_image = image[min_y:max_y, min_x:max_x]
        cropped_images[i] = cropped_image
    return cropped_images