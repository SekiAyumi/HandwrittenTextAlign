#!/usr/bin/env python
#draft to img

text_1 = '　わたしは、なつやすみに、はじめて　かぶとむしをつかまえました。はやおきを　してうらやまにいきました。てで、つかまえるとき、どきどきしました。三びきも つかまえました。　らいねんは、くわがたむしもつかまえたいと　おもいました。'

text_2 = ('　直線の数などに目をつけると、形をなかまに分けることができます。　　　　　　　　' +
          '　３本の直線でかこまれた形を、三角形といいます。４本の直線でかこまれた形を、四角形といいます。　　　　　　　　　　　　　' +
          '　三角形や四角形のまわりの直線をへん、かどの点をちょう点といいます。かどがみんな直角になっている四角形を、長方形といいます。長方形のむかい合っているへんの長さは同じです。')

text_3 = '　日本のまわりの海には、たくさんのしゅるいのさかなが、すんでいます。つめたい海がすきなさかながいます。あたたかい海がすきなさかなもいます。わたり鳥のように、きせつごとに北から南へ、南から北へと、いどうしながらくらしているさかなもいます。　　　海岸ちかくのいそにすみついているさかなや、そこのほうにいて、あまりうごきまわらないさかなもいます。かたちも、色も、もようも、じつにさまざまです。ひろい海の、どこかで、どんなふうにくらしているのか、のぞいてみましょう。'

text_4 = '　日本のまわりの海には、たくさんのしゅるいの魚が、すんでいます。冷たい海がすきな魚がいます。あたたかい海がすきな魚もいます。わたり鳥のように、季節ごとに北から南へ、南から北へと、いどうしながらくらしている魚もいます。　　　　　　　　　　　　　海岸近くのいそにすみついている魚や、底のほうにいて、あまり動きまわらない魚もいます。形も、色も、もようも、じつにさまざまです。ひろい海の、どこかで、どんなふうにくらしているのか、のぞいてみましょう。'

text_5 = '　雑木林は、燃料にするまきや炭を作るために植えられた人工の林で、落ち葉も肥料として利用されました。まきや炭をほとんどつかわなくなった今では、雑木林のもつ役わりも少なくなってきたようにみえますが、雑木林にはたくさんの生きものがすみ、命が生まれています。コナラの木１本をみても、どれだけ多くの虫がコナラの葉や実をたべていきているかがわかります。木がかれても、べつの虫がきて、そこにまた命が生まれます。雑木林は、四季をとおして楽しい観察ができます。'

text_6 = '　自由自在に空中を飛び交っている虫たちと川のなかの魚とは、一見なんの関係もないようにみえます。しかし、谷川でつったイワナの胃の中を調べてみると、虫の幼虫が、たくさんみつかります。このことから、自然のなかでは、虫の幼虫は、魚たちにとって大切なエサであることがわかります。同時に、水面からは見えにくい水中の石のうらがわや、すきまに、意外に多くの幼虫が生息していることも想像できます。実際、流れのなかの小さな石ひとつにも、思いがけずたくさんの種類と数の虫がついていて、おどろかされます。'



import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from glob import glob
import os
from torchvision import transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt    #グラフ出力用module
import math as m
import sys

#param_path = "/home/abababam1/HandwrittenTextAlign/py_cnn/4_cnn.pth"
param_path = "/home/yuxuanteng/CNN_pytorch/nlp2024/ETL_wb/ETL_wb.pth"
#param_path = "/home/yuxuanteng/CNN_pytorch/nlp2024/ETL+kodomo+d_r&c/ETL+kodomo+d_r&c.pth"
#param_path = "/home/yuxuanteng/CNN_pytorch/nlp2024/etl_wb+kodomo_real+diffusion_r&c/etl_wb+kodomo_real+d_r&c.pth"
#param_path = "/home/yuxuanteng/CNN_pytorch/nlp2024/etl_wb+diffusion_radical/etl_wb+diffusion_radical_tran550.pth"
#param_path = "/home/abababam1/HandwrittenTextAlign/kodomottAdam+SAM.pth"

#path_to_img_dir = '/data2/abababam1/HandwrittenTextAlign/kodomo_small/'
path_to_img_dir = '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/kodomo_noclass'
#all_file = glob('{path_to_img_dir}/**/**/**') # ~/grade-class/テンプレート/chara_png

#original_dir = '/data2/abababam1/cnn_png_as_uni/'
#original_dir = '/data2/abababam1/HandwrittenTextAlign/kodomotrain2/train/'
original_dir = '/data2/yuxuanteng/CNN_pytorch/ETL_images_tran/train/'
class_name = os.listdir(original_dir)
class_name = sorted(class_name)
class_dir = dict()
for i, classname in enumerate(class_name):
    class_dir[classname] = i
    
def get_keys_from_value(directory, val):
    key_list = [k for k, v in directory.items() if v == val]
    if len(key_list) == 0:
        return False
    else:
        return key_list[0]

# 前処理（例：画像のサイズ変更、正規化など）
transform_test = transforms.Compose([
    transforms.Resize((64, 63)),  # 画像のサイズ変更
    transforms.ToTensor(),           # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Net_(nn.Module): #4_cnn
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.conv3 = nn.Conv2d(32, 96, 3)
        self.conv4 = nn.Conv2d(96, 288, 3)

        self.fc1 = nn.Linear(1152, 120)#6272 = 32*14*14 #3456 = 96*6*6 #
        self.fc2 = nn.Linear(120, 3192)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
class Net_(nn.Module): #4_cnn
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3,16,3)
        self.conv2 = nn.Conv2d(16,32,3)
        self.conv3 = nn.Conv2d(32, 96, 3)
        self.conv4 = nn.Conv2d(96, 288, 3)

        self.fc1 = nn.Linear(1152, 120)#6272 = 32*14*14 #3456 = 96*6*6 #
        self.fc2 = nn.Linear(120, 3192)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Net_(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding_dropout = nn.Dropout(p = 0.005)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3,64,3)
        nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(64,128,3)
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.1)
        self.bn2 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(128,512,3)
        nn.init.normal_(self.conv3.weight, mean=0.0, std=0.1)
        self.bn3 = nn.BatchNorm2d(num_features=512)
        self.conv4 = nn.Conv2d(512,512,3)
        nn.init.normal_(self.conv4.weight, mean=0.0, std=0.1)

        self.fc1 = nn.Linear(512 * 5 * 5, 4096)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        self.fc2 = nn.Linear(4096, 4096)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        self.fc3 = nn.Linear(4096, 143)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.embedding_dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.embedding_dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.embedding_dropout(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.embedding_dropout(x)
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.embedding_dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.embedding_dropout(x)
        x = self.fc3(x)
        return x

# 2023-12-28: mtzk:
# Initialization of Conv2D parameters according to He et al. (2015)
def he_init(conv2d_layer):
    # kernel size: k1 x k2
    k1, k2 = conv2d_layer.kernel_size

    # input channel
    c = conv2d_layer.in_channels

    # number of summands
    n = k1 * k2 * c
    
    # good standard deviation : sqrt(2/n)
    std = (2 / n) ** 0.5

    # init kernel params ~ Normal(0, std^2)
    nn.init.normal_(conv2d_layer.weight, mean=0.0, std=std)

    # init bias = 0
    nn.init.zeros_(conv2d_layer.bias)

    #print(f"{c=}, {k1=}, {k2=}, {std=}")

class Net(nn.Module):
    def __init__(self,num_classes=3088):
        super(Net, self).__init__()
        # mtzk: p=0.005 is too small
        #self.embedding_dropout = nn.Dropout(p = 0.005)
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(3,64,3)
        # nn.init.normal_(self.conv1.weight, mean=0.0, std=0.1)
        he_init(self.conv1)
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(64,128,3)
        # nn.init.normal_(self.conv2.weight, mean=0.0, std=0.1)
        he_init(self.conv2)
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(128,512,3)
        # nn.init.normal_(self.conv3.weight, mean=0.0, std=0.1)
        he_init(self.conv3)
        self.bn3 = nn.BatchNorm2d(num_features=512)

        self.conv4 = nn.Conv2d(512,512,3)
        # nn.init.normal_(self.conv4.weight, mean=0.0, std=0.1)
        he_init(self.conv4)

        # mtzk: BatchNorm2d は学習パラメータがあるので層ごとに違うのを使うべき
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.fc1 = nn.Linear(512 * 5 * 5, 4096)
        # mtzk: - std changed according to He et al. (2015)
        #       - init bias = zero
        # nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc1.bias)

        # mtzk: 2023-12-28: BN also in fc
        self.bn_fc1 = nn.BatchNorm1d(num_features=4096)

        self.fc2 = nn.Linear(4096, 4096)
        # nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.fc2.bias)

        # mtzk: 2023-12-28: BN also in fc
        self.bn_fc2 = nn.BatchNorm1d(num_features=4096)

        self.fc3 = nn.Linear(4096, num_classes)
        # nn.init.normal_(self.fc3.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.fc3.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.embedding_dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.embedding_dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.embedding_dropout(x)
        x = self.conv4(x)
        # mtzk: BatchNorm2d は学習パラメータがあるので層ごとに違うのを使うべき
        # x = self.bn3(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.embedding_dropout(x)
        
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        # mtzk: 2023-12-28: BN also in fc
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.embedding_dropout(x)
        x = self.fc2(x)
        # mtzk: 2023-12-28: BN also in fc
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.embedding_dropout(x)
        x = self.fc3(x)
        return x

# モデルにパラメータ読み込み
device = torch.device("cuda:2")
model = Net()

print("Model to device .. ", end="", flush=True)
model = model.to(device)
print("done")

print("Loading parameters .. ", end="", flush=True)
# model.load_state_dict(torch.load('4_cnn.pth'))
model.load_state_dict(torch.load(param_path))
print("done")

#model = torch.load('4_cnn.pth')
model.eval()

def convert_number(number):
    small = '0123456789'
    return small[int(number)]

def max_with_index(*args):
    max_value = max(args)
    max_index = args.index(max_value)
    return max_value, max_index

def max_with_index_new(*args, count_continue_SK, count_continue_IN):
    args = list(args)
    #args[0] += SKIP_SCORE*too_many(count_continue_SK)
    #args[1] += INSERT_SCORE*too_many(count_continue_IN)
    #args[2] += CONTINUE_SPACE_SCORE*too_many_continue(count_continue_IN) + CONTINUE_SPACE_SCORE*too_many_continue(count_continue_SK)
    max_value = max(args)
    max_index = args.index(max_value)
    if max_index == 0:
        count_continue_SK += 1
        count_continue_IN = 0
    elif max_index == 1:
        count_continue_SK = 0
        count_continue_IN += 1
    else:
        assert max_index == 2
        count_continue_SK = 0
        count_continue_IN = 0
    return max_value, max_index, count_continue_SK, count_continue_IN
def too_many(count_continue):
    if count_continue + 1 > 1:
        return count_continue + 1
    else:
        return 0
def too_many_continue(count_continue):
    if count_continue > 10:
        return count_continue
    else:
        return 0

def skip1_insert1(count_continue):
    if count_continue == 1:
        re
# if direction == SKIP:
#     count_continue_SK += 1
#     count_continue_IN = 0
#     if count_continue_SK > 1:
#         path_score[i][j] += TOO_MANY*count_continue_SK
# elif direction == INSERT:
#     count_continue_IN += 1
#     count_continue_SK = 0
#     if count_continue_IN > 1:
#         path_score[i][j] += TOO_MANY*count_continue_IN
# else:
#     count_continue_SK = 0
#     count_continue_IN = 0

# logp[i][j] = P(お手本のi番目の文字 | 用紙のj番目の文字画像)
# 
# 方向
# 0 .. お手本のi文字目を飛ばした
# 1 .. 用紙のj文字目は削除（余計）
# 2 .. お手本のi文字目と用紙のj文字目がマッチ
SKIP   = 0
INSERT = 1
MATCH  = 2

# SKIP, INESRT は確率でいうと 1/10000 の文字がマッチするのと同程度に悪いということにしてみる
SKIP_SCORE   = m.log(1e-3)
SKIP_SCORE_EMPUTY = SKIP_SCORE#m.log(1e-3) # ⭐️
INSERT_SCORE = m.log(1e-3)
INSERT_SCORE_EMPUTY = INSERT_SCORE
#TOO_MANY = m.log(1e-12)
#CONTINUE_SPACE_SCORE = m.log(1e-4)

def serch_align(logp, truth_text, file_in_dir, count_continue_space):
    leni, lenj = logp.shape
    lenj -= count_continue_space
    leni = lenj
    position_index = np.zeros((leni+1, lenj+1), dtype=int)

    # mtzk: matrix とは別にパスのスコアを入れる２次元配列を用意したほうがよい（混乱しない）:
    path_score = np.zeros((leni+1, lenj+1), dtype=float)
    
    count_continue_SK = 0
    count_continue_IN = 0
    
    char_im_0 = Image.open(file_in_dir[0]) 
    char_im_0_is_255 = detect_255(char_im_0)
    char_im_0_is_0 = detect_0(char_im_0)
    
    #if logp[0][0] == 0.0: # (0, 0)に効く
    # if detect_255(char_im_0) and truth_text[0] != '　':
    if char_im_0_is_255 and truth_text[0] != '　':
        position_index[0][1] = INSERT
    # elif detect_0(char_im_0) and truth_text[0] == '　':
    elif char_im_0_is_0 and truth_text[0] == '　':
        position_index[1][0] = SKIP
        
        
    for i in range(1, leni+1):
        
        # skip_score = SKIP_SCORE_EMPUTY if detect_255(char_im_0) or truth_text[i-1] == '　' else SKIP_SCORE # ⭐️3 # どちらかが空白であればスコア抑える
        skip_score = SKIP_SCORE_EMPUTY if char_im_0_is_255 or truth_text[i-1] == '　' else SKIP_SCORE # ⭐️3 # どちらかが空白であればスコア抑える
        path_score[i][0] = path_score[i-1][0] + skip_score
        position_index[i][0] = SKIP

    for j in range(1, lenj+1):
        
        char_im = Image.open(file_in_dir[j-1])
        insert_score = INSERT_SCORE_EMPUTY if detect_255(char_im) or truth_text[0] == '　' else INSERT_SCORE # ⭐️3 # どちらかが空白であればスコア抑える
        path_score[0][j] = path_score[0][j-1] + insert_score
        position_index[0][j] = INSERT

    is_255 = []
    is_0 = []
    for j in range(1, lenj+1):
        char_im = Image.open(file_in_dir[j-1])
        is_255.append( detect_255(char_im) )
        is_0.append( detect_0(char_im) )

    for i in range(1, leni+1):
        for j in range(1, lenj+1):
            #matrix[i][j] = m.log(matrix[i][j]) if matrix[i][j] > 0 else 0

            # char_im = Image.open(file_in_dir[j-1])
            skip_score = SKIP_SCORE_EMPUTY if is_255[j-1] or truth_text[i-1] == '　' else SKIP_SCORE # ⭐️3
            insert_score = INSERT_SCORE_EMPUTY if is_255[j-1] or truth_text[i-1] == '　' else INSERT_SCORE # ⭐️3
            # if truth_text[i-1] == '　' and detect_0(char_im): # お手本が「空白」で、画像が明らかに「空白じゃないとき」はSKIP 
            if truth_text[i-1] == '　' and is_0[j-1]: # お手本が「空白」で、画像が明らかに「空白じゃないとき」はSKIP 
                position_index[i][j] = SKIP
                path_score[i][j] = path_score[i-1][j] + skip_score # (0,1)にはヒットするが
                count_continue_IN = 0
                #if count_continue_SK > 1:
                    #path_score[i][j] += SKIP_SCORE*count_continue_SK
            # elif truth_text[i-1] != '　' and detect_255(char_im): # お手本が「文字」で、画像が明らかに「空白のとき」はINSERT
            elif truth_text[i-1] != '　' and is_255[j-1]: # お手本が「文字」で、画像が明らかに「空白のとき」はINSERT
                position_index[i][j] = INSERT
                path_score[i][j] = path_score[i][j-1] + insert_score # (1,0)にはヒットするが
                count_continue_SK = 0
                #if count_continue_IN > 1:
                    #path_score[i][j] += INSERT_SCORE*count_continue_IN
            else:
                path_score[i][j], direction, count_continue_SK, count_continue_IN = max_with_index_new(
                    path_score[i-1][j] + skip_score,# + 0.05*logp[i-1][j-1],
                    path_score[i][j-1] + insert_score,# + 0.05*logp[i-1][j-1],
                    path_score[i-1][j-1] + logp[i-1][j-1],
                    count_continue_SK = count_continue_SK,
                    count_continue_IN = count_continue_IN
                )
                position_index[i][j] = direction
                
    i, j = leni, lenj
    script = []
    position_path = []
    while 0 < i or 0 < j:
        position_path.append((i, j))
        if position_index[i][j] == SKIP:
            script.append("SKIP")
            i -= 1
        elif position_index[i][j] == INSERT:
            script.append("INSERT")
            j -= 1
        else:
            assert position_index[i][j] == MATCH
            script.append("MATCH")
            i -= 1
            j -= 1
    position_path.append((0, 0))
    position_path.reverse()
    script.reverse()
    return path_score[leni][lenj], script, position_path


#お手本テキストのみの辞書
def space_of_truth(truth_align):
    truthset = set()
    for chara in truth_align:
        unicode = format(ord(chara),'#06x')
        truthset.add(unicode)
    return truthset

    
def detect_255(img):
    #img_gray = img.convert("L")
    # 画像をNumPy配列に変換
    img_array = np.array(img)
    return np.count_nonzero(img_array > 239) >= img_array.size*0.995 # 0.001くらいは白くなくてOK
    #return np.all(img_array > 200)
def detect_0(img):
    img_array = np.array(img)
    return np.count_nonzero(img_array < 208) >= img_array.size*0.01

def make_prob_array(truth_text, char_imgs):
    logp = np.zeros((len(truth_text), len(char_imgs)), dtype=float)

    # mtzk: for i と for j の順番を変えれば CNN の実行が 1/お手本の文字数 になる

    imgs_ = []
    imgs = []
    for j, chara_png_path in enumerate(char_imgs):
        img_ = Image.open(chara_png_path)
        img = img_.convert("RGB")
        img = transform_test(img) # .unsqueeze(0) # バッチ次元を追加
        imgs_.append(img_)
        imgs.append(img)

    imgs = torch.stack(imgs).to(device)

    # 画像をモデルに入力して分類結果を得る
    with torch.no_grad():
        outputs = model(imgs)

    count_continue_space = 0
    # for j, chara_png_path in enumerate(char_imgs):
    for j, (output, img_) in enumerate(zip(outputs, imgs_)):
        
        class_probabilities = F.log_softmax(output, dim=0)

        for i, char in enumerate(truth_text):
            char = ' ' if char == '　' else char
            classdir_index = format(ord(char),'#06x') # ユニコードに変換

            # 分類結果から指定したクラスの確率を取得
            if classdir_index in class_dir:
                target_class_index = class_dir[classdir_index] # ユニコードから辞書内の位置
                target_class_prob = class_probabilities[target_class_index].item()
            else:
                target_class_prob = -1
                
            logp[i][j] = target_class_prob
            
        
        # 記入なしの画像を検知 ⭐️
        if detect_255(img_):
            
            count_continue_space += 1
            
            # (0,0) お手本が「文字」で、画像が明らかに「空白のとき」　⭐️3
            #logp[0][0] = m.log(1e-13) if truth_text[0] != '　' and j == 0 else logp[0][0]
            #for i, char in enumerate(truth_text):
                # 空白と判定できた画像とお手本の空白位置にあたるところを確率を1にする 
                #logp[i][j] = m.log(m.exp(1)) if char == '　' else logp[i][j] #⭐️2 m.log(1) -> m.log(m.exp(1))
                # 空白と判定できた画像とお手本の文字にあたるところの確率を低くする #231204
                #logp[i][j] = m.log(1e-13) if char != '　' else logp[i][j]
        # 記入ありの画像を検知 ⭐️
        elif detect_0(img_):
            
            count_continue_space = 0 if count_continue_space != 0 else 0
            
            # (0,0)　お手本が「空白」で、画像が明らかに「空白じゃないとき」
            #logp[0][0] = m.log(1e-13) if truth_text[0] == '　' and j == 0 else logp[0][0]
            #for i, char in enumerate(truth_text):
                # 画像が明らかに「空白のとき」お手本の空白にあたるところの確率を低くする #231204
                #logp[i][j] = m.log(1e-13) if char == '　' else logp[i][j]
    
    return logp, count_continue_space

#手作業で合わせたjsonファイルのアラインメントを真の値とする
import os
from pathlib import Path
import shutil
import json
from glob import glob
from tqdm import tqdm
from PIL import Image


## main

def process_all(img_dir):
    elementary_school = [1, 2, 3, 4, 5, 6]
    text_plate = [text_1, text_2, text_3, text_4, text_5, text_6]
    zipped = zip(elementary_school, text_plate)
    for grade, truth_text in zipped: # 学年ごとの用紙
        # 正解テキストに予測するクラスにない文字があるか
        for char in truth_text:
            char = ' ' if char == '　' else char
            if char in '０１２３４５６７８９':
                char = convert_number(char)
            if format(ord(char),'#06x') not in class_dir:
                print(f"WARN: OutOfVocabulary char '{char}'", file=sys.stderr)
    
        grade_all_plate = glob(f"{img_dir}/{grade}/**")
        for plate_path in grade_all_plate: # 用紙
            file_in_dir = glob(f"{plate_path}/**.png")
            file_in_dir.sort()
            file_in_dir = list(file_in_dir)
    
            # prob_array = np.zeros((len(truth_text), len(file_in_dir)), dtype=object)
            prob_array, count_continue_space = make_prob_array(truth_text, file_in_dir)
    
            align_score, script, position_path = serch_align(prob_array, truth_text, file_in_dir, count_continue_space)
        
            #acc = eval_align(script, grade, ix)
            #acc_D.append(acc)
    #return acc_D
    
            return align_score, script, truth_text, file_in_dir
    
            #ocr_align_i = ''
            #ocr_align_j = ''
            #for i, j in index:
            #    label = j
            #    ocr_chara = truth_text[label] if 0 <= j < len(truth_text) else '0' #chr(int(truth_text[label], 16))
            #    ocr_align_j += ocr_chara
            #print(f'j：{ocr_align_j}', flush = True)
            #print(align_score, script)
    
            #return align_score, script, truth_text, file_in_dir

# とある学年のix番目のテキストプレートだけ
def process_one(img_dir, grade, ix):
    text_plate = [text_1, text_2, text_3, text_4, text_5, text_6]

    truth_text = text_plate[grade - 1] #　学年とテキストプレートの関係（zipしなくても）

    for char in truth_text:
        char = ' ' if char == '　' else char
        if char in '０１２３４５６７８９':
            char = convert_number(char)
        if format(ord(char),'#06x') not in class_dir:
            print(f"WARN: OutOfVocabulary char '{char}'", file=sys.stderr)

    grade_all_plate = glob(f"{img_dir}/{grade}/**")
    if not (0 <= ix < len(grade_all_plate)):
        assert False, "subject index out of bound"

    plate_path = grade_all_plate[ix]
    plate_name = os.path.basename(plate_path)
    file_in_dir = glob(f"{plate_path}/**.png")
    file_in_dir.sort()
    file_in_dir = list(file_in_dir)
    
    prob_array, count_continue_space = make_prob_array(truth_text, file_in_dir)
    
    align_score, script, position_path = serch_align(prob_array, truth_text, file_in_dir, count_continue_space)
    
    #acc = eval_align(script, grade, ix)
    
    return align_score, script, truth_text, file_in_dir

#align_score, script, truth_text, file_in_dir = process_one(1, 3, 5)
#print(align_score, script, truth_text, file_in_dir, flush = True)

if __name__ == "__main__":
    print(process_one(path_to_img_dir, 1, 0))
    
    # nlp2024 テストデータに対する評価用
    path_to_test_img_dir = '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/kodomo_charimgs_tt/test/'
    align_score, script, truth_text, file_in_dir = process_all(path_to_test_img_dir)
    import matplotlib.pyplot as plt

    # ヒストグラムを作成
    plt.hist(acc_D, bins=300)  # 'bins'はヒストグラムのバーの数を指定します

    # ヒストグラムの表示
    plt.show()
    
