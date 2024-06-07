import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt    #グラフ出力用module
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset
import PIL
from PIL import Image
from tqdm import tqdm
import json
import random
import os

model_name = 'toda_for_align'

device = torch.device("cuda:0") 

json_paths = ['/data2/yuxuanteng/CNN_pytorch/binary_ETL_images_tran.json',
              '/data2/yuxuanteng/CNN_pytorch/binary_kodomo_train_new.json',
              '/data2/yuxuanteng/CNN_pytorch/binary_diffusion_radical_tran550.json'
             ]
json_paths_A = ['/data2/yuxuanteng/CNN_pytorch/ETL_images_tran.json',
              '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/0227_+3/0227_+3_tran.json'
             ]
json_paths_A_kodomo = ['/data2/yuxuanteng/CNN_pytorch/ETL_images_tran.json',
              '/data2/yuxuanteng/CNN_pytorch/kodomo_train_new.json',
              '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/0227_+3/0227_+3_tran.json',
              '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/0227_+3/0301_+3_kodomo_real_tran.json'
             ]
json_paths_A_bi = ['/data2/yuxuanteng/CNN_pytorch/binary_ETL_images_tran.json',
              '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/0227_+3/0301_+3_bi_tran.json'
             ]
# C
json_paths = ['/data2/yuxuanteng/CNN_pytorch/ETL_images_tran.json',
              '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/0227_+3/0227_+3_tran.json',
              '/data2/yuxuanteng/CNN_pytorch/diffusion_radical_tran550.json'
             ]
# 戸田
json_paths = ['/data2/yuxuanteng/CNN_pytorch/ETL_images_tran.json',
              '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/0227_+3/0227_+3_tran.json',
              '/data2/abababam1/HandwrittenTextAlign/kakko_for_toda.json',
              '/data2/yuxuanteng/CNN_pytorch/kodomo_train_new.json',
              '/data2/abababam1/HandwrittenTextAlign/nlp2024_data/0227_+3/0301_+3_kodomo_real_tran.json'
             ]

NUM_CLASSES = 3088 + 3 + 4

BATCH_SIZE = 16
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 25

# GaussianNoise
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
         
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
# 前処理（例：画像のサイズ変更、正規化など）
transform_train = transforms.Compose([
    transforms.Resize((64, 63)),  # 画像のサイズ変更
    #transforms.Grayscale(num_output_channels=1), #single-channel
    #transforms.RandomErasing(),
    #AddGaussianNoise(0.1, 0.08), # Gaussian
    # mtzk: fill はたぶん 0 にするべき
    transforms.RandomAffine(degrees=(-15, 15), scale=(0.8, 1.2), fill = 255), # 15度回転 0.8~1.2拡大, white_background
    #transforms.RandomAffine(degrees=(-15, 15), scale=(0.8, 1.2), fill = 0), # 15度回転 0.8~1.2拡大, black_background
    transforms.ToTensor(),           # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 3チャネルごとに画像の正規化 例えば極端な明度を調整する。。。など
    #transforms.Normalize((0.5,), (0.5,)), #single-channel normalization
])

transform_test = transforms.Compose([
    transforms.Resize((64, 63)),  # 画像のサイズ変更
    #transforms.Grayscale(num_output_channels=1), #single-channel
    transforms.ToTensor(),           # テンソルに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize((0.5,), (0.5,)), #single-channel normalization
])


#PATH_TO_TRAIN = '/data2/yuxuanteng/CNN_pytorch/ETL_tran/'
#PATH_TO_TEST = '/data2/yuxuanteng/CNN_pytorch/ETL/test/'

'''
#before train delete checkpoint
import os
import shutil
def remove_ipynb_checkpoints(directory):
    for root, dirs, files in os.walk(directory):
        if '.ipynb_checkpoints' in dirs:
            shutil.rmtree(os.path.join(root, '.ipynb_checkpoints'))

remove_ipynb_checkpoints(PATH_TO_TRAIN)
remove_ipynb_checkpoints(PATH_TO_TEST)

'.ipynb_checkpoints'

trainset = ImageFolder(root=PATH_TO_TRAIN, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

#testset = ImageFolder(root=PATH_TO_TEST, transform=transform_test)
#testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
'''

class CustomImageFolder(Dataset):
    '''Load from multiple jsonfile'''
    def __init__(self, json_paths, transform=None):
        self.image_paths = []
        self.labels = []

        #遍历所有json并合并数据
        for json_path in json_paths:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                self.image_paths.extend(list(data.keys()))
                self.labels.extend(list(data.values()))

        self.classes = sorted(set(self.labels))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        label_index = self.classes.index(label)

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label_index

trainset = CustomImageFolder(json_paths=json_paths, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) 
#drop_lastをTrueに指定することで、データセットサイズがバッチサイズで割り切れない場合に最後のバッチが削除されます。

def create_test_json(json_path, test_json_path, test_ratio=0.2): # 検証データセットの作成
    '''each trainset extracts 20% of the data of each class to make a test set'''
    if os.path.exists(test_json_path): #不重复制作testset
        print(f'{test_json_path} exists')
        return
    with open(json_path, 'r') as file:
        data = json.load(file)

    class_samples = {}
    for path, label in data.items():
        if label not in class_samples:
            class_samples[label] = []
        class_samples[label].append(path)

    test_samples = {}
    for label, paths in class_samples.items():
        num_test_samples = int(len(paths) * test_ratio)
        selected_paths = random.sample(paths, num_test_samples)
        for path in selected_paths:
            test_samples[path] = label

    with open(test_json_path, 'w') as file:
        json.dump(test_samples, file, indent=4)

for json_path in json_paths:
    test_json_path = json_path.replace('.json', '_test.json')
    create_test_json(json_path, test_json_path)

test_json_paths = [path.replace('.json', '_test.json') for path in json_paths]
testset = CustomImageFolder(json_paths=test_json_paths, transform=transform_test)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

'''
#每个类别的样本数量相同的情况下testset/testloader: 
num_samples_per_class = len(trainset) // len(trainset.classes)
num_test_samples_per_class = int(num_samples_per_class * 0.2)

test_indices = []
for class_idx in range(len(trainset.classes)):
    start_idx = class_idx * num_samples_per_class
    end_idx = start_idx + num_test_samples_per_class
    test_indices.extend(range(start_idx, end_idx))

testset = Subset(trainset, test_indices)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)
'''

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
    def __init__(self,num_classes=NUM_CLASSES):
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

class Net_bi(nn.Module):
    def __init__(self,num_classes=NUM_CLASSES):
        super(Net, self).__init__()
        # mtzk: p=0.005 is too small
        #self.embedding_dropout = nn.Dropout(p = 0.005)
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        #self.conv1 = nn.Conv2d(3,64,3)
        self.conv1 = nn.Conv2d(1,64,3) #single-channel
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
    

net = Net()
net = net.to(device)
criterion = nn.CrossEntropyLoss() # 兼softmax
#optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist 
best_accuracy = 0

    
for epoch in range(EPOCH):

    print('epoch', epoch+1, flush=True) #epoch数の出力
    net.train() # 訓練モードに変更

    train_sum_loss = 0

    i = 0
    for (inputs, labels) in tqdm(trainloader, ncols=80):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() # 勾配を0に初期化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_sum_loss += loss.item()
    train_mean_loss = train_sum_loss * BATCH_SIZE / len(trainloader.dataset)
    train_loss_value.append(train_mean_loss)
    print("train mean loss={}".format(train_mean_loss), flush=True)

    net.eval() # 評価モードに変更
    '''
    sum_loss = 0.0          #lossの合計
    sum_correct = 0         #正解率の合計
    sum_total = 0           #dataの数の合計

    #train dataを使ってテストをする(パラメータ更新がないようになっている)
    for (inputs, labels) in tqdm(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        #with torch.no_grad():
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()                            #lossを足していく
        _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
        sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
    print("train mean loss={}, accuracy={}"
            .format(sum_loss*BATCH_SIZE/len(trainloader.dataset), float(sum_correct/sum_total)), flush=True)  #lossとaccuracy出力
    train_loss_value.append(sum_loss*BATCH_SIZE/len(trainloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持
    '''
    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    #test dataを使ってテストをする
    with torch.no_grad():  #不计算梯度
        for (inputs, labels) in tqdm(testloader, ncols=80):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
            _, predicted = outputs.max(1)
            sum_total += labels.size(0)
            sum_correct += (predicted == labels).sum().item()
        test_mean_lost = sum_loss*BATCH_SIZE/len(testloader.dataset)
        accuracy = float(sum_correct/sum_total)
        print("test mean loss={}, accuracy={}".format(test_mean_lost, accuracy), flush=True)
        if accuracy > best_accuracy:
            torch.save(net.state_dict(), f'/home/abababam1/HandwrittenTextAlign/nlp2024/param/{model_name}.pth')
            best_accuracy = accuracy
            print("best accuracy={}".format(best_accuracy), flush=True)
        test_loss_value.append(test_mean_lost)
        test_acc_value.append(accuracy)

    scheduler.step()
    
#torch.save(net.state_dict(), 'etl.pth')
    
'''
plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("1222_loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("1222_accuracy_image.png")
'''


with open(f'/home/abababam1/HandwrittenTextAlign/nlp2024/param/info_{model_name}.txt', 'w') as file:
    file.write(f"batch size: {BATCH_SIZE}, weight decay: {WEIGHT_DECAY}, learning rate:{LEARNING_RATE}, epoch:{EPOCH}\n") 
    
    file.write(f"train set: {json_paths}\n") 
    file.write(f"test set: {test_json_paths}\n")
    
    file.write(f"best accuracy: {best_accuracy}\n")
    
    file.write(f"train_loss_value: {train_loss_value}\n")
    file.write(f"test_loss_value: {test_loss_value}\n")
    file.write(f"train_acc_value: {train_acc_value}\n")
    file.write(f"test_acc_value: {test_acc_value}\n")

