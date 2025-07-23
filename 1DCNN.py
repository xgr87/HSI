import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, cohen_kappa_score

# 超参数配置
config = {
    'k1_ratio': 1 / 9,  # k1 = floor(n1 * k1_ratio)
    'num_filters': 20,  # 卷积核数量
    'n4': 100,  # 全连接层神经元数
    'lr': 0.01,  # 学习率
    'batch_size': 16,  # 批大小
    'epochs': 200  # 训练轮数
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1D CNN模型定义
class HyperspectralCNN(nn.Module):
    def __init__(self, n1, n5):
        super().__init__()
        # 计算k1和k2
        k1 = int(n1 * config['k1_ratio'])
        k2 = int(np.ceil(k1 / 5))

        # 卷积层
        self.conv = nn.Conv1d(1, config['num_filters'], kernel_size=k1)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=k2)

        # 动态计算全连接层输入维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, n1)
            conv_out = self.conv(dummy)
            pool_out = self.pool(conv_out)
            fc_input_dim = pool_out.view(-1).shape[0]

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_dim, config['n4']),
            nn.Tanh(),
            nn.Linear(config['n4'], n5)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                nn.init.uniform_(m.bias, -0.05, 0.05)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(self.pool(x))
        x = self.fc(x)
        return x


# 数据加载函数
def load_data(image_path, label_path):
    image_data = sio.loadmat(image_path)['salinas'].astype(np.float32)
    label_data = sio.loadmat(label_path)['salinas_gt'].astype(np.int64)

    # 数据标准化到[-1, 1]
    image_data = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data)) * 2 - 1
    image_data = image_data[:,:,::10]
    # 展平数据
    X = image_data.reshape(-1, image_data.shape[-1])
    y = label_data.ravel() - 1  # 假设标签从1开始，调整为0-based

    # 移除未标记样本（标签为0）
    mask = y != -1
    X, y = X[mask], y[mask]

    # 划分训练集和测试集（10%训练）
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.1, stratify=y)
    # 从训练集中划分验证集（5%验证）
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, stratify=y_train)

    # 转换为Tensor
    train_dataset = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train))
    val_dataset = TensorDataset(torch.tensor(X_val).unsqueeze(1), torch.tensor(y_val))
    test_dataset = TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test))

    return train_dataset, val_dataset, test_dataset


# 评估函数
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)
    OA = np.sum(np.diag(cm)) / np.sum(cm)
    AA = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    kappa = cohen_kappa_score(y_true, y_pred)
    return OA, AA, kappa


# 训练函数
def train(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    best_OA = 0
    for epoch in range(config['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # 验证
        val_OA, val_AA, val_kappa = evaluate(model, val_loader)
        if val_OA > best_OA:
            best_OA = val_OA
            torch.save(model.state_dict(), 'best_model.pth')

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}/{config['epochs']} | Val OA: {val_OA:.4f} | AA: {val_AA:.4f} | Kappa: {val_kappa:.4f}")

# 主程序
if __name__ == "__main__":
    # 以PaviaU为例
    train_set, val_set, test_set = load_data('./datasets/Salinas.mat',
                                             './datasets/Salinas_gt.mat')

    # 数据加载器
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config['batch_size'])
    test_loader = DataLoader(test_set, batch_size=config['batch_size'])

    # 初始化模型
    n1 = train_set[0][0].shape[-1]
    n5 = len(torch.unique(train_set.tensors[1]))
    model = HyperspectralCNN(n1, n5).to(device)

    # 训练
    train(model, train_loader, val_loader)

    # 测试最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    test_OA, test_AA, test_kappa = evaluate(model, test_loader)
    print(f"\nTest Results | OA: {test_OA:.4f} | AA: {test_AA:.4f} | Kappa: {test_kappa:.4f}")