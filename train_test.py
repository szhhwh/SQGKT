import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
from data_process import min_seq_len, max_seq_len
from dataset import UserDataset
from sqgkt import sqgkt
from params import *
from utils import gen_sqgkt_graph, build_adj_list, build_adj_list_uq, gen_sqgkt_graph_uq

os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
time_now = datetime.now().strftime("%Y_%m_%d#%H_%M_%S")
output_path = os.path.join("output", time_now)
os.makedirs(output_path, exist_ok=True)  # 创建目录
output_file_path = os.path.join(output_path, "log.txt")
output_file = open(output_file_path, "w")

# 训练时的超参数
params = {
    "max_seq_len": max_seq_len,
    "min_seq_len": min_seq_len,
    "epochs": 2,
    "lr": 0.01,
    "batch_size": 128,
    "size_q_neighbors": 4,
    "size_q_neighbors_2": 5,
    "size_s_neighbors": 10,
    "size_u_neighbors": 5,
    "num_workers": 0,
    "agg_hops": 3,
    "emb_dim": 100,
    "hard_recap": False,
    "dropout": (0.2, 0.4),
    "rank_k": 10,
}

# 打印并写超参数
output_file.write(str(params) + "\n")
print(params)
batch_size = params["batch_size"]

qs_table = torch.tensor(
    sparse.load_npz("data/qs_table.npz").toarray(), dtype=torch.int64, device=DEVICE
)
uq_table = torch.tensor(
    np.load("data/uq_table.npy"), dtype=torch.float32, device=DEVICE
)

num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
num_user = torch.tensor(uq_table.shape[0], device=DEVICE)

q_neighbors_list, s_neighbors_list = build_adj_list()
q_neighbors, s_neighbors = gen_sqgkt_graph(
    q_neighbors_list,
    s_neighbors_list,
    params["size_q_neighbors"],
    params["size_s_neighbors"],
)
q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)
s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)

u_neighbors_list, q_neighbors_list = build_adj_list_uq()
u_neighbors, q_neighbors_2 = gen_sqgkt_graph_uq(
    u_neighbors_list,
    q_neighbors_list,
    params["size_u_neighbors"],
    params["size_q_neighbors_2"],
)
u_neighbors = torch.tensor(u_neighbors, dtype=torch.int64, device=DEVICE)
q_neighbors_2 = torch.tensor(q_neighbors_2, dtype=torch.int64, device=DEVICE)

# 初始化模型
model = sqgkt(
    num_question,
    num_skill,
    q_neighbors,
    s_neighbors,
    qs_table,
    num_user,
    u_neighbors,
    q_neighbors_2,
    uq_table,
    agg_hops=params["agg_hops"],
    emb_dim=params["emb_dim"],
    dropout=params["dropout"],
    hard_recap=params["hard_recap"],
).to(DEVICE)

# 读取数据集
dataset = UserDataset()

# 划分训练集和测试集
# 按照0.8:0.2划分训练集和测试集
train_data_len = int(len(dataset) * 0.8)
test_data_len = len(dataset) - train_data_len
indices = np.arange(len(dataset))
np.random.shuffle(indices)
train_indices = indices[:train_data_len]
test_indices = indices[train_data_len:]
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=params["num_workers"],
    drop_last=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=params["num_workers"],
    drop_last=True,
)

optimizer = torch.optim.Adam(params=model.parameters(), lr=params["lr"])
loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE)  # 损失函数

epoch_total = 0
for epoch in range(params["epochs"]):
    train_loss_aver = train_acc_aver = train_auc_aver = 0
    test_loss_aver = test_acc_aver = test_auc_aver = 0

    print(
        "==================="
        + LOG_Y
        + f"epoch: {epoch_total + 1}"
        + LOG_END
        + "===================="
    )

    print("-------------------training------------------")
    time0 = time.time()
    train_step = train_loss = train_total = train_right = train_auc = 0

    for data in train_loader:
        optimizer.zero_grad()
        u, x, y_target, mask = (
            data[:, :, 0].to(DEVICE),
            data[:, :, 1].to(DEVICE),
            data[:, :, 2].to(DEVICE),
            data[:, :, 3].to(torch.bool).to(DEVICE),
        )
        y_hat = model(u, x, y_target, mask)
        y_hat = torch.masked_select(y_hat, mask)
        y_pred = torch.ge(y_hat, torch.tensor(0.5)).to(torch.int)
        y_target = torch.masked_select(y_target, mask)
        loss = loss_fun(y_hat, y_target.to(torch.float32))
        train_loss += loss.item()

        acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask).to(float)
        train_right += torch.sum(torch.eq(y_target, y_pred)).to(float)
        train_total += torch.sum(mask).to(float)

        auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
        train_auc += auc * len(x) / train_data_len
        loss.backward(retain_graph=True)
        optimizer.step()
        train_step += 1
        print(
            f"step: {train_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}"
        )
    train_loss, train_acc = train_loss / train_step, train_right / train_total
    train_loss_aver += train_loss
    train_acc_aver += train_acc
    train_auc_aver += train_auc

    print("-------------------testing------------------")
    test_step = test_loss = test_total = test_right = test_auc = 0

    for data in test_loader:
        u, x, y_target, mask = (
            data[:, :, 0].to(DEVICE),
            data[:, :, 1].to(DEVICE),
            data[:, :, 2].to(DEVICE),
            data[:, :, 3].to(torch.bool).to(DEVICE),
        )
        y_hat = model(u, x, y_target, mask)

        y_hat = torch.masked_select(y_hat, mask.to(torch.bool))
        y_pred = torch.ge(y_hat, torch.tensor(0.5)).to(torch.int)
        y_target = torch.masked_select(y_target, mask.to(torch.bool))
        loss = loss_fun(y_hat, y_target.to(torch.float32))
        test_loss += loss.item()

        acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask).to(float)
        test_right += torch.sum(torch.eq(y_target, y_pred)).to(float)
        test_total += torch.sum(mask).to(float)

        auc = roc_auc_score(y_target.cpu(), y_pred.cpu())

        test_auc += auc * len(x) / test_data_len
        test_step += 1
        print(
            f"step: {test_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}"
        )

    test_loss, test_acc = test_loss / test_step, float(test_right) / test_total
    test_loss_aver += test_loss
    test_acc_aver += test_acc
    test_auc_aver += test_auc

    time1 = time.time()
    run_time = time1 - time0
    epoch_total += 1

    # 打印结果
    print(
        LOG_B
        + f"training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f}"
        + LOG_END
    )
    print(
        LOG_B
        + f"testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc: .4f}"
        + LOG_END
    )
    print(
        LOG_B
        + f"epoch time: {run_time:.2f}s"
        + LOG_END
    )

    # 写入结果到文件
    output_file.write(f"epoch {epoch_total} | ")
    output_file.write(
        f"training: loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc: .4f} | "
    )
    output_file.write(
        f"testing: loss: {test_loss:.4f}, acc: {test_acc:.4f}, auc: {test_auc: .4f} | "
    )
    output_file.write(
        f"epoch time: {run_time:.2f}s\n"
    )
    output_file.flush()  # 刷新文件

output_file.close()
torch.save(model.state_dict(), f=f"model/{time_now}.pt")
