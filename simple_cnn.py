# input is dataframe with 1 row = 1 run
# input --> convolution layer 1 --> flatten --> fully connected layer 1 --> output
# conv1 has 8 filters of the same filter size
# activation function: leaky relu (conv1, f1)
# objective function: MSE + l2 norm
# batch normalization after flatten, f1
# backprop with adam optimizer
# to avoid gradient vanishing/exploding, use variance scaling for weight init
# early stopping
# objective function: 
# Kernel size: 9
# Stride 1: 5
# Hidden number: 64
# Batch size: 512
# Dropout rate: 0.5
# Regularization coef: 0.05
# learning rate: 0.1
# learning rate decay: 0.001

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset


def process_csv_data(input_csv_path, output_csv_path):
    input_df = pd.read_csv(input_csv_path, header=None).T
    output_df = pd.read_csv(output_csv_path)
    output_df["Time label"] = output_df["Date"] + '/' + output_df["Time"]
    output_df = output_df.set_index("Time label")
    output_df = output_df.iloc[:, 3:]
    output_df = output_df.drop(columns=["Cu Error1s", "Zn Error1s", "Pb Error1s"])

    # Remove first row (labels)
    input_df = input_df.iloc[1:]
    # Change row label to be reading time
    input_df["Time label"] = input_df[0] + '/' + input_df[1]
    input_df = input_df.set_index("Time label")
    # Remove rows with exposure number = 2
    input_df = input_df[input_df[2] == "1"]
    output_df = output_df[input_df[2] == "1"]
    # Remove date, time, exposure number cols
    input_df = input_df.iloc[:, 3:]
    input_df.columns = [i for i in range(0, input_df.shape[1])]

    output_df = output_df.replace("<LOD", "0")
    input_df = input_df.astype(int)
    output_df = output_df.astype(int)

    input_df = input_df.drop(index="2026-04-27/15:34:10")
    output_df = output_df.drop(index="2026-04-27/15:34:10")

    return input_df, output_df

# ---------------------------------------------------------
# 1. Dataset Definition
# ---------------------------------------------------------
class AlignedXRFDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Shape: (N, 1, Channels)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------
# 2. Model Architecture
# ---------------------------------------------------------
class XRF1DCNN(nn.Module):

    def __init__(self, input_length: int, num_outputs: int = 3):
        super().__init__()
        # Conv1D: kernel size 9, stride 5
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=8,
            kernel_size=9,
            stride=5,
            padding=0,
        )
        self.act1 = nn.LeakyReLU(negative_slope=0.01)

        conv_out_len = (input_length - 9) // 5 + 1
        flattened_dim = 8 * conv_out_len

        self.fc_hidden = nn.Linear(flattened_dim, 64)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.1)
        self.fc_out = nn.Linear(64, num_outputs)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight,
                    a=0.01,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.act2(self.fc_hidden(x)))
        return self.fc_out(x)

# ---------------------------------------------------------
# 3. Pipeline Execution: Split, Standardize, Train, Evaluate
# ---------------------------------------------------------
# Train/test split (80/20) preserving paired samples
input_df, output_df = process_csv_data("input.csv", "output.csv")
X_raw = input_df.to_numpy(dtype=np.float32)
y_raw = output_df.to_numpy(dtype=np.float32)
target_names = output_df.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)

# Standardize inputs and targets on the training set
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

train_dataset = AlignedXRFDataset(X_train_scaled, y_train_scaled)
test_dataset = AlignedXRFDataset(X_test_scaled, y_test_scaled)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Instantiate network, loss, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_length = X_train.shape[1]
num_outputs = y_train.shape[1]

model = XRF1DCNN(input_length=input_length, num_outputs=num_outputs).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = StepLR(optimizer, step_size=1, gamma=(1.0 - 0.001))

# Lists to track loss per epoch
train_losses = []
test_losses = []

# Training loop
epochs = 120
for epoch in range(epochs):
    # --- Training Phase ---
    model.train()
    running_train_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * batch_x.size(0)

    scheduler.step()
    epoch_train_loss = running_train_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # --- Testing/Validation Phase ---
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            running_test_loss += loss.item() * batch_x.size(0)

    epoch_test_loss = running_test_loss / len(test_loader.dataset)
    test_losses.append(epoch_test_loss)

# ---------------------------------------------------------
# 4. Plot Training and Testing Loss
# ---------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(
    range(1, epochs + 1),
    train_losses,
    label="Training Loss (Scaled MSE)",
    color="#1f77b4",
    linewidth=2,
)
plt.plot(
    range(1, epochs + 1),
    test_losses,
    label="Testing Loss (Scaled MSE)",
    color="#d62728",
    linewidth=2,
)
plt.title("XRF 1D-CNN Loss per Epoch", fontsize=13)
plt.xlabel("Epoch", fontsize=11)
plt.ylabel("MSE Loss (Scaled)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. Evaluation on held-out test data
# ---------------------------------------------------------
model.eval()
all_preds_scaled = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        preds = model(batch_x)
        all_preds_scaled.append(preds.cpu().numpy())

y_pred_scaled = np.vstack(all_preds_scaled)

# Invert target scaling back to original physical units (PPM and errors)
y_pred_orig = scaler_y.inverse_transform(y_pred_scaled)

print(y_test)
print(y_pred_orig)

# Compute metrics per column
metrics_summary = []
for i, col_name in enumerate(target_names):
    mse_val = mean_squared_error(y_test[:, i], y_pred_orig[:, i])
    r2_val = r2_score(y_test[:, i], y_pred_orig[:, i])
    metrics_summary.append({"Target": col_name, "MSE": mse_val, "R^2": r2_val})

overall_mse = mean_squared_error(y_test, y_pred_orig)
overall_r2 = r2_score(y_test, y_pred_orig, multioutput="uniform_average")

metrics_df = pd.DataFrame(metrics_summary)
print("\n=== Model Performance (Original Units) ===")
print(metrics_df.to_string(index=False))
print(f"\nOverall Average MSE: {overall_mse:.4f}")
print(f"Overall Average R^2: {overall_r2:.4f}")