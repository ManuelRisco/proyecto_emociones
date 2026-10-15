# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/plots", exist_ok=True)

def render_code_figure(filename, title_text, code_lines, width=9.2, height=6.4, dpi=300):
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    ax.axis('off')
    
    bg_color = "#181818"
    fig.patch.set_facecolor(bg_color)
    
    ax.text(0.025, 0.955, f"[PYTHON 3.10] {title_text}", fontsize=9.5, fontweight="bold",
            color="#4FC1FF", transform=ax.transAxes, family="monospace")
    
    ax.plot([0.02, 0.98], [0.925, 0.925], color="#333333", lw=1.2, transform=ax.transAxes)
    
    n_lines = len(code_lines)
    y_start = 0.885
    line_spacing = 0.84 / max(n_lines, 24)
    
    for i, line in enumerate(code_lines):
        y = y_start - i * line_spacing
        
        ax.text(0.025, y, f"{i+1:2d} ", fontsize=8.0, color="#6E7681",
                transform=ax.transAxes, family="monospace", va="top")
        
        stripped = line.strip()
        t_color = "#E6EDF3"
        if stripped.startswith("#"):
            t_color = "#7EE787"
        elif any(stripped.startswith(k) for k in ["class ", "def ", "return ", "import ", "from ", "if ", "else:", "for ", "super("]):
            t_color = "#FF7B72"
        elif any(k in line for k in ["nn.Module", "nn.Conv2d", "nn.BatchNorm2d", "nn.ReLU", "nn.Linear", "nn.Sequential", "torch.Tensor", "nn.Identity", "nn.Dropout2d"]):
            t_color = "#79C0FF"
        elif '"' in line or "'" in line:
            t_color = "#A5D6FF"
            
        ax.text(0.075, y, line, fontsize=8.0, color=t_color,
                transform=ax.transAxes, family="monospace", va="top")
                
    save_path = os.path.join("outputs", "plots", filename)
    plt.savefig(save_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight", dpi=dpi)
    plt.close()
    print(f"Generated clean code figure: {save_path}")

# =============================================================================
# 1. BLOQUE DE ARQUITECTURA: EmotionResNetSE + SEBlock
# =============================================================================
code_model = [
    "class SEBlock(nn.Module):",
    "    def __init__(self, channels: int, reduction: int = 16):",
    "        super().__init__()",
    "        reduced = max(channels // reduction, 8)",
    "        self.fc = nn.Sequential(",
    "            nn.AdaptiveAvgPool2d(1),",
    "            nn.Flatten(),",
    "            nn.Linear(channels, reduced, bias=False),",
    "            nn.ReLU(inplace=True),",
    "            nn.Linear(reduced, channels, bias=False),",
    "            nn.Sigmoid()",
    "        )",
    "",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
    "        b, c, _, _ = x.size()",
    "        return x * self.fc(x).view(b, c, 1, 1)",
    "",
    "class ResidualSEBlock(nn.Module):",
    "    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):",
    "        super().__init__()",
    "        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)",
    "        self.bn1   = nn.BatchNorm2d(out_ch)",
    "        self.relu  = nn.ReLU(inplace=True)",
    "        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)",
    "        self.bn2   = nn.BatchNorm2d(out_ch)",
    "        self.se    = SEBlock(out_ch)",
    "        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()",
    "        self.shortcut = nn.Sequential(",
    "            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),",
    "            nn.BatchNorm2d(out_ch)",
    "        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()",
    "",
    "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
    "        residual = self.shortcut(x)",
    "        out = self.se(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))",
    "        out = self.dropout(out)",
    "        return self.relu(out + residual)"
]
render_code_figure("codigo_figura_emotionresnetse.png", "src/model.py — SEBlock & ResidualSEBlock", code_model, height=7.2)

# =============================================================================
# 2. BLOQUE DE COSTO: FocalLoss con Class Weights y Gamma=2.0
# =============================================================================
code_loss = [
    "class FocalLoss(nn.Module):",
    "    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, smooth: float = 0.05):",
    "        super().__init__()",
    "        self.alpha = alpha",
    "        self.gamma = gamma",
    "        self.smooth = smooth",
    "",
    "    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:",
    "        log_probs = torch.log_softmax(inputs, dim=1)",
    "        probs = torch.exp(log_probs)",
    "",
    "        if targets.dim() == 1:",
    "            num_classes = inputs.size(1)",
    "            y_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1.0)",
    "            y_smooth = (1.0 - self.smooth) * y_one_hot + (self.smooth / num_classes)",
    "        else:",
    "            y_smooth = targets",
    "",
    "        focal_weight = torch.pow(1.0 - probs, self.gamma)",
    "        loss = -focal_weight * log_probs * y_smooth",
    "",
    "        if self.alpha is not None:",
    "            loss = loss * self.alpha.view(1, -1)",
    "",
    "        return loss.sum(dim=1).mean()"
]
render_code_figure("codigo_figura_focalloss.png", "src/train.py — FocalLoss", code_loss, height=6.2)

# =============================================================================
# 3. BLOQUE DE INFERENCIA EN TIEMPO REAL: YuNet + EMA Tracker
# =============================================================================
code_realtime = [
    "def process_live_frame(frame, yunet_detector, cnn_model, subjects_db, ema_alpha=0.18):",
    "    h, w, _ = frame.shape",
    "    yunet_detector.setInputSize((w, h))",
    "    _, faces = yunet_detector.detect(frame)",
    "    if faces is None:",
    "        return frame",
    "",
    "    for face in faces:",
    "        bbox = face[0:4].astype(int)",
    "        landmarks = face[4:14].reshape((5, 2))",
    "",
    "        eye_left, eye_right = landmarks[0], landmarks[1]",
    "        angle = np.degrees(np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))",
    "        rot_matrix = cv2.getRotationMatrix2D(tuple(eye_left), angle, scale=1.0)",
    "        aligned_face = cv2.warpAffine(frame, rot_matrix, (w, h))",
    "",
    "        x, y, bw, bh = bbox",
    "        crop = cv2.resize(cv2.cvtColor(aligned_face[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY), (48, 48))",
    "        tensor_in = (clahe.apply(crop) - 128.0) / 64.0",
    "",
    "        raw_probs = (cnn_model.predict(tensor_in) + cnn_model.predict(np.fliplr(tensor_in))) / 2.0",
    "        subject_id = match_or_register_subject(crop, subjects_db, threshold=0.62)",
    "        smoothed_probs = ema_alpha * raw_probs + (1.0 - ema_alpha) * subjects_db[subject_id]['ema']",
    "        subjects_db[subject_id]['ema'] = smoothed_probs",
    "",
    "        draw_holographic_hud(frame, bbox, smoothed_probs)",
    "    return frame"
]
render_code_figure("codigo_figura_inferencia_realtime.png", "src/realtime.py — Pipeline de Inferencia en Tiempo Real", code_realtime, height=7.2)

print("Todas las figuras de código limpias generadas con éxito.")
