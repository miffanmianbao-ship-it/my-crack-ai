import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ================= 1. 网页整体设计 =================
st.set_page_config(page_title="建筑裂缝检测", page_icon="🏗️", layout="centered")
st.title("🏗️ AI 建筑裂缝智能检测系统")
st.write("请上传一张混凝土墙面或地面的照片，AI 将为您排查是否有安全隐患。")
st.divider() 

# ================= 2. 加载你刚刚训练好的大脑 =================
@st.cache_resource
def load_model():
    # 自动识别是用显卡还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 搭建骨架
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    # 加载你的专属权重文件
    model.load_state_dict(torch.load("best_crack_model.pth", map_location=device))
    model = model.to(device)
    model.eval() 
    return model, device

model, device = load_model()

# ================= 3. 图像预处理 =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ================= 4. 前端交互 =================
uploaded_file = st.file_uploader("📸 拖拽或选择一张图片 (支持 jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图像")
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, use_column_width=True)
        
    with col2:
        st.subheader("AI 诊断结果")
        with st.spinner('🔄 正在进行深度像素扫描...'):
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # 0 是无裂缝 (Negative)，1 是有裂缝 (Positive)
            pred_idx = preds[0].item()
            confidence = probabilities[pred_idx].item() * 100
            
            if pred_idx == 1:
                st.error("#### ⚠️ 警告：检测到结构裂缝！")
                st.write(f"**AI 置信度:** `{confidence:.2f}%`")
            else:
                st.success("#### ✅ 安全：未发现明显裂缝。")
                st.write(f"**AI 置信度:** `{confidence:.2f}%`")