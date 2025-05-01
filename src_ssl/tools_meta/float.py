import plotly.graph_objects as go
import os

# 检查并安装kaleido（若缺少该库）
try:
    import kaleido
except ImportError:
    os.system("pip install kaleido")

# ---------- 定义节点与布局 ----------
node_positions = {
    # 标题层 (Level 0)
    "Adversarial Attack Taxonomy": {"x": 0.5, "y": 0.9, "color": "#FFFFFF"},
    # 攻击分类 (Level 1)
    "Upstream Attacks": {"x": 0.2, "y": 0.7, "color": "#FFCDD2"},       # 浅红
    "Downstream Attacks": {"x": 0.5, "y": 0.7, "color": "#BBDEFB"},    # 浅蓝
    "Unknown Attacks": {"x": 0.8, "y": 0.7, "color": "#C8E6C9"},       # 浅绿
    # 攻击子类 (Level 2)
    "Feature Poisoning": {"x": 0.1, "y": 0.5, "color": "#FF8A80"},
    "Transferable Perturbations": {"x": 0.3, "y": 0.5, "color": "#FF8A80"},
    "FGSM/PGD Attacks": {"x": 0.5, "y": 0.5, "color": "#64B5F6"},
    "Black-box/Zero-day": {"x": 0.8, "y": 0.5, "color": "#81C784"},
    # 防御策略 (Level 3)
    "Robust-PFP (Upstream Defense)": {"x": 0.1, "y": 0.3, "color": "#FFF176"},  # 黄
    "AT-RPT (Downstream Defense)": {"x": 0.5, "y": 0.3, "color": "#81D4FA"},    # 浅蓝
    "Unresolved Gaps": {"x": 0.8, "y": 0.3, "color": "#EF9A9A"}                 # 红
}

edges = [
    # 主分类连接
    ("Adversarial Attack Taxonomy", "Upstream Attacks"),
    ("Adversarial Attack Taxonomy", "Downstream Attacks"),
    ("Adversarial Attack Taxonomy", "Unknown Attacks"),
    # 上游攻击连接
    ("Upstream Attacks", "Feature Poisoning"),
    ("Upstream Attacks", "Transferable Perturbations"),
    # 下游攻击连接
    ("Downstream Attacks", "FGSM/PGD Attacks"),
    # 未知攻击连接
    ("Unknown Attacks", "Black-box/Zero-day"),
    # 防御关联
    ("Feature Poisoning", "Robust-PFP (Upstream Defense)"),
    ("FGSM/PGD Attacks", "AT-RPT (Downstream Defense)"),
    ("Black-box/Zero-day", "Unresolved Gaps")
]

# ---------- 绘图 ----------
fig = go.Figure()

# 添加节点 (Text + Marker)
for node, pos in node_positions.items():
    fig.add_trace(go.Scatter(
        x=[pos["x"]], y=[pos["y"]],
        mode="markers+text",
        marker=dict(size=40, color=pos["color"], line=dict(width=2, color="black")),
        text=node,
        textfont=dict(size=14, family="Arial", color="black"),
        textposition="middle center",
        hoverinfo="none"
    ))

# 添加连接线 (带箭头)
for src, tgt in edges:
    fig.add_trace(go.Scatter(
        x=[node_positions[src]["x"], node_positions[tgt]["x"]],
        y=[node_positions[src]["y"], node_positions[tgt]["y"]],
        mode="lines",
        line=dict(color="grey", width=2),
        hoverinfo="none"
    ))

# ---------- 保存图片 ----------
fig.update_layout(

    # 添加渐变背景
    plot_bgcolor='rgba(230,244,255,0.3)',  # 浅蓝底色
    shapes=[
        # 绘制抽象河流路径（贝塞尔曲线）
        dict(
            type="path",
            path="M 0.0 0.8 L 0.4 0.4 Q 0.6 0.5, 0.8 0.4 L 1.0 0.2", 
            line=dict(color="#1890FF", width=4, dash="dot"),
            opacity=0.5
        ),
        # 深色填充区域（下游汇聚）
        dict(
            type="rect",
            x0=0.7, y0=0, x1=1.0, y1=0.4,
            fillcolor="rgba(24,144,255,0.1)",
            line=dict(width=0)
        )
    ],
    # 添加文字标注
    annotations=[
        dict(
            x=0.15, y=0.85, 
            text="Upstream (Pretraining)", 
            showarrow=False, 
            font=dict(color="#1890FF", size=12)
        ),
        dict(
            x=0.7, y=0.15, 
            text="Downstream (Deployment)", 
            showarrow=False, 
            font=dict(color="#1890FF", size=12)
        )
    ],
    title=dict(
        text="<b>Adversarial Attack Taxonomy and Defense Mapping</b>",
        x=0.5, y=0.95,
        font=dict(size=20, family="Arial")
    ),
    showlegend=False,
    template="simple_white",
    width=1200, height=800,
    margin=dict(l=50, r=50, b=50, t=100),
    xaxis=dict(showgrid=False, visible=False, range=[0,1]),
    yaxis=dict(showgrid=False, visible=False, range=[0,1])
    )

# 保存为PNG/PDF（高分辨率）
fig.write_image("attack_taxonomy.png", scale=2)  # scale=2提升分辨率
# fig.write_image("attack_taxonomy.pdf")         # 矢量图格式

print("图表已保存至当前目录: attack_taxonomy.png")
