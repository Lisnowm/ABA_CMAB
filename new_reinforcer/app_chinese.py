import numpy as np
import copy
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 确保这两个文件在同目录下且没有语法错误
from new_VirtualChild import VirtualChild
from new_VirtualTherapist import VirtualTherapist

# ==========================================
# 0. 页面配置与全局 CSS 美化
# ==========================================
st.set_page_config(page_title="ABA 康复师模拟器", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 900px;}
div[data-testid="stButton"] button {border-radius: 8px; border: 1px dashed #cbd5e1; background-color: #f8fafc; color: #64748b; font-weight: 600; transition: all 0.2s; margin-top: -8px; margin-bottom: 16px; height: 40px;}
div[data-testid="stButton"] button:hover {border-color: #94a3b8; background-color: #f1f5f9; color: #0f172a; border-style: solid;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 侧边栏：个人宣传与实习求职 ---
with st.sidebar:
    st.markdown("### 👨‍💻 关于作者")
    st.markdown("欢迎体验基于强化学习（CMAB）的 ABA 康复师模拟器！")
    st.markdown("---")
    st.markdown("🎯 **Status: 寻找实习中**")
    st.markdown("我目前正在积极寻找 **202X年 暑期/日常实习**（数据科学 / 软件开发方向）。")
    st.markdown("🔗 [点击查看我的 GitHub 源码](https://github.com/Lisnowm") # 记得替换成你的真实链接
    st.markdown("📧 联系方式: lix34@gmail.com") # 记得替换成你的邮箱

# --- Configuration Section ---
REINFORCERS = [
    {'name': 'iPad', 'transition': 0.4, 'init_pref': 0.9, 'satiation_rate': 0.7, 'recovery_rate': 0.1, 'fatigue_recovery':0.02, 'icon': '📱', 'color': '#4f46e5'},
    {'name': '薯片 (Chips)', 'transition': 0.2, 'init_pref': 0.6, 'satiation_rate': 0.4, 'recovery_rate': 0.3, 'fatigue_recovery':0.25, 'icon': '🍪', 'color': '#d97706'},
    {'name': '贴纸 (Sticker)', 'transition': 0.0, 'init_pref': 0.3, 'satiation_rate': 0.9, 'recovery_rate': 0.8, 'fatigue_recovery':0.10, 'icon': '⭐', 'color': '#059669'}
]

TASKS = [
    {'name': '简单 (Easy)', 'difficulty': 0.1, 'mastery': 0.1},
    {'name': '中等 (Medium)', 'difficulty': 0.4, 'mastery': 0.1},
    {'name': '困难 (Hard)', 'difficulty': 0.8, 'mastery': 0.1}
]

def update_task_mastery(current_mastery, focus, difficulty, alpha=0.1, beta=0.05):
    noise = np.random.normal(0, 0.01)
    base_change = alpha * (1 - current_mastery) + beta * focus + noise
    learning_rate_modifier = 1.0 - (difficulty * 0.8) 
    change = base_change * learning_rate_modifier
    return np.clip(current_mastery + change, 0.0, 1.0)

def generate_squares(score, color):
    squares = ""
    for i in range(1, 6):
        bg = color if i <= score else "#e2e8f0"
        squares += f'<div style="width: 14px; height: 14px; background-color: {bg}; border-radius: 3px; display: inline-block; margin-right: 4px;"></div>'
    return squares

# ==========================================
# 1. 状态初始化与回调函数
# ==========================================
if 'app_state' not in st.session_state:
    st.session_state.app_state = 'selection'

def select_profile_callback(profile_type):
    st.session_state.current_profile = profile_type
    st.session_state.trial = 1
    st.session_state.current_reinforcers = copy.deepcopy(REINFORCERS)
    st.session_state.current_tasks = copy.deepcopy(TASKS)
    
    st.session_state.child = VirtualChild(profile_type, st.session_state.current_reinforcers)
    st.session_state.therapist = VirtualTherapist(n_arms=len(REINFORCERS), n_features=9)
    
    # 根据选择的儿童画像初始化基线临床状态
    if profile_type == 'novelty_seeker':
        st.session_state.emotion = 4.0     # 情绪高涨，充满好奇
        st.session_state.fatigue = 0.0     # 精力充沛，无初始疲劳
        st.session_state.focus = 0.7       # 注意力较好（被新环境吸引）
        st.session_state.compliance = 0.8  # 初始配合度高
        st.session_state.resistance = 0.1  # 几乎没有抗拒行为
        
    elif profile_type == 'low_endurance':
        st.session_state.emotion = 3.0     # 情绪稳定但略显迟钝
        st.session_state.fatigue = 0.2     # [关键] 初始带有轻微疲劳，反映“低电量”状态
        st.session_state.focus = 0.5       # 注意力一般
        st.session_state.compliance = 0.6  # 配合度尚可，但容易下降
        st.session_state.resistance = 0.2  # 轻度抗拒（因为觉得累）
        
    elif profile_type == 'rigid':
        st.session_state.emotion = 2.0     # 情绪低落 / 防备心重
        st.session_state.fatigue = 0.0     # 身体不疲劳
        st.session_state.focus = 0.3       # 沉浸在自我刺激中，对康复师关注度低
        st.session_state.compliance = 0.2  # 极度不配合（除非给出特定的偏好物）
        st.session_state.resistance = 0.8  # 初始防备和抗拒度极高

    st.session_state.desire = {r['name']: int(r['init_pref'] * 100) for r in st.session_state.child.reinforcers}
    
    st.session_state.cum_reward = 0
    st.session_state.history = []
    st.session_state.suggested_idx = 0
    
    st.session_state.app_state = 'simulation'

def go_back_callback():
    st.session_state.app_state = 'selection'

def execute_trial_callback(choice_name):
    chosen_idx = next(i for i, r in enumerate(st.session_state.current_reinforcers) if r['name'] == choice_name)
    task_idx = np.random.choice(len(st.session_state.current_tasks))
    current_task = st.session_state.current_tasks[task_idx] 
    
    child_state = st.session_state.child.get_state() 

    # 基于上下文选择强化物
    suggested_idx, context_vector = st.session_state.therapist.choose_reinforcer(
        child_state, current_task['mastery'], current_task['difficulty'], st.session_state.current_reinforcers
    )
    st.session_state.suggested_idx = suggested_idx
    
    # 更新情绪、配合度等状态
    discrete_emo, discrete_com, discrete_res_score, emotion, compliance, resistance, focus, fatigue = st.session_state.child.react(
        current_task['difficulty'], chosen_idx
    )
    
    actual_reward = (10.0 * focus) + (8.0 * discrete_com) + (5.0 * discrete_emo) - (8.0 * discrete_res_score)
    st.session_state.therapist.update_strategy(context_vector, actual_reward)
    
    # 更新任务掌握度
    st.session_state.current_tasks[task_idx]['mastery'] = update_task_mastery(
        current_task['mastery'], focus, current_task['difficulty']
    )
    st.session_state.child.update_internal_states(chosen_idx, st.session_state.current_reinforcers[chosen_idx]['fatigue_recovery'])
    
    st.session_state.emotion = max(1, min(5, int((emotion + 1) * 2) + 1))
    st.session_state.compliance = max(1, min(5, int((compliance + 1) * 2) + 1))
    st.session_state.focus = max(1, min(5, int(focus * 4) + 1))
    st.session_state.resistance = max(1, min(5, int(resistance * 4) + 1))
    st.session_state.fatigue = int(fatigue * 100)
    
    new_state = st.session_state.child.get_state()
    for i, r in enumerate(st.session_state.current_reinforcers):
        st.session_state.desire[r['name']] = int(new_state['preferences'][i] * 100)
    
    st.session_state.cum_reward += actual_reward
    st.session_state.history.append({'trial': st.session_state.trial, 'reward': st.session_state.cum_reward})
    st.session_state.trial += 1


# ==========================================
# 2. 页面路由与 UI 渲染
# ==========================================

if st.session_state.app_state == 'selection':
    st.markdown("<h1 style='text-align: center; color: #1e293b; margin-bottom: 2rem;'>🧩 ABA 康复师模拟器</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>🌟 追求新奇型<br><span style='font-size: 1rem; color: #64748b;'>Novelty Seeker</span></h3>", unsafe_allow_html=True)
            st.button("选择此画像", key="btn_novel", use_container_width=True, on_click=select_profile_callback, args=('novelty_seeker',))
    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>🔋 易疲劳型<br><span style='font-size: 1rem; color: #64748b;'>Low Endurance</span></h3>", unsafe_allow_html=True)
            st.button("选择此画像", key="btn_low", use_container_width=True, on_click=select_profile_callback, args=('low_endurance',))
    with col3:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>🧱 刻板固执型<br><span style='font-size: 1rem; color: #64748b;'>Rigid</span></h3>", unsafe_allow_html=True)
            st.button("选择此画像", key="btn_rigid", use_container_width=True, on_click=select_profile_callback, args=('rigid',))

elif st.session_state.app_state == 'simulation':
    
    st.button("⬅️ 返回重选", on_click=go_back_callback)

    # ---------------------------------------------------------
    # Section 1: Therapist Assessment (Simulated)
    # ---------------------------------------------------------
    assessment_html = f"""
    <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 8px; color: #1e3a8a; font-weight: 700; font-size: 1.05rem; margin-bottom: 4px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
            康复师临床评估 (模拟生成)
        </div>
        <div style="color: #60a5fa; font-size: 0.8rem; margin-bottom: 20px;">*在实际临床部署中，康复师会手动输入这些 1-5 的评分。此处由随机用户模型自动生成。</div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;"><div style="width: 150px; color: #64748b; font-size: 0.95rem;">情绪 (Emotion)</div><div style="flex-grow: 1;">{generate_squares(st.session_state.emotion, "#4ade80")}</div><div style="font-weight: 700; color: #0f172a;">{st.session_state.emotion}</div></div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;"><div style="width: 150px; color: #64748b; font-size: 0.95rem;">配合度 (Compliance)</div><div style="flex-grow: 1;">{generate_squares(st.session_state.compliance, "#60a5fa")}</div><div style="font-weight: 700; color: #0f172a;">{st.session_state.compliance}</div></div>
        <div style="display: flex; align-items: center;"><div style="width: 150px; color: #64748b; font-size: 0.95rem;">抗拒度 (Resistance)</div><div style="flex-grow: 1;">{generate_squares(st.session_state.resistance, "#f87171")}</div><div style="font-weight: 700; color: #0f172a;">{st.session_state.resistance}</div></div>
    </div>
    """
    st.markdown(assessment_html, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Section 2: Internal State: Fatigue
    # ---------------------------------------------------------
    fatigue_html = f"""
    <div style="background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin-bottom: 25px;">
        <div style="display: flex; align-items: center; gap: 8px; color: #334155; font-weight: 700; font-size: 1rem; margin-bottom: 15px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ea580c" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            儿童内在状态：疲劳值 (Fatigue)
        </div>
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="flex-grow: 1; background-color: #e2e8f0; border-radius: 999px; height: 12px; overflow: hidden;">
                <div style="background-color: #f97316; height: 100%; width: {st.session_state.fatigue}%; border-radius: 999px;"></div>
            </div>
            <div style="color: #64748b; font-size: 0.9rem; font-weight: 600; min-width: 40px; text-align: right;">{st.session_state.fatigue}%</div>
        </div>
    </div>
    """
    st.markdown(fatigue_html, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Section 3: Select Reinforcer
    # ---------------------------------------------------------
    header_html = f"""
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 0 5px;">
        <div style="font-weight: 700; color: #1e293b; font-size: 1.15rem;">选择强化物 (第 {st.session_state.trial} 回合)</div>
        <div style="display: flex; align-items: center; gap: 8px; font-size: 0.85rem; color: #64748b;">
            AI 辅助: <span style="background-color: #dbeafe; color: #2563eb; padding: 3px 12px; border-radius: 999px; font-weight: 700;">已开启</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    card_styles = [
        {'bg': '#eef2ff', 'border': '#c7d2fe', 'text': '#4f46e5', 'badge_bg': '#4f46e5', 'badge_text': 'white'},
        {'bg': '#fffbeb', 'border': '#fde68a', 'text': '#d97706', 'badge_bg': '#d97706', 'badge_text': 'white'},
        {'bg': '#ecfdf5', 'border': '#a7f3d0', 'text': '#059669', 'badge_bg': '#059669', 'badge_text': 'white'}
    ]

    for idx, r in enumerate(st.session_state.current_reinforcers):
        style = card_styles[idx % len(card_styles)]
        desire = st.session_state.desire[r['name']]
        is_suggested = (st.session_state.suggested_idx == idx)
        
        if is_suggested:
            right_content = f"""
            <div style="background-color: {style["badge_bg"]}; color: {style["badge_text"]}; padding: 4px 12px; border-radius: 6px; font-size: 0.8rem; font-weight: 700; margin-bottom: 4px; display: inline-block;">算法推荐</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: {style["text"]};">{desire}%</div>
            """
        else:
            right_content = f"""
            <div style="color: {style["text"]}; font-size: 0.8rem; margin-bottom: 4px; opacity: 0.8; font-weight: 600;">当前偏好度</div>
            <div style="font-size: 1.3rem; font-weight: 800; color: {style["text"]};">{desire}%</div>
            """

        card_html = f"""
        <div style="background-color: {style["bg"]}; border: 2px solid {style["border"]}; border-radius: 10px; padding: 18px 24px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <div style="display: flex; align-items: center; gap: 15px; color: {style["text"]}; font-size: 1.3rem; font-weight: 700;">
                <span>{r["icon"]}</span> {r["name"]}
            </div>
            <div style="text-align: right;">{right_content}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        st.button(f"给予 {r['name']}", key=f"btn_{r['name']}_{st.session_state.trial}", use_container_width=True, on_click=execute_trial_callback, args=(r['name'],))

    # ---------------------------------------------------------
    # Section 4: Cumulative Reward Chart
    # ---------------------------------------------------------
    if len(st.session_state.history) > 0:
        chart_header_html = '<div style="font-weight: 700; color: #1e293b; font-size: 1.15rem; margin-bottom: 10px; margin-top: 25px; padding: 0 5px;">累计奖励 (全局优化表现)</div>'
        st.markdown(chart_header_html, unsafe_allow_html=True)
        
        df = pd.DataFrame(st.session_state.history)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['trial'], 
            y=df['reward'],
            mode='lines',
            line=dict(color='#3b82f6', width=3),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        fig.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            height=250,
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9', title="回合 (Trial)"),
            yaxis=dict(showgrid=True, gridcolor='#f1f5f9')
        )
        st.plotly_chart(fig, use_container_width=True)
