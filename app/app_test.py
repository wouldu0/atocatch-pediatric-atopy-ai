# -*- coding: utf-8 -*-
import streamlit as st
import textwrap
import time
import os
import json
import hashlib
import joblib
import pandas as pd
import numpy as np
import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import requests
import base64
from datetime import datetime
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

def clean_html(html_str):
    return " ".join(line.strip() for line in html_str.splitlines() if line.strip())




# ==========================================
# 0. 초기 설정 및 모델, 상수 정의
# ==========================================
st.set_page_config(layout="wide", page_title="AtoCatch", initial_sidebar_state="collapsed")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE   = os.path.join(_BASE_DIR, "atocatch_users.json")
HIST_FILE = os.path.join(_BASE_DIR, "atocatch_history.json")

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

try:
    from gradcam_module import predict_with_gradcam
    GRADCAM_OK = True
except ImportError:
    GRADCAM_OK = False

# DB 관리 함수
def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_json(path, data):
    import tempfile
    dir_name = os.path.dirname(path)
    try:
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if os.path.exists(path):
            try: os.replace(path, path + ".bak")
            except: pass
        os.rename(tmp_path, path)
        if os.path.exists(path + ".bak"):
            try: os.remove(path + ".bak")
            except: pass
    except Exception:
        pass

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
def load_users(): return _load_json(DB_FILE)
def save_users(data): _save_json(DB_FILE, data)
def load_history(): return _load_json(HIST_FILE)
def save_history(data): _save_json(HIST_FILE, data)

def add_history(username, record_type, detail, extra=None):
    try:
        hist = load_history()
        if username not in hist: hist[username] = []
        entry = {
            "type": record_type,
            "detail": detail,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        if extra: entry.update(extra)
        hist[username].append(entry)
        hist[username] = hist[username][-50:]
        save_history(hist)
    except: pass

def generate_html_report(display_name, time_str, detail, image_b64=None, gradcam_b64=None):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>AtoCatch 아토피 분석 결과 보고서</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
            body {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #1e293b;
                margin: 0;
                padding: 40px;
                background-color: #f8fafc;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
                border: 1px solid #e2e8f0;
            }}
            .header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 2px solid #1b6554;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .logo {{
                font-size: 24px;
                font-weight: 800;
                color: #1b6554;
            }}
            .doc-title {{
                font-size: 16px;
                color: #64748b;
                font-weight: 600;
            }}
            .meta-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
                background: #f0f7f4;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #d1e5de;
            }}
            .meta-item span {{
                display: block;
                font-size: 12px;
                color: #64748b;
                margin-bottom: 4px;
            }}
            .meta-item strong {{
                font-size: 16px;
                color: #0f172a;
            }}
            .section-title {{
                font-size: 18px;
                font-weight: 700;
                color: #1b6554;
                margin-top: 30px;
                margin-bottom: 15px;
                border-left: 4px solid #1b6554;
                padding-left: 10px;
            }}
            .result-box {{
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                padding: 20px;
                border-radius: 12px;
                font-size: 18px;
                font-weight: 700;
                color: #0f172a;
                margin-bottom: 30px;
                text-align: center;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .image-card {{
                text-align: center;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                padding: 15px;
                border-radius: 12px;
            }}
            .image-card img {{
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                border: 1px solid #cbd5e1;
            }}
            .image-caption {{
                font-size: 13px;
                color: #64748b;
                margin-top: 10px;
                font-weight: 600;
            }}
            .guide-box {{
                background: #f0fdf4;
                border: 1px solid #d1e5de;
                padding: 20px;
                border-radius: 12px;
                margin-top: 30px;
            }}
            .guide-box h4 {{
                margin: 0 0 10px 0;
                color: #1b6554;
            }}
            .guide-box p {{
                margin: 0;
                font-size: 14px;
                color: #334155;
                line-height: 1.6;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 12px;
                color: #94a3b8;
                border-top: 1px solid #e2e8f0;
                padding-top: 20px;
            }}
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                .container {{
                    box-shadow: none;
                    border: none;
                    padding: 0;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="logo">☻ AtoCatch</div>
                <div class="doc-title">AI 피부 분석 결과 보고서</div>
            </div>
            
            <div class="meta-grid">
                <div class="meta-item">
                    <span>환자(보호자)명</span>
                    <strong>{display_name} 님</strong>
                </div>
                <div class="meta-item">
                    <span>분석 일시</span>
                    <strong>{time_str}</strong>
                </div>
            </div>
            
            <div class="section-title">종합 분석 소견</div>
            <div class="result-box">
                {detail}
            </div>
            
            <div class="section-title">피부 분석 이미지</div>
            <div class="image-grid">
    """
    if image_b64:
        html_content += f"""
                <div class="image-card">
                    <img src="data:image/jpeg;base64,{image_b64}" alt="원본 이미지">
                    <div class="image-caption">원본 피부 사진</div>
                </div>
        """
    if gradcam_b64:
        html_content += f"""
                <div class="image-card">
                    <img src="data:image/jpeg;base64,{gradcam_b64}" alt="Grad-CAM">
                    <div class="image-caption">AI 분석 히트맵 (Grad-CAM)</div>
                </div>
        """
    html_content += f"""
            </div>
            
            <div class="guide-box">
                <h4>📌 아토피 케어 생활 수칙 가이드</h4>
                <p>
                    1. <strong>철저한 보습 관리</strong>: 목욕 후 3분 이내에 무향, 저자극성 보습제를 하루 2회 이상 충분히 도포해 주세요.<br>
                    2. <strong>적절한 실내 온도/습도 유지</strong>: 온도 20-22°C, 습도 50-60%를 유지하여 건조함을 예방해 주세요.<br>
                    3. <strong>자극물 노출 차단</strong>: 100% 면 소재 의류를 착용하고, 피부를 긁지 않도록 손톱을 짧고 깨끗하게 관리해 주세요.<br>
                    4. <strong>전문의 상담</strong>: 본 보고서는 AI 예측 수치이므로 정확한 진단과 약물 치료는 소아청소년과 또는 피부과 전문의와 상담하시기 바랍니다.
                </p>
            </div>
            
            <div class="footer">
                본 보고서는 AtoCatch AI 멀티모달 솔루션에 의해 생성되었습니다. © 2026 AtoCatch. All rights reserved.
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# 모델 로딩
@st.cache_resource
def load_risk_model(): return joblib.load("atopy_service_model.joblib")

@st.cache_resource
def load_image_model():
    m = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=2)
    try:
        m.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    except: pass
    m.eval()
    return m

@st.cache_resource
def load_iga_model():
    m = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=2)
    try:
        m.load_state_dict(torch.load("best_iga_model.pth", map_location="cpu"))
    except: pass
    m.eval()
    return m

risk_model  = load_risk_model()
image_model = load_image_model()
iga_model   = load_iga_model()

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

AB_MAP = {"없음 (0회)": 0, "1~2회": 1, "3~4회": 2, "5회 이상": 3}
FACTORS = {
    "antibiotic": {"label":"항생제 복용 이력 (3회↑)", "sig":True},
    "parent_AD": {"label":"부모 아토피 진단", "sig":True},
    "child_passive_smoke": {"label":"아동 간접흡연", "sig":True},
    "parent_AR": {"label":"부모 알레르기 비염", "sig":True},
    "parent_asthma": {"label":"부모 천식", "sig":False},
    "sibling_allergy": {"label":"형제자매 알레르기", "sig":False},
    "mold_ever": {"label":"실내 곰팡이 노출", "sig":False},
    "pet_ever": {"label":"반려동물 양육", "sig":False},
    "passive_smoke_ever": {"label":"가족 흡연 노출 (누적)", "sig":False},
}
ADVICE = {
    "antibiotic": ("⚠️ 항생제 3회 이상 복용은 아토피 위험을 높입니다. 꼭 필요한 경우에만 사용하고 의사와 상의하세요.", "✅ 항생제 복용 이력이 적습니다."),
    "parent_AD": ("⚠️ 부모 아토피 병력 시 발병 위험이 높아집니다. 보습을 철저히 하세요.", "✅ 부모 아토피 병력이 없습니다."),
    "child_passive_smoke": ("⚠️ 간접흡연은 아토피 위험을 높입니다. 완전 금연 환경을 만드세요.", "✅ 간접흡연 노출이 없습니다."),
    "parent_AR": ("⚠️ 부모 알레르기 비염 시 아토피 위험이 증가합니다. 공기질 관리에 신경 쓰세요.", "✅ 부모 알레르기 비염 병력이 없습니다."),
    "mold_ever": ("⚠️ 실내 곰팡이는 아토피를 악화시킬 수 있습니다. 환기를 생활화하세요.", "✅ 실내 곰팡이 노출이 없습니다."),
    "parent_asthma": ("⚠️ 부모 천식이 있으면 알레르기 소인이 높아질 수 있습니다.", "✅ 부모 천식 병력이 없습니다."),
    "sibling_allergy": ("⚠️ 형제자매 알레르기가 있으면 알레르겐 관리가 중요합니다.", "✅ 형제자매 알레르기 병력이 없습니다."),
    "pet_ever": ("ℹ️ 반려동물 양육이 연관될 수 있습니다.", "✅ 반려동물 관련 소견 없습니다."),
    "passive_smoke_ever": ("⚠️ 가족 흡연 노출이 있었습니다. 금연 유지가 중요합니다.", "✅ 가족 흡연 누적 노출이 없습니다."),
}

# ----------------- 세션 상태 관리 -----------------
if 'user_db' not in st.session_state: st.session_state.user_db = load_users()
if 'is_logged_in' not in st.session_state: st.session_state.is_logged_in = False
if 'auth_page' not in st.session_state: st.session_state.auth_page = 'login'
if 'current_page' not in st.session_state: st.session_state.current_page = "홈"
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'survey_result' not in st.session_state: st.session_state.survey_result = None
if 'img_result' not in st.session_state: st.session_state.img_result = None

# ==========================================
# 1. 인증 화면 (로그인 / 회원가입)
# ==========================================
def apply_auth_css():
    bg_img_path = os.path.join(_BASE_DIR, "design", "bg_main.png")
    bg_b64 = ""
    if os.path.exists(bg_img_path):
        with open(bg_img_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
            
    bg_style = f"background: url('data:image/png;base64,{bg_b64}') center/cover no-repeat !important;" if bg_b64 else "background: linear-gradient(135deg, #1B6554 0%, #103F34 100%);"
    
    st.markdown(clean_html(f"""
    <style>
        /* 전체 화면 배경화면 (이미지 깨짐 방지 및 여백 제거) */
        [data-testid="stApp"] {{
            {bg_style}
        }}
        
        /* 메인 컨테이너 (화면을 꽉 차게 쓰되 상단/좌측으로 밀착) */
        .block-container {{ 
            padding-top: 5vh !important; 
            padding-bottom: 5vh !important; 
            max-width: 95% !important; 
        }}
        header {{ visibility: hidden !important; }}
        
        /* 첫 번째 컬럼(왼쪽) 반투명 폼 카드 - 크기 확대 */
        [data-testid="stColumn"]:nth-of-type(1) {{
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(14px);
            padding: 2.8rem 2.5rem !important;
            border-radius: 24px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.15);
            border: 1px solid rgba(255, 255, 255, 0.8);
            max-width: 520px !important;
            margin-left: 2vw;
            margin-top: 0 !important;
        }}
        
        /* 오른쪽 컬럼은 배경이 투명하게 비워두기 */
        [data-testid="stColumn"]:nth-of-type(2) {{
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }}
        
        /* 공통 인풋 스타일 (컴팩트하게 축소) */
        .stTextInput > div > div > input {{ background-color: rgba(248, 250, 252, 0.9) !important; border: 1px solid #E2E8F0 !important; border-radius: 12px !important; padding: 14px 16px !important; transition: 0.2s; font-size: 1.1rem !important; }}
        .stTextInput > div > div > input:focus {{ border-color: #1B6554 !important; background-color: white !important; box-shadow: 0 0 0 2px rgba(27,101,84,0.2) !important; }}
        .stTextInput label {{ font-size: 1.0rem !important; color: #334155 !important; font-weight: 700 !important; margin-bottom: 6px !important; }}
        
        /* 모든 버튼 공통 스타일 축소 */
        div.stButton > button {{ border-radius: 12px !important; height: 54px !important; font-size: 1.1rem !important; font-weight: 700 !important; transition: 0.2s !important; margin-top: 12px !important; width: 100% !important; }}
        
        /* Primary 버튼 (로그인/가입 메인) 강제 덮어쓰기 (가장 강력한 타겟팅) */
        div.stButton > button[kind="primary"], button[data-testid="baseButton-primary"] {{ 
            background-color: #1B6554 !important; 
            color: white !important; 
            border: 2px solid #1B6554 !important; 
        }}
        div.stButton > button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {{ 
            background-color: #144E41 !important; 
            border-color: #144E41 !important;
            color: white !important; 
        }}
        
        /* Secondary 버튼 */
        div.stButton > button[kind="secondary"], button[data-testid="baseButton-secondary"] {{ 
            background-color: transparent !important; 
            color: #1B6554 !important; 
            border: 1.5px solid #1B6554 !important; 
        }}
        div.stButton > button[kind="secondary"]:hover, button[data-testid="baseButton-secondary"]:hover {{ 
            background-color: #F0FDF4 !important; 
            color: #1B6554 !important; 
        }}
        
        /* 카드 내부 텍스트 디자인 (폰트 사이즈 축소) */
        .card-badge {{
            background-color: #1B6554; color: white; padding: 5px 14px; border-radius: 15px; font-size: 0.88rem; font-weight: 700; margin-bottom: 1rem; display: inline-block; letter-spacing: 0.5px;
        }}
        .card-title {{ font-size: 1.8rem; font-weight: 800; line-height: 1.35; margin-bottom: 0.6rem; color: #111; }}
        .card-desc {{ font-size: 1.0rem; font-weight: 500; opacity: 0.8; line-height: 1.6; color: #333; margin-bottom: 1.8rem; }}
        
        .logo-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 1rem; font-size: 1.2rem; font-weight: 800; color: #1B6554; border-top: 1px solid #E2E8F0; padding-top: 1rem; }}
        .logo-icon {{ background-color: #1B6554; color: white; width: 26px; height: 26px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; }}
        
        /* 로고 이미지 중앙 정렬 */
        [data-testid="stImage"] {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1.5rem;
        }}
        [data-testid="stImage"] img {{
            max-width: 180px !important;
        }}
    </style>
    """), unsafe_allow_html=True)

def render_login():
    apply_auth_css()
    
    # 레이아웃: 비율을 0.8:2.5로 조정하여 왼쪽 카드가 화면의 약 25%만 차지하도록 대폭 축소
    col_card, col_empty = st.columns([1.0, 2.3])
    
    with col_card:
        st.markdown(clean_html("""
        <div><span class="card-badge">⛨ AI Healthcare</span></div>
        <div class="card-title">아토피 잡는 AI, AtoCatch<br>우리 아기 피부 안심을 위한 첫걸음</div>
        <div class="card-desc">AtoCatch AI가 미세한 피부 변화를 분석하여 <br>아토피 위험을 조기 예측하고 맞춤형 케어를 제안합니다.</div>
        <div style="border-top: 1px solid #E2E8F0; margin-top: 1rem; margin-bottom: 1.5rem;"></div>
        """), unsafe_allow_html=True)
        
        logo_path = os.path.join(_BASE_DIR, "design", "logo_main.png")
        if os.path.exists(logo_path):
            import base64
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<div style="display: flex; justify-content: center; margin-bottom: 1.8rem;">'
                f'<img src="data:image/png;base64,{logo_b64}" style="width: 180px;">'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="logo-header"><div class="logo-icon">☻</div>AtoCatch 로그인</div>', unsafe_allow_html=True)

        
        email = st.text_input("이메일 주소", placeholder="name@example.com")
        password = st.text_input("비밀번호", type="password", placeholder="••••••••")
        
        if st.button("로그인", use_container_width=True, type="primary"):
            if email in st.session_state.user_db and st.session_state.user_db[email].get("password") == hash_pw(password):
                st.session_state.is_logged_in = True
                st.session_state.username = email
                st.session_state.display_name = st.session_state.user_db[email].get("name", email)
                st.rerun()
            else:
                st.error("이메일 또는 비밀번호가 일치하지 않습니다.")
        
        st.write("")
        if st.button("아직 회원이 아니신가요? 가입하기", use_container_width=True, type="secondary"):
            st.session_state.auth_page = 'signup'
            st.rerun()
            
    with col_empty:
        # 배경 이미지가 잘 보이도록 오른쪽 영역은 비워둡니다.
        st.empty()

def render_signup():
    apply_auth_css()
    
    # 레이아웃: 비율을 0.8:2.5로 조정하여 왼쪽 카드가 화면의 약 25%만 차지하도록 대폭 축소
    col_card, col_empty = st.columns([1.0, 2.3])
    
    with col_card:
        st.markdown(clean_html("""
        <div><span class="card-badge">⛨ AI Infant Healthcare</span></div>
        <div class="card-title">우리아이 피부건강,<br>지금 바로 시작하세요</div>
        <div class="card-desc">AtoCatch 회원가입을 통해 <br>미세한 피부 변화를 체계적으로 관리하세요.</div>
        <div style="border-top: 1px solid #E2E8F0; margin-top: 1rem; margin-bottom: 1.5rem;"></div>
        """), unsafe_allow_html=True)
        
        logo_path = os.path.join(_BASE_DIR, "design", "logo_main.png")
        if os.path.exists(logo_path):
            import base64
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            st.markdown(
                f'<div style="display: flex; justify-content: center; margin-bottom: 1.8rem;">'
                f'<img src="data:image/png;base64,{logo_b64}" style="width: 180px;">'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="logo-header"><div class="logo-icon">☻</div>AtoCatch 회원가입</div>', unsafe_allow_html=True)
        
        name = st.text_input("보호자 이름", placeholder="이름을 입력하세요")
        email = st.text_input("이메일 주소", placeholder="name@example.com")
        password = st.text_input("비밀번호", type="password", placeholder="8자리 이상")
        password_confirm = st.text_input("비밀번호 확인", type="password", placeholder="비밀번호 다시 입력")
        
        agree = st.checkbox("이용약관 및 개인정보 처리방침에 동의합니다.")
        
        if st.button("가입하기", use_container_width=True, type="primary"):
            if password != password_confirm:
                st.error("비밀번호가 일치하지 않습니다.")
            elif not agree:
                st.warning("약관에 동의해 주세요.")
            elif email in st.session_state.user_db:
                st.error("이미 존재하는 이메일입니다.")
            elif not email or not password:
                st.error("모든 항목을 입력해 주세요.")
            else:
                st.session_state.user_db[email] = {"password": hash_pw(password), "name": name}
                save_users(st.session_state.user_db)
                st.success("🎉 회원가입 완료! 로그인 페이지로 이동합니다.")
                time.sleep(1.5)
                st.session_state.auth_page = 'login'
                st.rerun()
                
        st.write("")
        if st.button("이미 계정이 있으신가요? 로그인", use_container_width=True, type="secondary"):
            st.session_state.auth_page = 'login'
            st.rerun()
            
    with col_empty:
        st.empty()

# ==========================================
# 2. 메인 앱 화면 (로그인 이후)
# ==========================================
def apply_main_css():
    logo_path = os.path.join(_BASE_DIR, "design", "logo_main.png")
    logo_b64 = ""
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()

    if logo_b64:
        st.markdown(clean_html(f"""
        <style>
            /* [ centered 로고 버튼 스타일 ] */
            div.element-container:has(.nav-logo-wrapper) + div.element-container button {{
                background: url("data:image/png;base64,{logo_b64}") center/contain no-repeat !important;
                border: none !important;
                box-shadow: none !important;
                height: 44px !important;
                color: transparent !important;
                text-shadow: none !important;
                font-size: 0 !important;
                padding: 0 !important;
                cursor: pointer !important;
                transition: all 0.3s ease !important;
                background-color: transparent !important;
                width: 100% !important;
                transform: scale(2.0) !important;
                transform-origin: center center !important;
            }}
            div.element-container:has(.nav-logo-wrapper) + div.element-container button:hover {{
                transform: scale(2.15) !important;
                background-color: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }}
        </style>
        """), unsafe_allow_html=True)

    st.markdown(clean_html("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@300;400;500;700;900&display=swap');
        
        /* 전역 폰트 및 백그라운드 흐름 (에메랄드/파스텔 메쉬 그라데이션) */
        html, body, [data-testid="stApp"] {
            font-family: 'Plus Jakarta Sans', 'Noto Sans KR', sans-serif !important;
            background: radial-gradient(at 0% 0%, rgba(240,253,244,0.6) 0, transparent 50%),
                        radial-gradient(at 50% 0%, rgba(241,245,249,0.8) 0, transparent 50%),
                        radial-gradient(at 100% 0%, rgba(219,234,254,0.3) 0, transparent 50%),
                        #f8fafc !important;
        }
        
        /* [Hero Section Typography & Badges] */
        .gradient-text {
            font-size: 3.2rem !important;
            background: linear-gradient(135deg, #1B6554 0%, #10B981 100%) !important;
            -webkit-background-clip: text !important;
            background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            text-fill-color: transparent !important;
            color: #1B6554 !important; /* solid fallback green */
            font-weight: 950 !important;
            display: inline-block !important;
            margin-top: 10px !important;
        }
        .hero-badge {
            background-color: #E6FDF5 !important;
            color: #1B6554 !important;
            padding: 6px 14px !important;
            border-radius: 20px !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            display: inline-block !important;
            margin-bottom: 20px !important;
            border: 1.2px solid rgba(27, 101, 84, 0.15) !important;
            letter-spacing: -0.3px !important;
        }

        /* [Interactive Phone Mockup & Inner Elements] */
        .phone-container {
            width: 290px !important;
            height: 470px !important;
            border-radius: 38px !important;
            border: 10px solid #1E293B !important;
            background: #F8FAFC !important;
            box-shadow: 0 20px 40px rgba(0,0,0,0.12) !important;
            position: relative !important;
            overflow: hidden !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
            margin: 0 auto !important;
        }
        .phone-title-badge {
            font-size: 0.65rem !important;
            font-weight: 800 !important;
            color: #1B6554 !important;
            background: rgba(27, 101, 84, 0.08) !important;
            padding: 2px 6px !important;
            border-radius: 6px !important;
            display: inline-block !important;
        }
        .phone-status-text {
            font-size: 0.6rem !important;
            color: #64748B !important;
            font-weight: 750 !important;
            display: inline-block !important;
        }
        .phone-scan-message {
            position: absolute !important;
            bottom: 8px !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            font-size: 0.65rem !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            background: rgba(16, 185, 129, 0.75) !important;
            padding: 3px 10px !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            backdrop-filter: blur(4px) !important;
            z-index: 4 !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1) !important;
            white-space: nowrap !important;
        }
        .phone-patient-card {
            background: #ffffff !important;
            border-radius: 18px !important;
            padding: 10px 12px !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
            border: 1px solid rgba(0,0,0,0.02) !important;
            text-align: left !important;
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
        }
        .phone-patient-header {
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            margin-bottom: 4px !important;
            width: 100% !important;
        }
        .phone-patient-title {
            font-size: 0.68rem !important;
            font-weight: 800 !important;
            color: #1E293B !important;
            white-space: nowrap !important;
            display: inline-flex !important;
            align-items: center !important;
            gap: 2px !important;
        }
        .phone-patient-title-sub {
            font-size: 0.54rem !important;
            font-weight: 500 !important;
            color: #64748B !important;
            white-space: nowrap !important;
        }
        .phone-patient-status {
            font-size: 0.54rem !important;
            font-weight: 800 !important;
            color: #1B6554 !important;
            background: #E6FDF5 !important;
            padding: 1.5px 4px !important;
            border-radius: 5px !important;
            white-space: nowrap !important;
            display: inline-block !important;
            flex-shrink: 0 !important;
        }
        .phone-patient-list {
            display: flex !important;
            flex-direction: column !important;
            gap: 3px !important;
            width: 100% !important;
        }
        .phone-patient-list div,
        .phone-patient-list span,
        .phone-patient-list p {
            font-size: 0.63rem !important;
            color: #334155 !important;
            font-weight: 650 !important;
            line-height: 1.3 !important;
            white-space: nowrap !important;
        }
        .phone-topic-card {
            background: #F0FDFA !important;
            border: 1px dashed rgba(16, 185, 129, 0.3) !important;
            border-radius: 16px !important;
            padding: 8px 12px !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            text-align: left !important;
        }
        .phone-topic-emoji {
            font-size: 1.15rem !important;
            display: inline-block !important;
        }
        .phone-topic-title {
            font-size: 0.72rem !important;
            font-weight: 800 !important;
            color: #0B5C4B !important;
        }
        .phone-topic-desc {
            font-size: 0.65rem !important;
            color: #3A5F56 !important;
            font-weight: 700 !important;
            margin-top: 1.5px !important;
        }
        
        .block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 95% !important; padding-left: 5vw !important; padding-right: 5vw !important; }
        header { visibility: hidden !important; }
        
        /* [UX 대개선] 전역 텍스트 가독성 및 글꼴 크기 대폭 향상 (Streamlit 전용 요소로 한정하여 커스텀 HTML 파괴 방지) */
        .stMarkdown p, div[data-testid="stMarkdownContainer"] p, .stMarkdown li, div[data-testid="stMarkdownContainer"] li {
            font-size: 1.15rem !important;
            line-height: 1.75 !important;
        }
        
        /* 모든 입력 위젯, 텍스트에어리어, 셀렉트박스 내부 폰트 확대 */
        textarea, input, select, .stSelectbox div, [data-baseweb="select"] div {
            font-size: 1.15rem !important;
        }
        
        /* 1. 상단 플로팅 글래스 네비게이션 헤더 전체 스타일 */
        div.element-container:has(.nav-bar-marker) + div.element-container > div[data-testid="stHorizontalBlock"] {
            background: rgba(255, 255, 255, 0.75) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.6) !important;
            padding: 8px 24px !important;
            box-shadow: 0 10px 30px -10px rgba(27,101,84,0.06) !important;
            margin-bottom: 2rem !important;
            align-items: center !important;
        }
        
        /* 상단 네비게이션 6개 텍스트 버튼 스타일 (Glass capsules) */
        div.element-container:has(.nav-btn-wrapper) + div.element-container button {
            border-radius: 30px !important;
            width: 100% !important;
            height: 44px !important;
            padding: 0 16px !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 1.1rem !important;
            font-weight: 750 !important;
            background: rgba(255, 255, 255, 0.5) !important;
            border: 1.5px solid rgba(27,101,84,0.15) !important;
            color: #475569 !important;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }
        
        div.element-container:has(.nav-btn-wrapper) + div.element-container button:hover {
            transform: translateY(-2px) !important;
            border-color: #1B6554 !important;
            color: #1B6554 !important;
            background: rgba(27,101,84,0.08) !important;
            box-shadow: 0 8px 16px rgba(27,101,84,0.12) !important;
        }
        
        /* 활성화 상태 (Primary) 네비 텍스트 버튼 */
        div.element-container:has(.nav-btn-wrapper.active) + div.element-container button {
            background: linear-gradient(135deg, #1B6554, #144E41) !important;
            border-color: #1B6554 !important;
            color: white !important;
            box-shadow: 0 8px 20px -5px rgba(27,101,84,0.25) !important;
        }
        
        /* 로그아웃 버튼 스타일 */
        div.element-container:has(.nav-logout-wrapper) + div.element-container button,
        div[data-testid="element-container"]:has(.nav-logout-wrapper) button {
            background: transparent !important; 
            border: 1.5px solid rgba(226, 232, 240, 0.8) !important; 
            color: #64748B !important;
            font-size: 1.0rem !important; 
            font-weight: 700 !important;
            padding: 8px 16px !important; 
            box-shadow: none !important;
            border-radius: 30px !important; 
            width: 100% !important; 
            height: 44px !important;
            transition: all 0.3s ease;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        div.element-container:has(.nav-logout-wrapper) + div.element-container button:hover,
        div[data-testid="element-container"]:has(.nav-logout-wrapper) button:hover {
            background-color: #F8FAFC !important; 
            color: #ef4444 !important; 
            border-color: #fca5a5 !important;
        }
        
        /* 2. 본문 UI 요소 및 카드 스타일 고도화 */
        .card { 
            background: rgba(255, 255, 255, 0.7); 
            backdrop-filter: blur(12px);
            border-radius: 20px; 
            padding: 30px; 
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.04); 
            border: 1px solid rgba(255, 255, 255, 0.6); 
            height: 100%; 
            transition: all 0.3s ease;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 40px -15px rgba(27,101,84,0.06);
        }
        
        /* 잃어버렸던 영롱한 메인 배너(Hero) 그라데이션 배경 복구 및 글래스모피즘 */
        div.block-container > div[data-testid="stVerticalBlock"] > div.element-container:nth-child(3) > div[data-testid="stHorizontalBlock"],
        div.block-container > div[data-testid="stVerticalBlock"] > div.element-container:nth-child(4) > div[data-testid="stHorizontalBlock"] {
            background: linear-gradient(135deg, rgba(240, 247, 244, 0.8) 0%, rgba(255, 255, 255, 0.8) 100%) !important;
            backdrop-filter: blur(16px) !important;
            border-radius: 28px !important; 
            padding: 50px 40px !important; 
            margin-bottom: 2rem !important; 
            border: 1px solid rgba(255, 255, 255, 0.7) !important;
            box-shadow: 0 15px 35px -15px rgba(27,101,84,0.08) !important;
            align-items: center !important;
        }
        
        /* 본문 설문조사 폼 내부 라디오 버튼 (예/아니오) 간격 깔끔하게 정렬 */
        div[data-testid="stForm"] div[role="radiogroup"] { display: flex; gap: 30px; align-items: center; justify-content: flex-start; }
        
        /* 3. 대형 액션 버튼 스타일링 */
        button[kind="primary"], button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #1B6554 0%, #103F34 100%) !important; 
            color: white !important; 
            border-radius: 14px !important;
            border: none !important; 
            height: 64px !important; 
            font-size: 1.3rem !important; 
            font-weight: 800 !important;
            box-shadow: 0 8px 25px -5px rgba(27,101,84,0.3) !important; 
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important; 
            width: 100% !important;
        }
        button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
            transform: translateY(-3px) !important; 
            box-shadow: 0 12px 30px -5px rgba(27,101,84,0.45) !important;
        }
        
        /* 4. 카카오톡/iMessage 스타일의 곡선형 스마트 챗봇 버블 */
        [data-testid="chatAvatarIcon-user"], [data-testid="chatAvatarIcon-assistant"] {
            border-radius: 50% !important;
        }
        div[data-testid="chatMessage"] {
            padding: 1.2rem 1.5rem !important;
            margin-bottom: 1.5rem !important;
            border-radius: 20px !important;
            border: none !important;
            max-width: 85% !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.02) !important;
            transition: all 0.2s ease;
        }
        /* 유저 말풍선 (우측 정렬, 선명한 에메랄드 그라데이션) */
        div[data-testid="chatMessage"]:has([data-testid="chatAvatarIcon-user"]), 
        div[data-testid="chatMessage"]:has(img[src*="user"]) {
            background: linear-gradient(135deg, #1B6554, #144E41) !important;
            color: white !important;
            margin-left: auto !important;
            border-bottom-right-radius: 4px !important;
        }
        div[data-testid="chatMessage"]:has([data-testid="chatAvatarIcon-user"]) p,
        div[data-testid="chatMessage"]:has(img[src*="user"]) p {
            color: white !important;
            font-weight: 500 !important;
            font-size: 1.15rem !important;
        }
        /* 어시스턴트 말풍선 (좌측 정렬, 오프화이트 글래스모피즘) */
        div[data-testid="chatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
        div[data-testid="chatMessage"]:has(img[src*="logo"]) {
            background: rgba(255, 255, 255, 0.85) !important;
            border: 1px solid rgba(27,101,84,0.12) !important;
            margin-right: auto !important;
            border-bottom-left-radius: 4px !important;
        }
        div[data-testid="chatMessage"]:hover {
            box-shadow: 0 8px 25px rgba(27,101,84,0.05) !important;
            transform: translateY(-1px);
        }
        
        /* 5. 폼 위젯 리플래시 */
        .stSlider > div > div > div { background-color: #1B6554 !important; }
        .stSelectbox > div > div { border-radius: 10px !important; border: 1.5px solid rgba(27,101,84,0.15) !important; }
        
        /* 6. 설문지 폼 프리미엄 글래스 카드화 */
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.45) !important;
            backdrop-filter: blur(24px) !important;
            -webkit-backdrop-filter: blur(24px) !important;
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.6) !important;
            padding: 35px !important;
            box-shadow: 0 20px 40px -15px rgba(27,101,84,0.06) !important;
            margin-bottom: 2rem !important;
        }
        div[data-testid="stForm"] [data-testid="stSelectbox"] > div {
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.7) !important;
            border: 1.5px solid rgba(27,101,84,0.12) !important;
        }
        div[data-testid="stForm"] [data-testid="stNumberInput"] > div {
            border-radius: 12px !important;
            background: rgba(255, 255, 255, 0.7) !important;
            border: 1.5px solid rgba(27,101,84,0.12) !important;
        }
        div[data-testid="stForm"] [data-testid="stSlider"] > div {
            padding-top: 10px !important;
            padding-bottom: 10px !important;
        }
        div[data-testid="stForm"] label {
            font-size: 1.15rem !important;
            color: #334155 !important;
            font-weight: 750 !important;
            margin-bottom: 8px !important;
        }
        
        /* Plotly Chart Card Wrapper */
        div[data-testid="stPlotlyChart"] {
            background-color: #ffffff !important;
            border-radius: 16px !important;
            padding: 24px 24px 12px 24px !important;
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.04) !important;
            border: 1px solid rgba(226, 232, 240, 0.8) !important;
            margin-bottom: 1.5rem !important;
        }
    </style>
    """), unsafe_allow_html=True)

def render_main():
    scan_img_path = os.path.join(_BASE_DIR, "design", "skin.png")
    scan_img_b64 = ""
    if os.path.exists(scan_img_path):
        import base64
        with open(scan_img_path, "rb") as f:
            scan_img_b64 = base64.b64encode(f.read()).decode()
            

            
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown(clean_html("""<div style='margin-top: 30px; margin-bottom: 2.5rem; text-align: left;'>
    <span class="hero-badge">&#127808; [AI 아토피 조기 예측 솔루션], AtoCatch</span>
    <h1 style="font-size: 3.2rem; font-weight: 900; color: #1E293B; line-height: 1.35; margin: 0 0 20px 0; letter-spacing: -1px;">
        아토피일까 걱정된다면<br>AtoCatch AI 스캔으로<br>
        <span class="gradient-text">3초 만에 확인</span>
    </h1>
    <p style="font-size: 1.15rem; color: #475569; line-height: 1.7; font-weight: 500; margin: 0 0 35px 0; letter-spacing: -0.5px;">
        가족력 분석부터 AI 이미지 진단까지 !<br>
        AtoCatch가 아기 피부 변화를 실시간으로 감지하고 맞춤형 케어법을 제안합니다.
    </p>
</div>"""), unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns([1, 1])
        with btn_col1:
            if st.button("📊 우리 아이 아토피 위험 미리보기", use_container_width=True, type="primary"):
                st.session_state.current_page = "설문조사"
                st.rerun()
        with btn_col2:
            if st.button("📷 우리 아이 아토피 상태 바로보기", use_container_width=True, type="primary"):
                st.session_state.current_page = "피부 스캔"
                st.rerun()
    with col2:
        st.markdown(clean_html(f"""<div style="background: rgba(255, 255, 255, 0.4); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border-radius: 30px; border: 1px solid rgba(255, 255, 255, 0.5); padding: 25px; box-shadow: 0 20px 40px -15px rgba(27, 101, 84, 0.05); text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
    <!-- Interactive Phone Mockup -->
    <div class="phone-container" style="width: 290px; height: 470px; border-radius: 38px; border: 10px solid #1E293B; background: #F8FAFC; box-shadow: 0 20px 40px rgba(0,0,0,0.12); position: relative; overflow: hidden; display: flex; flex-direction: column; justify-content: space-between; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); margin: 0 auto;">
        <!-- Phone Speaker & Camera Notch -->
        <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 110px; height: 18px; background: #1E293B; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; z-index: 10; display: flex; align-items: center; justify-content: center; gap: 6px;">
            <div style="width: 35px; height: 3px; background: #475569; border-radius: 3px;"></div>
            <div style="width: 5px; height: 5px; background: #475569; border-radius: 50%;"></div>
        </div>
        
        <!-- Mock Screen Body -->
        <div style="flex-grow: 1; display: flex; flex-direction: column; justify-content: space-between; padding: 22px 14px 14px 14px; position: relative;">
            <!-- Screen Header -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.65rem !important; font-weight: 800; color: #1B6554; background: rgba(27, 101, 84, 0.08); padding: 2px 6px; border-radius: 6px;">AtoCatch AI</span>
                <div style="display: flex; gap: 4px; align-items: center;">
                    <div style="width: 6px; height: 6px; border-radius: 50%; background: #10B981; animation: blink 1s infinite;"></div>
                    <span style="font-size: 0.6rem !important; color: #64748B; font-weight: 750;">스캔 가동 중</span>
                </div>
            </div>

            <!-- CSS Camera Scan Graphic Viewport -->
            <div style="position: relative; width: 100%; height: 180px; background: #000; border-radius: 20px; display: flex; align-items: center; justify-content: center; overflow: hidden; border: 1.5px solid rgba(16, 185, 129, 0.2);">
                <!-- Baby Atopy Analysis Photo -->
                <img src="data:image/png;base64,{scan_img_b64}" style="width: 100%; height: 100%; object-fit: cover; position: absolute; top: 0; left: 0; opacity: 0.85; z-index: 1;" />
                
                <!-- Moving green laser scanline -->
                <div class="scanline" style="position: absolute; width: 100%; height: 3px; background: linear-gradient(90deg, transparent, #10B981, transparent); top: 0; left: 0; box-shadow: 0 0 8px #10B981; animation: scan-move 2.2s infinite linear; z-index: 3;"></div>
                
                <!-- Scanning Target Box overlayed on the image -->
                <div class="scan-target" style="position: absolute; width: 110px; height: 110px; border: 2.5px solid #10B981; border-radius: 16px; z-index: 2; animation: pulse-border 1.5s infinite; background: rgba(16, 185, 129, 0.05);">
                </div>
                
                <!-- Safe scanning message -->
                <div style="position: absolute; bottom: 8px; font-size: 0.65rem !important; font-weight: 800; color: #ffffff; background: rgba(16, 185, 129, 0.75); padding: 3px 10px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.2); backdrop-filter: blur(4px); z-index: 4; box-shadow: 0 4px 10px rgba(0,0,0,0.1);">
                    실시간 분석 감지 완료
                </div>
            </div>

            <!-- Analysis Indicator Dashboard -> Patient Information Card -->
            <div class="phone-patient-card">
                <div class="phone-patient-header">
                    <span class="phone-patient-title">👤 환자 정보 <span class="phone-patient-title-sub">(진단 대상)</span></span>
                    <span class="phone-patient-status">스캔 완료</span>
                </div>
                <div class="phone-patient-list">
                    <div>• 이름: <span style="font-weight: 800; color: #1E293B;">김아토 (Baby)</span></div>
                    <div>• 월령: <span style="font-weight: 800; color: #1E293B;">생후 10개월</span></div>
                    <div>• 성별: <span style="font-weight: 800; color: #1E293B;">남자아이</span></div>
                    <div>• 일시: <span style="font-weight: 800; color: #1E293B;">2026-05-20</span></div>
                </div>
            </div>

            <!-- Recommendation Card -> Atopy Relief Topic Card -->
            <div style="background: #F0FDFA; border: 1px dashed rgba(16, 185, 129, 0.3); border-radius: 16px; padding: 8px 12px; display: flex; align-items: center; gap: 8px; text-align: left;">
                <span style="font-size: 1.15rem !important;">🌿</span>
                <div>
                    <div style="font-size: 0.72rem !important; font-weight: 800; color: #0B5C4B;">💡 추천 아토피 완화 주제</div>
                    <div style="font-size: 0.65rem !important; color: #3A5F56; font-weight: 700; margin-top: 1.5px;">미온수 목욕 후 3분 이내 고보습제 집중 도포</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Visual Flow Breadcrumbs under Phone -->
    <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 22px;">
        <div style="background: rgba(255, 152, 0, 0.08); color: #E65100; font-size: 0.65rem !important; font-weight: 800; padding: 4px 10px; border-radius: 12px; border: 1px solid rgba(255, 152, 0, 0.15); display: flex; align-items: center; gap: 4px;">
            <span>&#128221;</span> 설문환경
        </div>
        <span style="color: #CBD5E1; font-weight: 900; font-size: 0.7rem;">&#10132;</span>
        <div style="background: rgba(16, 185, 129, 0.08); color: #065F46; font-size: 0.65rem !important; font-weight: 800; padding: 4px 10px; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.15); display: flex; align-items: center; gap: 4px;">
            <span>&#128247;</span> AI스캔
        </div>
        <span style="color: #CBD5E1; font-weight: 900; font-size: 0.7rem;">&#10132;</span>
        <div style="background: rgba(59, 130, 246, 0.08); color: #1E3A8A; font-size: 0.65rem !important; font-weight: 800; padding: 4px 10px; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.15); display: flex; align-items: center; gap: 4px;">
            <span>&#128157;</span> 안심케어
        </div>
    </div>
</div>
"""), unsafe_allow_html=True)

        st.markdown(clean_html("""
<style>
    .phone-container:hover {
        transform: translateY(-6px) scale(1.03);
        box-shadow: 0 25px 50px rgba(27, 101, 84, 0.15) !important;
        border-color: #1B6554 !important;
    }
    @keyframes blink {
        0% { opacity: 0.3; }
        50% { opacity: 1; }
        100% { opacity: 0.3; }
    }
    @keyframes scan-move {
        0% { top: 0%; opacity: 0.3; }
        50% { top: 100%; opacity: 1; }
        100% { top: 0%; opacity: 0.3; }
    }
    @keyframes pulse-border {
        0% { border-color: rgba(16, 185, 129, 0.4); transform: scale(1); }
        50% { border-color: rgba(16, 185, 129, 1); transform: scale(1.01); }
        100% { border-color: rgba(16, 185, 129, 0.4); transform: scale(1); }
    }
</style>"""), unsafe_allow_html=True)

def render_survey():
    st.markdown("<h2>📝 우리 아이 아토피 위험 미리보기</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#666;'>정확한 예측을 위해 아이와 가족의 정보를 입력해 주세요.</p>", unsafe_allow_html=True)
    
    with st.form("survey_form"):
        col1, col2 = st.columns(2)
        with col1:
            antibiotic = st.selectbox("생후 첫돌 이전 항생제 복용 횟수", list(AB_MAP.keys()))
            parent_AD = st.radio("부모 중 아토피 진단을 받은 분이 있습니까?", ["예", "아니오"], horizontal=True)
            parent_AR = st.radio("부모 중 알레르기 비염 진단을 받은 분이 있습니까?", ["예", "아니오"], horizontal=True)
            parent_asthma = st.radio("부모 중 천식 진단을 받은 분이 있습니까?", ["예", "아니오"], horizontal=True)
            sibling_allergy = st.radio("형제자매 중 알레르기 질환이 있나요?", ["예", "아니오"], horizontal=True)
        with col2:
            child_passive_smoke = st.radio("아이 현재 간접흡연 노출 여부", ["예", "아니오"], horizontal=True)
            mold_ever = st.radio("임신 중 또는 아이 첫돌 이전, 실내 곰팡이 노출 경험", ["예", "아니오"], horizontal=True)
            passive_smoke_ever = st.radio("아이가 태어난 이후 지금까지 가족 중 흡연자가 있나요?", ["예", "아니오"], horizontal=True)
            pet_ever = st.radio("현재 또는 과거 반려동물 양육 여부", ["예", "아니오"], horizontal=True)
            rural_years = st.slider("농촌 거주 기간 (만 0~5세)", 0, 6, 0)
            outdoor_avg = st.number_input("하루 평균 실외활동 시간", 0.0, 12.0, 1.0)
            
        submitted = st.form_submit_button("다음 단계로 이동 (설문 분석 저장) &#10132;", use_container_width=True)
        if submitted:
            yn = {"예": 0, "아니오": 1}
            input_df = pd.DataFrame([{
                "antibiotic": AB_MAP[antibiotic], "parent_AD": yn[parent_AD],
                "parent_AR": yn[parent_AR], "mold_ever": yn[mold_ever],
                "parent_asthma": yn[parent_asthma], "sibling_allergy": yn[sibling_allergy],
                "pet_ever": yn[pet_ever], "passive_smoke_ever": yn[passive_smoke_ever],
                "child_passive_smoke": yn[child_passive_smoke],
                "rural_years": rural_years, "outdoor_avg": outdoor_avg,
            }])
            
            # 위험도 계산
            prob = float(risk_model.predict_proba(input_df)[0, 1])
            if prob < 0.13: level = "저위험"
            elif prob < 0.20: level = "중위험"
            else: level = "고위험"
            
            st.session_state.survey_result = {
                "prob": prob, "level": level,
                "inputs": input_df.iloc[0].to_dict()
            }
            add_history(st.session_state.username, "설문조사", f"위험도: {level} ({prob*100:.1f}%)")
            # 자동 페이지 전환
            st.session_state.current_page = "피부 스캔"
            st.rerun()

def render_image_scan():
    st.markdown("<h2>📷 우리 아이 아토피 상태 바로보기</h2>", unsafe_allow_html=True)
    if not st.session_state.survey_result:
        st.info("💡 설문조사를 먼저 진행하면 더 정확한 종합 결과를 얻을 수 있습니다.")

    # 📸 정확한 AI 분석을 위한 피부 촬영 가이드 박스 추가
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown(clean_html("""
        <div style="background: rgba(255, 255, 255, 0.55); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 2px solid #10B981; border-radius: 24px; padding: 28px; box-shadow: 0 12px 32px -8px rgba(16, 185, 129, 0.12); margin-top: 15px;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 20px;">
                <span style="background: #10B981; color: white; width: 28px; height: 28px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.9rem; font-weight: 900; box-shadow: 0 0 12px rgba(16, 185, 129, 0.4); animation: pulse-green 2s infinite;">&#128161;</span>
                <h4 style="color: #065F46; margin: 0; font-weight: 850; font-size: 1.25rem; letter-spacing: -0.3px;">정확한 AI 피부 분석을 위한 촬영 가이드</h4>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
                <div style="background: rgba(255, 255, 255, 0.6); border: 1.5px solid rgba(16, 185, 129, 0.15); border-radius: 18px; padding: 18px; position: relative; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.01);">
                    <div style="position: absolute; right: 12px; top: 12px; font-size: 1.8rem; opacity: 0.8;">&#9728;</div>
                    <div style="background: #E6FDF5; color: #10B981; padding: 4px 8px; border-radius: 8px; font-size: 0.7rem; font-weight: 800; display: inline-block; margin-bottom: 10px;">LIGHT</div>
                    <h5 style="margin: 0 0 6px 0; color: #111827; font-size: 0.95rem; font-weight: 800;">밝은 조명 필수</h5>
                    <p style="margin: 0; color: #4B5563; font-size: 0.8rem; line-height: 1.5; font-weight: 500;">어두운 방이나 그늘은 오진율을 높입니다. 자연광이나 밝은 실내 조명 아래서 촬영해주세요.</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.6); border: 1.5px solid rgba(16, 185, 129, 0.15); border-radius: 18px; padding: 18px; position: relative; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.01);">
                    <div style="position: absolute; right: 12px; top: 12px; font-size: 1.8rem; opacity: 0.8;">&#128269;</div>
                    <div style="background: #E6FDF5; color: #10B981; padding: 4px 8px; border-radius: 8px; font-size: 0.7rem; font-weight: 800; display: inline-block; margin-bottom: 10px;">FOCUS</div>
                    <h5 style="margin: 0 0 6px 0; color: #111827; font-size: 0.95rem; font-weight: 800;">흔들림 없이 선명하게</h5>
                    <p style="margin: 0; color: #4B5563; font-size: 0.8rem; line-height: 1.5; font-weight: 500;">초점이 흐려지면 각질이나 붉은 병변 감지가 어렵습니다. 흔들림 없이 선명하게 고정해 주세요.</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.6); border: 1.5px solid rgba(16, 185, 129, 0.15); border-radius: 18px; padding: 18px; position: relative; overflow: hidden; box-shadow: 0 4px 15px rgba(0,0,0,0.01);">
                    <div style="position: absolute; right: 12px; top: 12px; font-size: 1.8rem; opacity: 0.8;">&#128207;</div>
                    <div style="background: #E6FDF5; color: #10B981; padding: 4px 8px; border-radius: 8px; font-size: 0.7rem; font-weight: 800; display: inline-block; margin-bottom: 10px;">DISTANCE</div>
                    <h5 style="margin: 0 0 6px 0; color: #111827; font-size: 0.95rem; font-weight: 800;">적정 거리 유지 (15~20cm)</h5>
                    <p style="margin: 0; color: #4B5563; font-size: 0.8rem; line-height: 1.5; font-weight: 500;">지나치게 가까운 접사보다는 병변 주변의 일반 피부가 함께 나오도록 살짝 떼어 촬영해 주세요.</p>
                </div>
            </div>
        </div>
        <style>
            @keyframes pulse-green {
                0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
                100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
            }
        </style>
    """), unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📁 파일 업로드", "📸 직접 촬영하기"])
    
    uploaded_file = None
    with tab1:
        file_input = st.file_uploader("증상 부위 사진 업로드 (JPG, PNG)", type=["png", "jpg", "jpeg"])
        if file_input:
            uploaded_file = file_input
            
    with tab2:
        camera_input = st.camera_input("스마트폰/웹 카메라로 증상 부위 촬영")
        if camera_input:
            uploaded_file = camera_input
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, use_container_width=True, caption="원본 이미지")
            
        with col2:
            if st.button("🔬 AI 피부 분석 시작", use_container_width=True, type="primary"):
                with st.spinner("AI가 이미지를 분석하고 있습니다..."):
                    if GRADCAM_OK:
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        gc_result = predict_with_gradcam(img, image_model, iga_model, device=device)
                        atopy_prob = gc_result["atopy_prob"]
                        is_atopy   = gc_result["is_atopy"]
                        iga_prob   = gc_result["severity_prob"]
                        iga_severe = gc_result["severity"]
                        
                        import io as _gcio
                        def _pil_to_b64(pil_img):
                            buf = _gcio.BytesIO()
                            pil_img.save(buf, format="JPEG", quality=85)
                            return base64.b64encode(buf.getvalue()).decode()
                        st.session_state.gradcam_a = _pil_to_b64(gc_result["gradcam_a_overlay"])
                    else:
                        tensor = IMG_TRANSFORM(img).unsqueeze(0)
                        with torch.no_grad():
                            logits = image_model(tensor)
                            probs  = torch.softmax(logits, dim=1)[0]
                        atopy_prob = float(probs[1])
                        is_atopy   = atopy_prob >= 0.29
                        iga_prob, iga_severe = None, None
                        if is_atopy:
                            with torch.no_grad():
                                iga_logits = iga_model(tensor)
                                iga_probs  = torch.softmax(iga_logits, dim=1)[0]
                            iga_prob   = float(iga_probs[1])
                            iga_severe = iga_prob >= 0.38
                        st.session_state.gradcam_a = None

                    st.session_state.img_result = {
                        "prob": atopy_prob, "is_atopy": is_atopy,
                        "iga_prob": iga_prob, "iga_severe": iga_severe
                    }
                    
                    # 원본 이미지 축소 및 Base64 인코딩
                    import io as _gcio
                    img_b64 = None
                    try:
                        img_small = img.copy()
                        img_small.thumbnail((400, 400))
                        buf = _gcio.BytesIO()
                        img_small.save(buf, format="JPEG", quality=80)
                        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        st.session_state.current_img_b64 = img_b64
                    except Exception as e:
                        pass

                    extra_data = {"atopy_prob": atopy_prob}
                    if img_b64:
                        extra_data["image_b64"] = img_b64
                    if st.session_state.get("gradcam_a"):
                        extra_data["gradcam_b64"] = st.session_state.gradcam_a

                    # 기록 저장
                    atopy_txt = f"아토피 의심 {atopy_prob*100:.1f}%" if is_atopy else f"아토피 낮음 {atopy_prob*100:.1f}%"
                    if iga_prob is not None:
                        iga_txt = "중등도·중증" if iga_severe else "경증 이하"
                        atopy_txt += f" / IGA {iga_txt} ({iga_prob*100:.1f}%)"
                        
                    add_history(st.session_state.username, "이미지분석", atopy_txt, extra=extra_data)
                    st.success("✅ AI 피부 분석이 완료되었습니다!")
                    
            if st.session_state.img_result:
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                st.info("💡 아래 버튼을 눌러 상세한 분석 결과를 확인하세요.")
                if st.button("📊 분석 결과 보러가기 &#10132;", type="primary", use_container_width=True):
                    st.session_state.current_page = "분석 결과"
                    st.rerun()

def render_result():
    st.markdown("<h2>📊 종합 분석 결과</h2>", unsafe_allow_html=True)
    
    if not st.session_state.survey_result and not st.session_state.img_result:
        st.warning("먼저 설문조사나 피부 스캔을 진행해 주세요.")
        return
        
    # Calculate metrics
    survey_val = "미진행"
    survey_lbl = "가족력/환경 설문 위험도"
    survey_color = "#64748B"
    survey_level = ""
    if st.session_state.survey_result:
        sr = st.session_state.survey_result
        survey_val = f"{sr['prob']*100:.1f}%"
        survey_level = sr['level']
        survey_color = "#EF4444" if sr['level'] == "고위험" else ("#F59E0B" if sr['level'] == "중위험" else "#10B981")

    img_val = "미진행"
    img_lbl = "피부 AI 아토피 여부"
    img_color = "#64748B"
    img_level = ""
    if st.session_state.img_result:
        ir = st.session_state.img_result
        img_val = f"{ir['prob']*100:.1f}%"
        img_level = "의심" if ir['is_atopy'] else "낮음"
        img_color = "#EF4444" if ir['is_atopy'] else "#10B981"

    iga_val = "미진행"
    iga_lbl = "피부 AI 중증도 (IGA)"
    iga_color = "#64748B"
    iga_level = ""
    if st.session_state.img_result:
        ir = st.session_state.img_result
        if ir['iga_prob'] is not None:
            iga_val = f"{ir['iga_prob']*100:.1f}%"
            iga_level = "중등도·중증" if ir.get('iga_severe') else "경증 이하"
            iga_color = "#EF4444" if ir.get('iga_severe') else "#10B981"

    # HTML content construction for each card value to emphasize the qualitative risk level over numeric percentage
    if st.session_state.survey_result:
        card1_content = f"""
        <span style="font-size: 2.1rem; font-weight: 900; color: {survey_color}; line-height: 1;">{survey_level}</span>
        <span style="font-size: 1.15rem; font-weight: 700; color: #64748B; margin-left: 6px;">({survey_val})</span>
        """
    else:
        card1_content = f"""
        <span style="font-size: 2.1rem; font-weight: 900; color: #64748B; line-height: 1;">미진행</span>
        """

    if st.session_state.img_result:
        card2_content = f"""
        <span style="font-size: 2.1rem; font-weight: 900; color: {img_color}; line-height: 1;">{img_level}</span>
        <span style="font-size: 1.15rem; font-weight: 700; color: #64748B; margin-left: 6px;">({img_val})</span>
        """
    else:
        card2_content = f"""
        <span style="font-size: 2.1rem; font-weight: 900; color: #64748B; line-height: 1;">미진행</span>
        """

    if st.session_state.img_result and st.session_state.img_result['iga_prob'] is not None:
        card3_content = f"""
        <span style="font-size: 2.1rem; font-weight: 900; color: {iga_color}; line-height: 1;">{iga_level}</span>
        <span style="font-size: 1.15rem; font-weight: 700; color: #64748B; margin-left: 6px;">({iga_val})</span>
        """
    else:
        card3_content = f"""
        <span style="font-size: 2.1rem; font-weight: 900; color: #64748B; line-height: 1;">미진행</span>
        """

    st.markdown(clean_html(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 25px;">
        <!-- Card 1 -->
        <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.6); padding: 24px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: space-between;">
            <span style="font-size: 0.85rem; font-weight: 750; color: #64748B; letter-spacing: -0.2px;">📝 {survey_lbl}</span>
            <div style="margin-top: 14px; display: flex; align-items: baseline;">
                {card1_content}
            </div>
        </div>
        <!-- Card 2 -->
        <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.6); padding: 24px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: space-between;">
            <span style="font-size: 0.85rem; font-weight: 750; color: #64748B; letter-spacing: -0.2px;">📷 {img_lbl}</span>
            <div style="margin-top: 14px; display: flex; align-items: baseline;">
                {card2_content}
            </div>
        </div>
        <!-- Card 3 -->
        <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.6); padding: 24px; box-shadow: 0 10px 30px -10px rgba(0,0,0,0.02); display: flex; flex-direction: column; justify-content: space-between;">
            <span style="font-size: 0.85rem; font-weight: 750; color: #64748B; letter-spacing: -0.2px;">🔬 {iga_lbl}</span>
            <div style="margin-top: 14px; display: flex; align-items: baseline;">
                {card3_content}
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)

        
    st.markdown("---")
    
    col_desc, col_img = st.columns([1.5, 1])
    with col_desc:
        st.subheader("맞춤 생활 권고사항")
        if st.session_state.survey_result:
            inp = st.session_state.survey_result["inputs"]
            for key in ["antibiotic", "parent_AD", "child_passive_smoke", "parent_AR", "mold_ever", "pet_ever"]:
                present = (inp[key] >= 2 if key == "antibiotic" else inp[key] == 1)
                text = ADVICE[key][0] if present else ADVICE[key][1]
                if present:
                    st.error(f"**{FACTORS[key]['label']}**: {text}")
                else:
                    st.success(f"**{FACTORS[key]['label']}**: {text}")
        else:
            st.write("설문 결과가 없어 맞춤 권고를 생성할 수 없습니다.")
        col_actions = st.columns(2)
        with col_actions[0]:
            st.button("AI 챗봇에게 물어보기 &#10132;", on_click=lambda: st.session_state.update(current_page="AI 챗봇 서비스"), use_container_width=True)
        with col_actions[1]:
            if st.session_state.get("img_result"):
                ir = st.session_state.img_result
                atopy_label = "의심" if ir['is_atopy'] else "낮음"
                iga_label = " / IGA 중등도·중증" if ir.get('iga_severe') else (" / IGA 경증 이하" if ir['iga_prob'] is not None else "")
                detail_str = f"아토피 {atopy_label} ({ir['prob']*100:.1f}%){iga_label}"
                
                html_rep = generate_html_report(
                    display_name=st.session_state.display_name,
                    time_str=datetime.now().strftime("%Y-%m-%d %H:%M"),
                    detail=detail_str,
                    image_b64=st.session_state.get("current_img_b64"),
                    gradcam_b64=st.session_state.get("gradcam_a")
                )
                st.download_button(
                    label="📄 PDF용 HTML 리포트 다운로드",
                    data=html_rep,
                    file_name=f"{st.session_state.username}_atocatch_report.html",
                    mime="text/html",
                    use_container_width=True
                )
                
    with col_img:
        st.subheader("XAI 히트맵 (Grad-CAM)")
        if st.session_state.get("gradcam_a"):
            gcam_a = st.session_state.gradcam_a
            st.markdown(f'<img src="data:image/jpeg;base64,{gcam_a}" style="width:100%; border-radius:10px;"/>', unsafe_allow_html=True)
            st.caption("AI가 아토피 판단에 주목한 주요 병변 부위 (붉은색일수록 영향 큼)")
        else:
            st.info("이미지 분석 결과가 없거나 Grad-CAM 모듈을 로드하지 못했습니다.")

def render_guide():
    logo_path = os.path.join(_BASE_DIR, "design", "chatbot_logo.png")
    logo_b64 = ""
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
        except:
            pass

    if logo_b64:
        st.markdown(clean_html(f"""
            <div style='display: flex; align-items: center; gap: 20px; margin-bottom: 2rem; background-color: #F0F7F4; padding: 24px; border-radius: 18px; border: 1px solid #E2EFEA;'>
                <img src='data:image/png;base64,{logo_b64}' style='width: 85px; height: 85px; object-fit: contain; filter: drop-shadow(0 4px 8px rgba(27,101,84,0.15));'/>
                <div>
                    <h2 style='margin: 0; color: #1B6554; font-weight: 850; font-size: 2.1rem;'>💬 AtoCatch AI 챗봇 서비스</h2>
                    <p style='margin: 8px 0 0 0; color: #3A5F56; font-size: 1.15rem; font-weight: 550; line-height: 1.6;'>대한아토피피부염학회 진료지침에 근거하여 24시간 실시간 맞춤형 피부 관리 상담을 제공합니다.</p>
                </div>
            </div>
        """), unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color: #1B6554; font-weight: 850; font-size: 2.1rem; margin-bottom: 10px;'>💬 AtoCatch AI 챗봇 서비스</h2>", unsafe_allow_html=True)
        st.markdown("<p style='color:#3A5F56; font-size: 1.15rem; font-weight: 550; line-height: 1.6; margin-bottom: 2rem;'>분석 결과를 바탕으로 AI가 대한아토피피부염학회 진료지침에 근거하여 맞춤형 상담을 제공합니다.</p>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='font-size: 1.6rem; font-weight: 850; color: #1E293B; margin-top: 2rem; margin-bottom: 1.2rem;'>💬 대화 내역</h3>", unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown(clean_html("""
            <div style="background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 20px; border: 1px solid rgba(27,101,84,0.1); padding: 40px 30px; text-align: center; color: #64748B;">
                <div style="font-size: 3.2rem; margin-bottom: 16px;">💬</div>
                <div style="font-weight: 800; font-size: 1.35rem; color: #334155; margin-bottom: 8px;">아직 대화 내역이 없습니다.</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #64748B;">아래에 질문을 입력해 보세요!</div>
            </div>
        """), unsafe_allow_html=True)
    else:
        chat_html = "<div style='display: flex; flex-direction: column; gap: 18px; width: 100%;'>"
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += clean_html(f"""
                <div style='display: flex; justify-content: flex-end; width: 100%; margin-bottom: 8px;'>
                    <div style='background: linear-gradient(135deg, #1B6554, #144E41); color: white; padding: 16px 24px; border-radius: 20px 20px 4px 20px; max-width: 75%; box-shadow: 0 4px 15px rgba(27, 101, 84, 0.15); font-size: 1.15rem; line-height: 1.65; font-family: inherit; font-weight: 500;'>
                        {msg["content"]}
                    </div>
                </div>
                """) + "\n"
            else:
                assistant_avatar = ""
                if logo_b64:
                    assistant_avatar = f"<img src='data:image/png;base64,{logo_b64}' style='width: 44px; height: 44px; object-fit: contain; filter: drop-shadow(0 2px 4px rgba(27,101,84,0.12));'/>"
                else:
                    assistant_avatar = "<div style='width: 44px; height: 44px; border-radius: 50%; background: #1B6554; color: white; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; font-weight: 900;'>☻</div>"
                
                chat_html += clean_html(f"""
                <div style='display: flex; gap: 14px; width: 100%; align-items: flex-start; margin-bottom: 8px;'>
                    {assistant_avatar}
                    <div style='display: flex; flex-direction: column; gap: 6px; max-width: 75%;'>
                        <span style='font-size: 0.95rem; font-weight: 800; color: #1B6554; margin-left: 2px;'>AtoCatch AI</span>
                        <div style='background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1.5px solid rgba(27, 101, 84, 0.12); color: #1E293B; padding: 16px 24px; border-radius: 4px 20px 20px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.02); font-size: 1.15rem; line-height: 1.65; font-family: inherit; font-weight: 500;'>
                            {msg["content"]}
                        </div>
                    </div>
                </div>
                """) + "\n"
        chat_html += "</div>"
        st.markdown(clean_html(chat_html), unsafe_allow_html=True)
        
    st.markdown("<hr style='margin-top: 2rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_area("질문을 입력하세요. (예: 보습제는 하루에 몇 번 바르나요?)", height=120)
        submit_btn = st.form_submit_button("질문 전송 &#10132;", type="primary", use_container_width=True)
        
    if submit_btn and prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        if not OPENAI_API_KEY:
            bot_reply = "API 키가 설정되지 않아 챗봇을 사용할 수 없습니다."
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        else:
            with st.spinner("답변을 고민 중입니다..."):
                context_info = "아직 설문이나 이미지 분석 데이터가 없습니다."
                if st.session_state.survey_result:
                    sr = st.session_state.survey_result
                    context_info = f"아토피 설문 위험도: {sr['level']} ({sr['prob']*100:.1f}%)"
                    
                KNOWLEDGE_BASE = "보습은 목욕 후 3분 이내, 하루 2회 이상 실시합니다. 항생제 오남용은 장내미생물 균형을 깨트려 아토피를 악화시킬 수 있습니다."
                
                system_prompt = f"""당신은 AtoCatch의 영유아 아토피 예방 전문 AI 상담사입니다.
                현재 환자 상태: {context_info}
                기본 지식: {KNOWLEDGE_BASE}
                - 의학적 진단을 내리지 마세요. 예방 및 생활습관 관점에서 답변하세요.
                - 3~5문장으로 짧게 작성하세요."""
                
                try:
                    resp = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={"model": "gpt-4o-mini", "max_tokens": 300,
                              "messages": [{"role": "system", "content": system_prompt}] + st.session_state.chat_history},
                        timeout=15
                    )
                    resp_json = resp.json()
                    if "choices" in resp_json:
                        bot_reply = resp_json["choices"][0]["message"]["content"]
                    else:
                        bot_reply = "오류가 발생했습니다."
                except Exception as e:
                    bot_reply = f"API 호출 오류: {e}"
                    
                st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
        st.rerun()

def render_history():
    my_hist = load_history().get(st.session_state.username, [])
    
    if not my_hist:
        st.markdown("<h2>📋 우리 아이 아토피 상태 기록보기</h2>", unsafe_allow_html=True)
        st.info("아직 저장된 분석 기록이 없습니다.")
        return
        
    # 헤더와 다운로드 버튼을 나란히 배치 (2.3:0.7 비율)
    col_title, col_btn = st.columns([2.3, 0.7])
    with col_title:
        st.markdown("<h2 style='margin:0; padding-top:4px;'>📋 우리 아이 아토피 상태 기록보기</h2>", unsafe_allow_html=True)
    with col_btn:
        df = pd.DataFrame(my_hist)
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 전체 기록 CSV 다운로드",
            data=csv,
            file_name=f"{st.session_state.username}_atocatch_history.csv",
            mime="text/csv",
            use_container_width=True,
            type="secondary"
        )
        
    # 이미지 분석 기록만 추출하여 그래프 그리기
    img_recs = [r for r in my_hist if r.get("type") == "이미지분석" and "atopy_prob" in r]
    
    if img_recs:
        st.markdown("<h4 style='margin-top:20px; margin-bottom:15px; color:#111;'>📈 아토피 예측 확률 추이</h4>", unsafe_allow_html=True)
        
        dates = [r["time"] for r in img_recs]
        probs = [r["atopy_prob"] * 100 for r in img_recs]
        
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=probs,
                mode="lines+markers",
                line=dict(color="#1B6554", width=3.5),
                marker=dict(size=11, color="#1B6554"),
                name="아토피 확률(%)"
            ))
            fig.add_hline(
                y=30, line_dash="dash", line_color="#ef4444", line_width=2.0,
                annotation_text="<b>⚠️ 주의선 (30%)</b>",
                annotation_position="bottom right",
                annotation_font=dict(size=13, color="#ef4444", family="'Plus Jakarta Sans', 'Noto Sans KR', sans-serif")
            )
            fig.update_layout(
                height=280, margin=dict(l=55, r=20, t=15, b=40),
                yaxis=dict(
                    range=[0, 100],
                    title=dict(
                        text="<b>확률 (%)</b>",
                        font=dict(size=14, color="#1E293B")
                    ),
                    tickfont=dict(size=12, color="#475569"),
                    gridcolor="#f1f5f9"
                ),
                xaxis=dict(
                    tickfont=dict(size=12, color="#475569"),
                    gridcolor="#f1f5f9"
                ),
                plot_bgcolor="white", paper_bgcolor="white",
                showlegend=False,
                font=dict(family="'Plus Jakarta Sans', 'Noto Sans KR', sans-serif")
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"그래프 렌더링 오류: {e}")
    
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-bottom:15px; color:#111;'>🗂 분석 기록 상세 목록</h4>", unsafe_allow_html=True)
    
    for idx, rec in enumerate(reversed(my_hist)):
        icon = "📊" if rec.get("type") == "설문조사" else ("📷" if rec.get("type") == "이미지분석" else "📝")
        with st.expander(f"{icon} {rec['time']} - {rec['type']}"):
            st.markdown(f"<div style='font-size:1.05rem; font-weight:600; color:#334155; margin-bottom:12px;'>분석 결과: {rec['detail']}</div>", unsafe_allow_html=True)
            
            # 이미지 분석 사진 표시 (원본 및 히트맵)
            if rec.get("type") == "이미지분석" and ("image_b64" in rec or "gradcam_b64" in rec):
                st.markdown("<div style='margin: 15px 0;'>", unsafe_allow_html=True)
                col_orig, col_grad = st.columns(2)
                with col_orig:
                    if "image_b64" in rec:
                        st.image(f"data:image/jpeg;base64,{rec['image_b64']}", caption="원본 피부 사진", use_container_width=True)
                with col_grad:
                    if "gradcam_b64" in rec:
                        st.image(f"data:image/jpeg;base64,{rec['gradcam_b64']}", caption="AI 분석 히트맵 (Grad-CAM)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("<hr style='margin: 12px 0; border-color: #f1f5f9;'>", unsafe_allow_html=True)
            
            # 개별 기록 삭제 및 HTML 다운로드 버튼
            real_idx = len(my_hist) - 1 - idx
            col_space, col_dl, col_del = st.columns([4, 1.5, 1.2])
            with col_dl:
                if rec.get("type") == "이미지분석":
                    rec_html = generate_html_report(
                        display_name=st.session_state.display_name,
                        time_str=rec.get("time", ""),
                        detail=rec.get("detail", ""),
                        image_b64=rec.get("image_b64"),
                        gradcam_b64=rec.get("gradcam_b64")
                    )
                    st.download_button(
                        label="📄 HTML 보고서 받기",
                        data=rec_html,
                        file_name=f"atocatch_report_{rec.get('time', '').replace(' ', '_').replace(':', '')}.html",
                        mime="text/html",
                        key=f"dl_{real_idx}",
                        use_container_width=True
                    )
            with col_del:
                if st.button("🗑 이 기록 삭제", key=f"del_{real_idx}", type="secondary", use_container_width=True):
                    hist = load_history()
                    if st.session_state.username in hist:
                        user_hist = hist[st.session_state.username]
                        if 0 <= real_idx < len(user_hist):
                            user_hist.pop(real_idx)
                            save_history(hist)
                            st.success("기록이 성공적으로 삭제되었습니다.")
                            time.sleep(0.5)
                            st.rerun()

# ==========================================
# 앱 실행 컨트롤러
# ==========================================
def main():
    if not st.session_state.is_logged_in:
        if st.session_state.auth_page == 'login':
            render_login()
        else:
            render_signup()
    else:
        apply_main_css()
        st.markdown('<div class="nav-bar-marker"></div>', unsafe_allow_html=True)
        col_nav, col_logout = st.columns([9.2, 0.8])
        
        with col_nav:
            nav_cols = st.columns([1, 1.1, 1.1, 1.2, 1.1, 1.1, 1])
            pages_left = {
                "홈": "홈",
                "설문조사": "설문조사",
                "피부 스캔": "피부 스캔"
            }
            pages_right = {
                "분석 결과": "분석 결과",
                "AI 챗봇 서비스": "AI 챗봇",
                "기록보기": "기록보기"
            }
            
            # Left pages (0, 1, 2)
            for i, (page_name, label) in enumerate(pages_left.items()):
                with nav_cols[i]:
                    is_active = (st.session_state.current_page == page_name)
                    st.markdown(f'<div class="nav-btn-wrapper {"active" if is_active else ""}"></div>', unsafe_allow_html=True)
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(label, key=f"nav_btn_{page_name}", use_container_width=True, type=btn_type):
                        st.session_state.current_page = page_name
                        st.rerun()
            
            # Middle Brand Logo (3)
            with nav_cols[3]:
                st.markdown('<div class="nav-logo-wrapper"></div>', unsafe_allow_html=True)
                if st.button(" ", key="nav_logo_btn", use_container_width=True):
                    st.session_state.current_page = "홈"
                    st.rerun()
            
            # Right pages (4, 5, 6)
            for i, (page_name, label) in enumerate(pages_right.items()):
                with nav_cols[i + 4]:
                    is_active = (st.session_state.current_page == page_name)
                    st.markdown(f'<div class="nav-btn-wrapper {"active" if is_active else ""}"></div>', unsafe_allow_html=True)
                    btn_type = "primary" if is_active else "secondary"
                    if st.button(label, key=f"nav_btn_{page_name}", use_container_width=True, type=btn_type):
                        st.session_state.current_page = page_name
                        st.rerun()
                
        with col_logout:
            st.markdown('<div class="nav-logout-wrapper"></div>', unsafe_allow_html=True)
            if st.button("로그아웃", use_container_width=True):
                st.session_state.is_logged_in = False
                st.session_state.current_page = "홈"
                st.rerun()
                
        st.markdown("<hr style='margin-top: 0px; margin-bottom: 0.5rem; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        
        if st.session_state.current_page == "홈": render_main()
        elif st.session_state.current_page == "설문조사": render_survey()
        elif st.session_state.current_page == "피부 스캔": render_image_scan()
        elif st.session_state.current_page == "분석 결과": render_result()
        elif st.session_state.current_page == "AI 챗봇 서비스": render_guide()
        elif st.session_state.current_page == "기록보기": render_history()

if __name__ == "__main__":
    main()

# streamlit run app_main.py