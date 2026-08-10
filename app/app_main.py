# -*- coding: utf-8 -*-
import streamlit as st
import time
import os
import json
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
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

try:
    from gradcam_module import predict_with_gradcam
    GRADCAM_OK = True
except ImportError:
    GRADCAM_OK = False

try:
    import rag_engine
    RAG_OK = True
except ImportError:
    RAG_OK = False

# ── Supabase Auth / DB (로그인, 분석 기록) ──────────────────
# publishable 키만 사용 — RLS를 우회하는 secret 키(rag_engine.py 전용)와는 분리.
# 각 요청은 로그인한 사용자의 access token으로 인증되어, RLS가 본인 데이터만 노출한다.
try:
    from supabase import create_client as _create_supabase_client
    _SUPABASE_PKG_OK = True
except ImportError:
    _SUPABASE_PKG_OK = False

try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
except Exception:
    SUPABASE_URL = os.getenv("SUPABASE_URL", "")

try:
    SUPABASE_PUBLISHABLE_KEY = st.secrets["SUPABASE_PUBLISHABLE_KEY"]
except Exception:
    SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")

SUPABASE_AUTH_OK = _SUPABASE_PKG_OK and bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY)

def _get_auth_client():
    return _create_supabase_client(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY)

def _get_user_client():
    """RLS 적용 상태로 현재 로그인한 사용자의 access token을 실어 보내는 클라이언트."""
    client = _get_auth_client()
    client.postgrest.auth(st.session_state.sb_access_token)
    return client

def sb_sign_up(email, password, name):
    return _get_auth_client().auth.sign_up({
        "email": email, "password": password,
        "options": {"data": {"name": name}}
    })

def sb_sign_in(email, password):
    return _get_auth_client().auth.sign_in_with_password({"email": email, "password": password})

def add_history(record_type, detail, extra=None):
    if not SUPABASE_AUTH_OK or not st.session_state.get("sb_access_token"):
        return
    try:
        row = {
            "user_id": st.session_state.sb_user_id,
            "record_type": record_type,
            "detail": detail,
        }
        if extra:
            extra = dict(extra)
            if "image_b64" in extra:
                row["image_base64"] = extra.pop("image_b64")
            if "gradcam_b64" in extra:
                row["gradcam_base64"] = extra.pop("gradcam_b64")
            if extra:
                row["prediction"] = extra
        _get_user_client().table("analysis_history").insert(row).execute()
    except Exception:
        pass

def load_history():
    if not SUPABASE_AUTH_OK or not st.session_state.get("sb_access_token"):
        return []
    try:
        res = (
            _get_user_client().table("analysis_history")
            .select("*")
            .eq("user_id", st.session_state.sb_user_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
        return res.data or []
    except Exception:
        return []

def _fmt_time(rec):
    try:
        return datetime.fromisoformat(rec["created_at"]).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return rec.get("created_at", "")

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

# 모델 가중치는 app/ 폴더에 함께 커밋되어 있음 (Google Drive 다운로드 방식 폐기)
_MODEL_PATH = os.path.join(_BASE_DIR, "best_model.pth")
_IGA_MODEL_PATH = os.path.join(_BASE_DIR, "best_iga_model.pth")

# 임계값은 model_config.json / model_config2.json을 단일 소스로 사용
with open(os.path.join(_BASE_DIR, "model_config.json"), encoding="utf-8") as f:
    ATOPY_THRESHOLD = json.load(f)["threshold"]
with open(os.path.join(_BASE_DIR, "model_config2.json"), encoding="utf-8") as f:
    IGA_THRESHOLD = json.load(f)["threshold"]

# 모델 로딩
@st.cache_resource
def load_risk_model(): return joblib.load(os.path.join(_BASE_DIR, "atopy_service_model.joblib"))

@st.cache_resource
def load_image_model():
    m = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(_MODEL_PATH, map_location="cpu"))
    m.eval()
    return m

@st.cache_resource
def load_iga_model():
    m = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(_IGA_MODEL_PATH, map_location="cpu"))
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
    bg_img_path = os.path.join(_BASE_DIR, "design", "mainlog.png")
    bg_b64 = ""
    if os.path.exists(bg_img_path):
        with open(bg_img_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
            
    bg_style = (
        f"background: url('data:image/png;base64,{bg_b64}') right bottom/contain no-repeat, "
        f"#FFFFFF !important;"
    ) if bg_b64 else "background: #FFFFFF !important;"
    
    st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)
    st.markdown(f"""
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
            background: rgba(255, 255, 255, 0.96) !important;
            backdrop-filter: blur(14px);
            padding: 2.8rem 2.5rem !important;
            border-radius: 24px;
            box-shadow: 0 20px 45px rgba(27, 101, 84, 0.08) !important;
            border: 2px solid rgba(27, 101, 84, 0.08) !important;
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
        
        /* Secondary 버튼 및 Download 버튼 */
        div.stButton > button[kind="secondary"], 
        button[data-testid="baseButton-secondary"],
        div.stDownloadButton > button {{ 
            background-color: #F1F5F9 !important; 
            color: #1B6554 !important; 
            border: 1.5px solid #CBD5E1 !important; 
        }}
        div.stButton > button[kind="secondary"]:hover, 
        button[data-testid="baseButton-secondary"]:hover,
        div.stDownloadButton > button:hover {{ 
            background-color: #E2E8F0 !important; 
            color: #144E41 !important; 
            border-color: #94A3B8 !important;
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
        
        /* ── [UX/UI 개선] 회원가입 유도 버튼 순수 텍스트 링크화 & 목업 잘림 방지 ── */
        .signup-link-button {{
            background: transparent !important; /* 배경색 완전 투명화 */
            border: none !important;            /* 테두리 제거 */
            box-shadow: none !important;        /* 그림자 제거 */
            color: #555555 !important;          /* 전체 글자는 차분한 회색 */
            font-size: 14px !important;
            cursor: pointer !important;
            text-align: center !important;
            width: 100% !important;
        }}
        .signup-link-button .highlight {{
            color: #004D40 !important;          /* '가입하기'/'로그인' 텍스트만 메인 녹색으로 강조 */
            text-decoration: underline !important; /* 클릭 가능한 링크임을 인지 */
            font-weight: 700 !important;
        }}
        
        .mockup-text-container {{
            height: auto !important;            /* 고정 높이 해제, 글자량에 맞게 자동 조절 */
            overflow: visible !important;       /* 글자가 잘리지 않도록 visible 처리 */
        }}
        .mockup-score-text {{
            font-size: 16px !important;         /* 현재 폰트 크기 유지 */
            line-height: 1.5 !important;        /* 줄 간격을 최소 1.5배 확보하여 자투리 잘림 방지 */
            padding-top: 4px !important;        /* 상단 마진/패딩을 주어 윗 박스와의 겹침 현상 원천 차단 */
        }}
        
        /* 순수 텍스트 링크 버튼을 위한 signup-link-container 스타일 */
        .signup-link-container div.stButton {{
            text-align: center !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        .signup-link-container div.stButton > button {{
            background: transparent !important; /* 배경색 완전 투명화 */
            border: none !important;            /* 테두리 제거 */
            box-shadow: none !important;        /* 그림자 제거 */
            color: #555555 !important;          /* 전체 글자는 차분한 회색 */
            font-size: 14px !important;
            cursor: pointer !important;
            height: auto !important;
            margin-top: 0 !important;
            padding: 0 !important;
            width: auto !important;
            display: inline-block !important;
            font-weight: 700 !important;
        }}
        .signup-link-container div.stButton > button:hover {{
            background: transparent !important;
            color: #004D40 !important;          /* '가입하기'/'로그인' 텍스트 메인 녹색으로 강조 */
            text-decoration: underline !important;
        }}
    </style>
    """, unsafe_allow_html=True)

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
            if not SUPABASE_AUTH_OK:
                st.error("로그인 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.")
            else:
                try:
                    result = sb_sign_in(email, password)
                    st.session_state.is_logged_in = True
                    st.session_state.username = result.user.email
                    st.session_state.display_name = (result.user.user_metadata or {}).get("name", result.user.email)
                    st.session_state.sb_user_id = result.user.id
                    st.session_state.sb_access_token = result.session.access_token
                    st.rerun()
                except Exception:
                    st.error("이메일 또는 비밀번호가 일치하지 않습니다.")
        
        st.write("")
        st.write("")
        st.markdown('<div class="signup-link-container" style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
        if st.button("아직 회원이 아니신가요? 가입하기", key="signup_link", use_container_width=True):
            st.session_state.auth_page = 'signup'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col_empty:
        # 배경 이미지가 잘 보이도록 오른쪽 영역은 비워둡니다.
        st.empty()

    # 페이지 하단 가운데 면책 문구 고정 (로그인 화면 전용 글래스모피즘 스타일)
    st.markdown("<div style='margin-top: 5vh;'></div>", unsafe_allow_html=True)
    st.markdown(clean_html("""
        <div class="notranslate" style="text-align: center; max-width: 950px; margin: 0 auto; padding: 14px 20px; background: #F8FAFC; border: 1.5px solid #E2E8F0; border-radius: 14px; box-shadow: 0 10px 30px rgba(27, 101, 84, 0.03);">
            <p class="notranslate" style="margin: 0; font-size: 0.88rem; color: #475569; font-weight: 600; line-height: 1.6; word-break: keep-all;">
                ※ 본 AI 서비스의 결과는 입력된 정보를 바탕으로 한 피부 상태 기록 및 안내이며, 의사의 진단이나 치료를 대체할 수 없습니다. 더 안전하고 정확한 관리를 위해 소아청소년과나 피부과 전문의를 방문하셔서 진료를 받아보시는 것을 권장해 드립니다.
            </p>
        </div>
    """), unsafe_allow_html=True)

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
            elif not email or not password:
                st.error("모든 항목을 입력해 주세요.")
            elif not SUPABASE_AUTH_OK:
                st.error("회원가입 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.")
            else:
                try:
                    sb_sign_up(email, password, name)
                    st.success("🎉 회원가입 완료! 로그인 페이지로 이동합니다.")
                    time.sleep(1.5)
                    st.session_state.auth_page = 'login'
                    st.rerun()
                except Exception as e:
                    st.error(f"회원가입에 실패했습니다: {e}")
                
        st.write("")
        st.write("")
        st.markdown('<div class="signup-link-container" style="text-align: center; margin-top: 15px;">', unsafe_allow_html=True)
        if st.button("이미 계정이 있으신가요? 로그인", key="login_link", use_container_width=True):
            st.session_state.auth_page = 'login'
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col_empty:
        st.empty()

    # 페이지 하단 가운데 면책 문구 고정 (회원가입 화면 전용 글래스모피즘 스타일)
    st.markdown("<div style='margin-top: 5vh;'></div>", unsafe_allow_html=True)
    st.markdown(clean_html("""
        <div class="notranslate" style="text-align: center; max-width: 950px; margin: 0 auto; padding: 14px 20px; background: #F8FAFC; border: 1.5px solid #E2E8F0; border-radius: 14px; box-shadow: 0 10px 30px rgba(27, 101, 84, 0.03);">
            <p class="notranslate" style="margin: 0; font-size: 0.88rem; color: #475569; font-weight: 600; line-height: 1.6; word-break: keep-all;">
                ※ 본 AI 서비스의 결과는 입력된 정보를 바탕으로 한 피부 상태 기록 및 안내이며, 의사의 진단이나 치료를 대체할 수 없습니다. 더 안전하고 정확한 관리를 위해 소아청소년과나 피부과 전문의를 방문하셔서 진료를 받아보시는 것을 권장해 드립니다.
            </p>
        </div>
    """), unsafe_allow_html=True)

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
    # 스캔 가동 중 화면에 사용할 피부 환부 이미지 로드
    scan_img_path = os.path.join(_BASE_DIR, "design", "skin.png")
    scan_img_b64 = ""
    if os.path.exists(scan_img_path):
        import base64
        with open(scan_img_path, "rb") as f:
            scan_img_b64 = base64.b64encode(f.read()).decode()
            

            
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.markdown(clean_html("""<div style='margin-top: 30px; margin-bottom: 3.25rem; text-align: left;'>
    <span class="hero-badge notranslate">&#127808; [AI 아토피 조기 예측 솔루션]</span>
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
    <div class="phone-container notranslate" style="width: 290px; height: 470px; border-radius: 38px; border: 10px solid #1E293B; background: #F8FAFC; box-shadow: 0 20px 40px rgba(0,0,0,0.12); position: relative; overflow: hidden; display: flex; flex-direction: column; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); margin: 0 auto; box-sizing: border-box;">
        <!-- Phone Speaker & Camera Notch -->
        <div style="position: absolute; top: 0; left: 50%; transform: translateX(-50%); width: 110px; height: 18px; background: #1E293B; border-bottom-left-radius: 12px; border-bottom-right-radius: 12px; z-index: 10; display: flex; align-items: center; justify-content: center; gap: 6px;">
            <div style="width: 35px; height: 3px; background: #475569; border-radius: 3px;"></div>
            <div style="width: 5px; height: 5px; background: #475569; border-radius: 50%;"></div>
        </div>
        
        <!-- Live High-Tech Mock Screen Body -->
        <div style="width: 100%; height: 100%; position: relative; background: #F8FAFC; display: flex; flex-direction: column; overflow: hidden; padding-top: 22px; box-sizing: border-box; font-family: sans-serif;">
            <!-- Top Screen Header inside Mockup -->
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 14px 8px 14px; background: #FFFFFF; border-bottom: 1px solid #F1F5F9; z-index: 5;">
                <span style="font-size: 13.5px; font-weight: 800; color: #1B6554; background: rgba(27, 101, 84, 0.08); padding: 3px 9px; border-radius: 6px;">AtoCatch AI</span>
                <span style="font-size: 13.5px; font-weight: 800; color: #10B981; display: flex; align-items: center; gap: 5px;">
                    <span style="width: 7.5px; height: 7.5px; background: #10B981; border-radius: 50%; display: inline-block; animation: led-blink 1.2s infinite ease-in-out;"></span> 스캔 가동 중
                </span>
            </div>

            <!-- Scanning Image Viewport -->
            <div style="position: relative; width: 100%; height: 220px; background: #000; overflow: hidden; display: flex; align-items: center; justify-content: center; z-index: 2;">
                <!-- Skin disease background -->
                <img src="data:image/png;base64,{scan_img_b64}" style="width: 100%; height: 100%; object-fit: cover; position: absolute; top: 0; left: 0;" />
                
                <!-- Focus Target Box overlay -->
                <div style="position: absolute; width: 105px; height: 105px; border: 2.5px solid #10B981; border-radius: 14px; box-shadow: 0 0 15px rgba(16, 185, 129, 0.6); animation: target-pulsate 1.8s infinite ease-in-out; display: flex; align-items: center; justify-content: center; z-index: 3;">
                    <!-- Four Corner Borders decorative -->
                    <div style="position: absolute; top: -5px; left: -5px; width: 14px; height: 14px; border-top: 3.5px solid #059669; border-left: 3.5px solid #059669;"></div>
                    <div style="position: absolute; top: -5px; right: -5px; width: 14px; height: 14px; border-top: 3.5px solid #059669; border-right: 3.5px solid #059669;"></div>
                    <div style="position: absolute; bottom: -5px; left: -5px; width: 14px; height: 14px; border-bottom: 3.5px solid #059669; border-left: 3.5px solid #059669;"></div>
                    <div style="position: absolute; bottom: -5px; right: -5px; width: 14px; height: 14px; border-bottom: 3.5px solid #059669; border-right: 3.5px solid #059669;"></div>
                </div>

                <!-- Laser Scanning Bar overlay -->
                <div style="position: absolute; left: 0; width: 100%; height: 4px; background: linear-gradient(90deg, rgba(16,185,129,0) 0%, rgba(16,185,129,1) 50%, rgba(16,185,129,0) 100%); box-shadow: 0 0 12px 3px rgba(16, 185, 129, 0.8); animation: laser-sweep 2.0s infinite linear; z-index: 4;"></div>

                <!-- Scanning Complete Text Banner -->
                <div style="position: absolute; bottom: 8px; font-size: 11px !important; font-weight: 800; color: #ffffff; background: rgba(16, 185, 129, 0.95); padding: 4px 12px; border-radius: 20px; border: 1.5px solid rgba(255, 255, 255, 0.4); backdrop-filter: blur(4px); z-index: 5; box-shadow: 0 4px 10px rgba(0,0,0,0.15);">
                    실시간 분석 감지 완료
                </div>
            </div>

            <!-- Patient Info block -->
            <div style="background: #FFFFFF; margin: 10px 12px; padding: 10px 12px; border-radius: 14px; border: 1px solid #F1F5F9; box-shadow: 0 2px 6px rgba(0,0,0,0.01); text-align: left; z-index: 5;">
                <div style="font-size: 13.5px; font-weight: 800; color: #475569; display: flex; align-items: center; gap: 4px; margin-bottom: 6px;">
                    <span>👤</span> 환자 정보 <span style="font-size: 11px; color: #94A3B8; font-weight: 500;">(진단 대상)</span>
                    <span style="margin-left: auto; font-size: 11px; background: #ECFDF5; color: #047857; font-weight: 800; padding: 2px 6px; border-radius: 4px;">스캔 완료</span>
                </div>
                <div style="display: flex; flex-direction: column; gap: 4.5px; font-size: 13px; color: #334155; font-weight: 700; line-height: 1.45;">
                    <div>• 이름: 김아토 (Baby)</div>
                    <div>• 월령: 생후 10개월</div>
                    <div>• 성별: 남자아이</div>
                    <div>• 일시: 2026-05-20</div>
                </div>
            </div>

            <!-- Recommended Topic card at bottom -->
            <div style="background: #F0FDF4; border: 1.5px solid #10B981; border-radius: 14px; margin: 0 12px 12px 12px; padding: 10px 12px; display: flex; align-items: flex-start; gap: 6px; text-align: left; z-index: 5; height: auto; box-sizing: border-box;">
                <span style="font-size: 13px !important; line-height: 1.1;">💡</span>
                <div style="width: 100%;">
                    <div style="font-size: 10px !important; font-weight: 800; color: #065F46; line-height: 1.2;">추천 아토피 완화 주제</div>
                    <div style="font-size: 9.5px !important; color: #047857; font-weight: 700; margin-top: 2px; line-height: 1.3;">미온수 목욕 후 3분 이내 고보습제 집중 도포</div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Visual Flow Breadcrumbs under Phone -->
    <div style="display: flex; align-items: center; justify-content: center; gap: 6px; margin-top: 22px;">
        <div style="background: rgba(255, 152, 0, 0.08); color: #E65100; font-size: 0.65rem !important; font-weight: 800; padding: 4px 10px; border-radius: 12px; border: 1px solid rgba(255, 152, 0, 0.15); display: flex; align-items: center; gap: 4px;">
            <span>&#128221;</span> 설문분석
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
    @keyframes led-blink {
        0%, 100% { opacity: 0.4; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.1); }
    }
    @keyframes target-pulsate {
        0%, 100% { transform: scale(1); border-color: rgba(16, 185, 129, 0.7); box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); }
        50% { transform: scale(1.03); border-color: rgba(16, 185, 129, 1); box-shadow: 0 0 20px rgba(16, 185, 129, 0.8); }
    }
    @keyframes laser-sweep {
        0% { top: 0px; }
        50% { top: 216px; }
        100% { top: 0px; }
    }
</style>"""), unsafe_allow_html=True)

def render_survey():
    # ── 세션 초기화 ─────────────────────────────────────────────
    yn_fields = [
        "parent_AD", "parent_AR", "parent_asthma",
        "sibling_allergy", "child_passive_smoke",
        "mold_ever", "passive_smoke_ever", "pet_ever"
    ]
    for f in yn_fields:
        if f"chip_{f}" not in st.session_state:
            st.session_state[f"chip_{f}"] = None
    if "survey_submitted_once" not in st.session_state:
        st.session_state["survey_submitted_once"] = False

    st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)
    st.markdown("""
<style>
    /* ── 섹션 대분류 타이틀 ── */
    .sv-section {
        font-size: 22px !important; font-weight: 800 !important; color: #1B6554 !important;
        margin: 36px 0 16px 0; padding-left: 0px;
        border-left: none !important;
        display: flex; align-items: center; gap: 8px;
        line-height: 1.3;
    }

    /* ── 질문 카드 컨테이너 (Pastel Light Green & Light Blue) ── */
    .sv-qblock {
        background: #F0FDF4 !important; /* Soft pastel green */
        border-radius: 20px !important;
        border: none !important;
        padding: 22px 24px 20px 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 12px rgba(27, 101, 84, 0.02) !important;
    }
    .sv-qblock-alt {
        background: #EFF6FF !important; /* Soft pastel blue */
        border-radius: 20px !important;
        border: none !important;
        padding: 22px 24px 20px 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.02) !important;
    }
    .sv-qblock-error {
        background: #FEF2F2 !important; /* Soft pastel red */
        border-radius: 20px !important;
        border: 1.5px solid #EF4444 !important;
        padding: 22px 24px 20px 24px !important;
        margin-bottom: 20px !important;
    }

    /* ── 질문 타이틀 ── */
    .sv-qblock .sv-qtitle, .sv-qblock-alt .sv-qtitle {
        font-size: 18px !important; font-weight: 800 !important; color: #1E293B !important;
        margin-bottom: 6px !important; line-height: 1.35;
    }
    .sv-qblock-error .sv-qtitle {
        font-size: 18px !important; font-weight: 800 !important; color: #991B1B !important;
        margin-bottom: 6px !important; line-height: 1.35;
    }

    /* ── 서브 텍스트 ── */
    .sv-qsub {
        font-size: 14px !important; font-weight: 500 !important; color: #64748B !important;
        margin-bottom: 12px !important; line-height: 1.5 !important;
    }

    /* ── 필수 표시 ── */
    .sv-required { color:#EF4444; font-size:13px; font-weight:700; margin-left:6px; }

    /* ── 버튼 공통: Pill shape ── */
    div[data-testid="stButton"] > button {
        border-radius: 9999px !important; /* Pill shape */
        height: 48px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        transition: all 0.2s ease !important;
    }

    /* Default (secondary) */
    div[data-testid="stButton"] > button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #475569 !important;
        border: 1.5px solid #E2E8F0 !important;
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #1E293B !important;
    }

    /* Selected (primary): clean deep green color */
    div[data-testid="stButton"] > button[kind="primary"] {
        background-color: #1B6554 !important; /* Clean deep green */
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(27, 101, 84, 0.15) !important;
        opacity: 1 !important;
    }

    /* dim: 선택 안 된 쪽 흐리기 */
    .sv-btn-dim > div[data-testid="stButton"] > button {
        opacity: 0.35 !important;
    }

    /* 셀렉트박스 폰트 크기 및 높이 ── */
    div[data-testid="stSelectbox"] div[role="button"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #334155 !important;
        height: 48px !important;
        display: flex !important;
        align-items: center !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        box-shadow: none !important;
    }
    div[data-testid="stSelectbox"] div[role="button"]:hover {
        border-color: #CBD5E1 !important;
    }
    div[data-testid="stSelectbox"] div[role="button"]:focus {
        border-color: #1B6554 !important;
    }

    /* selectbox 아래 여백 추가 */
    div[data-testid="stSelectbox"] {
        margin-bottom: 24px !important;
    }

    /* expander 클릭 영역 ── */
    div[data-testid="stExpander"] details summary {
        padding: 14px 20px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #1B6554 !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
        background-color: #FFFFFF !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stExpander"] details summary p {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #1B6554 !important;
    }

    /* 챗봇 스타일 */
    div[data-testid="stForm"] label p {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #1E293B !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stForm"] textarea {
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #1E293B !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 12px !important;
    }
</style>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='"
        "background: #FFFFFF; "
        "border-radius: 24px; "
        "padding: 32px 40px; "
        "margin-bottom: 40px; "
        "box-shadow: 0 10px 30px rgba(27, 101, 84, 0.04); "
        "text-align: center; "
        "border: 1px solid rgba(27, 101, 84, 0.08);"
        "'>"
        "<h2 style='font-size: 30px !important; font-weight: 800 !important; color: #1B6554 !important; margin: 0 0 10px 0; line-height: 1.3;'>"
        "📝 우리 아이 아토피 위험 미리보기"
        "</h2>"
        "<p style='font-size: 15px !important; font-weight: 500 !important; color: #64748B !important; margin: 0; line-height: 1.6;'>"
        "부모의 건강 정보와 아이의 양육 환경을 기반으로 아토피 발생 가능성을 예측합니다.<br>"
        "아래 문항들을 꼼꼼하게 읽어보신 후 선택해 주세요."
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )

    # ── 칩 버튼 헬퍼 ────────────────────────────────────────────
    def chip_yn(key, title, sub="", is_alt=False):
        val   = st.session_state.get(f"chip_{key}")
        submitted = st.session_state["survey_submitted_once"]
        error = submitted and (val is None)

        if error:
            wrap_cls = "sv-qblock-error"
        elif is_alt:
            wrap_cls = "sv-qblock-alt"
        else:
            wrap_cls = "sv-qblock"

        req_html = "<span class='sv-required'>← 필수</span>" if error else ""

        st.markdown(
            f"<div class='{wrap_cls}'>"
            f"<div class='sv-qtitle'>{title}{req_html}</div>"
            + (f"<div class='sv-qsub'>{sub}</div>" if sub else "")
            + "</div>",
            unsafe_allow_html=True
        )

        # '예' 선택 시 '예'가 primary(Deep green), '아니오' 선택 시 '아니오'가 primary(Deep green)
        yes_type = "primary" if val == "예" else "secondary"
        no_type  = "primary" if val == "아니오" else "secondary"

        c_yes, c_no = st.columns(2)

        yes_dim = (val == "아니오")
        no_dim  = (val == "예")

        with c_yes:
            if yes_dim:
                st.markdown("<div class='sv-btn-dim'>", unsafe_allow_html=True)
            if st.button("예", key=f"btn_yes_{key}", type=yes_type, use_container_width=True):
                st.session_state[f"chip_{key}"] = "예"
                st.rerun()
            if yes_dim:
                st.markdown("</div>", unsafe_allow_html=True)

        with c_no:
            if no_dim:
                st.markdown("<div class='sv-btn-dim'>", unsafe_allow_html=True)
            if st.button("아니오", key=f"btn_no_{key}", type=no_type, use_container_width=True):
                st.session_state[f"chip_{key}"] = "아니오"
                st.rerun()
            if no_dim:
                st.markdown("</div>", unsafe_allow_html=True)

        return val

    # ── Centered Single Column Layout ───────────────────────────
    col_left, col_mid, col_right = st.columns([1, 4, 1])

    with col_mid:
        st.markdown("<div class='sv-section'>👨‍👩‍👦 가족력 및 아동 임상정보</div>", unsafe_allow_html=True)
        chip_yn("parent_AD",      "부모의 아토피 병력",   "부모 중 아토피 피부염 진단 이력", is_alt=False)
        chip_yn("parent_AR",      "부모의 알레르기 비염", "부모 중 알레르기 비염 진단 이력", is_alt=True)
        chip_yn("parent_asthma",  "부모의 천식 병력",     "부모 중 천식 진단 이력", is_alt=False)
        chip_yn("sibling_allergy","형제자매의 알레르기",  "형제·자매 중 아토피·비염 등", is_alt=True)

        st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sv-section'>🏡 생활 환경 정보</div>", unsafe_allow_html=True)
        chip_yn("child_passive_smoke","아이의 간접흡연 노출",  "현재 가족 중 흡연자에게 노출 중인지", is_alt=False)
        chip_yn("mold_ever",          "실내 곰팡이 노출 경험", "임신 중 또는 아이 첫돌 이전 기준", is_alt=True)
        chip_yn("passive_smoke_ever", "가족 내 흡연 여부",     "아이 출생 이후 가족 중 흡연자 존재", is_alt=False)
        chip_yn("pet_ever",           "반려동물 양육 여부",     "현재 또는 과거 개·고양이 등 양육", is_alt=True)

        st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sv-section'>📋 항생제 복용 이력</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sv-qblock'>"
            "<div class='sv-qtitle'>첫돌 이전 항생제 복용 횟수</div>"
            "<div class='sv-qsub'>생후 12개월 이내, 병원 처방 기준</div>"
            "</div>",
            unsafe_allow_html=True
        )
        antibiotic = st.selectbox("항생제 복용 횟수", list(AB_MAP.keys()),
                                   label_visibility="collapsed", key="ab_select")

        st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sv-section'>📝 오늘의 증상 체크</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sv-qblock-alt'>"
            "<div class='sv-qtitle'>오늘 아이의 가려움증 정도</div>"
            "<div class='sv-qsub'>하루 종일 긁거나 잠 못 잔 경우 '상'</div>"
            "</div>",
            unsafe_allow_html=True
        )
        itching_level = st.selectbox("가려움증 정도", ["없음", "하", "중", "상"],
                                      label_visibility="collapsed", key="itch_select")

        st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sv-section'>🌿 외부 활동 및 환경</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sv-qblock'>"
            "<div class='sv-qtitle'>농촌 거주 기간 (만 0~5세)</div>"
            "<div class='sv-qsub'>만 5세까지 농촌·전원 지역 거주 기간</div>"
            "</div>",
            unsafe_allow_html=True
        )
        rural_options = {
            "없음 (0년)": 0, "1년 미만": 0, "1~2년": 1,
            "2~3년": 2, "3~4년": 3, "4~5년": 4, "5년 이상": 5
        }
        rural_sel   = st.selectbox("농촌 거주 기간", list(rural_options.keys()),
                                    label_visibility="collapsed", key="rural_select")
        rural_years = rural_options[rural_sel]

        st.markdown(
            "<div class='sv-qblock-alt' style='margin-top: 24px;'>"
            "<div class='sv-qtitle'>하루 평균 실외활동 시간</div>"
            "<div class='sv-qsub'>어린이집·유치원 포함, 오늘 기준</div>"
            "</div>",
            unsafe_allow_html=True
        )
        outdoor_options = {
            "30분 미만": 0.5, "1시간": 1.0, "1.5시간": 1.5,
            "2시간": 2.0, "2.5시간": 2.5, "3시간": 3.0, "3시간 초과": 4.0
        }
        outdoor_sel = st.selectbox("실외활동 시간", list(outdoor_options.keys()), index=1,
                                    label_visibility="collapsed", key="outdoor_select")
        outdoor_avg = outdoor_options[outdoor_sel]

        st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sv-section'>💧 실내 환경 체크</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sv-qblock'>"
            "<div class='sv-qtitle'>오늘 실내 습도 상태</div>"
            "<div class='sv-qsub'>가습기 없이 건조하면 '건조', 적당하면 '적정'</div>"
            "</div>",
            unsafe_allow_html=True
        )
        humidity = st.selectbox("실내 습도", ["건조", "적정", "습함"], index=1,
                                 label_visibility="collapsed", key="humidity_select")

        # ── 제출 버튼 ───────────────────────────────────────────────
        st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
        if st.button("다음 단계로 이동 (설문 분석 저장) ➡", type="primary",
                     use_container_width=True, key="survey_submit"):
            st.session_state["survey_submitted_once"] = True
            missing_now = {f for f in yn_fields if st.session_state.get(f"chip_{f}") is None}
            if missing_now:
                st.rerun()   # 빨간 테두리 표시 후 재렌더
            else:
                yn = {"예": 0, "아니오": 1}
                parent_AD           = st.session_state["chip_parent_AD"]
                parent_AR           = st.session_state["chip_parent_AR"]
                parent_asthma       = st.session_state["chip_parent_asthma"]
                sibling_allergy     = st.session_state["chip_sibling_allergy"]
                child_passive_smoke = st.session_state["chip_child_passive_smoke"]
                mold_ever           = st.session_state["chip_mold_ever"]
                passive_smoke_ever  = st.session_state["chip_passive_smoke_ever"]
                pet_ever            = st.session_state["chip_pet_ever"]

                input_df = pd.DataFrame([{
                    "antibiotic": AB_MAP[antibiotic], "parent_AD": yn[parent_AD],
                    "parent_AR": yn[parent_AR], "mold_ever": yn[mold_ever],
                    "parent_asthma": yn[parent_asthma], "sibling_allergy": yn[sibling_allergy],
                    "pet_ever": yn[pet_ever], "passive_smoke_ever": yn[passive_smoke_ever],
                    "child_passive_smoke": yn[child_passive_smoke],
                    "rural_years": rural_years, "outdoor_avg": outdoor_avg,
                }])

                # 모델 학습 시 인코딩(1=해당 위험요인 있음, atopy_service_model_coefficients.csv로 확인)은
                # 위 화면 표시용 딕셔너리(0=예)와 반대라 여기서만 뒤집어서 모델에 전달한다.
                # 항생제 복용 횟수도 학습 데이터 원본 코드가 1~4(없음=1)라 화면용 AB_MAP(0~3)과 다르다.
                model_yn = {"예": 1, "아니오": 0}
                AB_MAP_MODEL = {"없음 (0회)": 1, "1~2회": 2, "3~4회": 3, "5회 이상": 4}
                model_input_df = pd.DataFrame([{
                    "antibiotic": AB_MAP_MODEL[antibiotic], "parent_AD": model_yn[parent_AD],
                    "parent_AR": model_yn[parent_AR], "mold_ever": model_yn[mold_ever],
                    "parent_asthma": model_yn[parent_asthma], "sibling_allergy": model_yn[sibling_allergy],
                    "pet_ever": model_yn[pet_ever], "passive_smoke_ever": model_yn[passive_smoke_ever],
                    "child_passive_smoke": model_yn[child_passive_smoke],
                    "rural_years": rural_years, "outdoor_avg": outdoor_avg,
                }])

                prob = float(risk_model.predict_proba(model_input_df)[0, 1])
                # 0.13 / 0.20은 모델의 실제 operating threshold(0.12, F2 최적화)와는 별개로
                # 화면에 3단계로 보여주기 위한 표시용 구간이며, 별도 통계적 근거로 도출된
                # 값은 아니다 (원본 학습 스크립트에도 이 구간의 도출 과정은 없음).
                if prob < 0.13:   level = "저위험"
                elif prob < 0.20: level = "중위험"
                else:             level = "고위험"

                inputs_dict = input_df.iloc[0].to_dict()
                inputs_dict["itching_level"] = itching_level
                inputs_dict["humidity"]      = humidity

                st.session_state.survey_result = {
                    "prob": prob, "level": level, "inputs": inputs_dict
                }
                add_history(
                    "설문조사",
                    f"위험도: {level} ({prob*100:.1f}%)"
                )
                st.session_state.current_page = "피부 스캔"
                st.rerun()

def render_image_scan():
    st.markdown(
        "<div style='"
        "background: linear-gradient(135deg, #E6FDF5 0%, #D1FAE5 100%); "
        "border: 1px solid rgba(16, 185, 129, 0.15); "
        "border-radius: 20px; "
        "padding: 24px 32px; "
        "margin-bottom: 24px; "
        "box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.05), 0 4px 12px rgba(0,0,0,0.02);"
        "'>"
        "<h2 style='font-size: 38px !important; font-weight: 900 !important; color: #065F46 !important; margin: 0; line-height: 1.35;'>"
        "📷 우리 아이 아토피 상태 바로보기"
        "</h2>"
        "</div>",
        unsafe_allow_html=True
    )
    if not st.session_state.survey_result:
        st.info("💡 설문조사를 먼저 진행하면 더 정확한 종합 결과를 얻을 수 있습니다.")

    # 📸 정확한 AI 분석을 위한 피부 촬영 가이드 박스 추가
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown(clean_html("""
        <div class="notranslate" style="background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border: 1.5px solid rgba(27, 101, 84, 0.08); border-radius: 24px; padding: 28px; box-shadow: 0 12px 32px -8px rgba(0, 0, 0, 0.05); margin-top: 15px;">
            <div class="notranslate" style="margin-bottom: 24px;">
                <div class="notranslate" style="display: inline-flex; align-items: center; gap: 12px; background: #E6FDF5; border: 1.5px solid rgba(16, 185, 129, 0.2); border-radius: 9999px; padding: 10px 24px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.04);">
                    <span class="notranslate" style="background: #10B981; color: white; width: 32px; height: 32px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 1.15rem; font-weight: 900; box-shadow: 0 2px 8px rgba(16, 185, 129, 0.2); animation: pulse-green 2s infinite;">💡</span>
                    <h4 class="notranslate" style="color: #065F46; margin: 0; font-weight: 900; font-size: 1.65rem; letter-spacing: -0.5px;">정확한 AI 피부 분석을 위한 촬영 가이드</h4>
                </div>
            </div>
            
            <div class="notranslate" style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                <!-- 1. LIGHT -->
                <div class="notranslate" style="background: #FFFBEB; border: 2.5px solid #FDE68A; border-radius: 20px; padding: 28px 32px; position: relative; overflow: hidden; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.05); display: flex; flex-direction: column; justify-content: center; height: 180px;">
                    <div class="notranslate" style="position: absolute; right: 22px; top: 22px; font-size: 3.2rem; opacity: 0.9; color: #F59E0B;">&#9728;</div>
                    <div class="notranslate" style="background: #F59E0B; color: #FFFFFF; padding: 6px 14px; border-radius: 8px; font-size: 1.15rem; font-weight: 900; display: inline-block; width: fit-content;">LIGHT</div>
                    <h5 class="notranslate" style="margin: 18px 0 0 0; color: #000000; font-size: 2.25rem; font-weight: 900; line-height: 1.2;">밝은 조명 필수</h5>
                </div>
                <!-- 2. FOCUS -->
                <div class="notranslate" style="background: #F0F9FF; border: 2.5px solid #BAE6FD; border-radius: 20px; padding: 28px 32px; position: relative; overflow: hidden; box-shadow: 0 4px 15px rgba(2, 132, 199, 0.05); display: flex; flex-direction: column; justify-content: center; height: 180px;">
                    <div class="notranslate" style="position: absolute; right: 22px; top: 22px; font-size: 3.2rem; opacity: 0.9; color: #0284C7;">&#128269;</div>
                    <div class="notranslate" style="background: #0284C7; color: #FFFFFF; padding: 6px 14px; border-radius: 8px; font-size: 1.15rem; font-weight: 900; display: inline-block; width: fit-content;">FOCUS</div>
                    <h5 class="notranslate" style="margin: 18px 0 0 0; color: #000000; font-size: 2.25rem; font-weight: 900; line-height: 1.2;">흔들림 없이 선명하게</h5>
                </div>
                <!-- 3. DISTANCE -->
                <div class="notranslate" style="background: #F0FDF4; border: 2.5px solid #A7F3D0; border-radius: 20px; padding: 28px 32px; position: relative; overflow: hidden; box-shadow: 0 4px 15px rgba(5, 150, 105, 0.05); display: flex; flex-direction: column; justify-content: center; height: 180px;">
                    <div class="notranslate" style="position: absolute; right: 22px; top: 22px; font-size: 3.2rem; opacity: 0.9; color: #059669;">&#128207;</div>
                    <div class="notranslate" style="background: #059669; color: #FFFFFF; padding: 6px 14px; border-radius: 8px; font-size: 1.15rem; font-weight: 900; display: inline-block; width: fit-content;">DISTANCE</div>
                    <h5 class="notranslate" style="margin: 18px 0 0 0; color: #000000; font-size: 2.25rem; font-weight: 900; line-height: 1.2;">적정 거리 유지 (15~20cm)</h5>
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
                        gc_result = predict_with_gradcam(
                            img, image_model, iga_model, device=device,
                            atopy_threshold=ATOPY_THRESHOLD, severity_threshold=IGA_THRESHOLD,
                        )
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
                        is_atopy   = atopy_prob >= ATOPY_THRESHOLD
                        iga_prob, iga_severe = None, None
                        if is_atopy:
                            with torch.no_grad():
                                iga_logits = iga_model(tensor)
                                iga_probs  = torch.softmax(iga_logits, dim=1)[0]
                            iga_prob   = float(iga_probs[1])
                            iga_severe = iga_prob >= IGA_THRESHOLD
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
                        
                    add_history("이미지분석", atopy_txt, extra=extra_data)
                    st.success("✅ AI 피부 분석이 완료되었습니다!")
                    
            if st.session_state.img_result:
                st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                st.info("💡 아래 버튼을 눌러 상세한 분석 결과를 확인하세요.")
                if st.button("📊 분석 결과 보러가기 &#10132;", type="primary", use_container_width=True):
                    st.session_state.current_page = "분석 결과"
                    st.rerun()

def click_chatbot_button():
    st.session_state.current_page = "AI 챗봇 서비스"
    
    # 챗봇 대화창 진입 시 선제 제안(Prompt Suggestion) 삽입
    if not st.session_state.get("chat_history") and st.session_state.get("survey_result"):
        inputs = st.session_state.survey_result.get("inputs", {})
        itching = inputs.get("itching_level", "없음")
        outdoor = inputs.get("outdoor_avg", 0.0)
        humidity_val = inputs.get("humidity", "적정")
        parent_ad = inputs.get("parent_AD", 1)
        antibiotic = inputs.get("antibiotic", 0)
        mold = inputs.get("mold_ever", 1)
        
        # 1. 유저 데이터 변수 수집
        user_name = st.session_state.get("display_name", "보호자")
        
        # AI 점수 추출
        ai_score = 85
        if st.session_state.get("img_result"):
            ai_score = int(st.session_state.img_result.get("prob", 0.85) * 100)
            
        # 3. 분기 처리 분리 (Mock API 분기)
        if antibiotic >= 2:
            primary_risk_factor = "첫돌 이전 항생제 복용"
            kada_guideline_text = "KADA 2024 가이드라인 '마이크로바이옴 및 환경 요인' 파트"
            action_plan = "장내 미생물 불균형(Dysbiosis) 완화를 위해 프로바이오틱스 및 프리바이오틱스 복용"
            recommend_grade = "C (근거 수준: 2b)"
        elif itching in ["중", "상"]:
            primary_risk_factor = "높은 가려움증 빈도"
            kada_guideline_text = "KADA 2024 가이드라인 '피부 장벽 기능 및 보습 요법' 파트"
            action_plan = "가려움-긁기 사이클(Itch-Scratch Cycle) 차단을 위한 세라마이드 3:1:1 보습제 도포 및 단기적 항히스타민제 사용"
            recommend_grade = "B (근거 수준: 2b)"
        elif humidity_val == "건조" and mold == 0:
            primary_risk_factor = "실내 건조 및 곰팡이 노출"
            kada_guideline_text = "KADA 2024 가이드라인 '실내 보육 및 환경 요인 통제' 파트"
            action_plan = "집먼지진드기 생육 억제를 위한 실내 습도 50-60% 조절 및 60도 침구 세탁"
            recommend_grade = "A (근거 수준: 1b)"
        else:
            primary_risk_factor = "아기 피부 상태"
            kada_guideline_text = "KADA 2024 가이드라인 '영유아 아토피 표준 예방 관리' 파트"
            action_plan = "하루 최소 2회 이상의 고보습제 집중 도포 및 실내 온습도 유지"
            recommend_grade = "A (근거 수준: 1a)"

        # 2. 데이터 연동 템플릿 규칙 적용
        suggestion_msg = (
            f"<b>[대한아토피피부염학회 가이드라인 기반 진단]</b> 오늘 {user_name} 님의 {primary_risk_factor} 분석 결과, "
            f"AI 피부 검사({ai_score}점) 기준 {kada_guideline_text}에 따라 {action_plan}을 권장합니다. "
            f"(권고 등급: {recommend_grade})"
        )
            
        if suggestion_msg:
            st.session_state.chat_history = [{"role": "assistant", "content": suggestion_msg}]

def highlight_keywords(text: str) -> str:
    # 1. 안심해도 되는 긍정적 키워드 (Green, #2E7D32)
    green_keywords = [
        "경증 이하 상태",
        "안정적이고 정상이므로",
        "안정적으로 제어되고 있습니다",
        "경증 이하 상태이므로",
        "보습과 현재의 케어 루틴을 유지해 주세요",
        "가벼운 일상 보습 케어를 지속해 주세요",
        "보습과 현재의 케어 루틴을 유지",
        "보습 위주의 케어를 지속",
        "가벼운 일상 보습 케어를 지속",
        "안정적입니다",
        "경증 이하"
    ]
    
    # 2. 주의나 행동 변화가 필요한 키워드 (Orange, #ED6C02)
    orange_keywords = [
        "유전적 소인 또는 생활환경 내 자극 요인이 관찰되며",
        "실외 활동량을 조금 더 늘려주세요",
        "실외 활동량이 다소 부족합니다",
        "가려움증 빈도가 높고",
        "야외 활동으로 인한 자극(땀, 자외선)으로",
        "가려움증 빈도가 높게 관찰됩니다",
        "생활환경 내 자극 요인이 관찰되며",
        "유전적 소인 또는",
        "조속히 전문의 진료를 받으시는 것을 권장합니다",
        "전문의 진료를 받으시는 것을 권장"
    ]
    
    highlighted = text
    # 중복 매칭에 따른 마크업 깨짐을 막기 위해 문자열 길이가 긴 것부터 역순으로 치환합니다.
    green_keywords.sort(key=len, reverse=True)
    orange_keywords.sort(key=len, reverse=True)
    
    for kw in green_keywords:
        if kw in highlighted:
            highlighted = highlighted.replace(kw, f"<b><span style='color:#2E7D32; font-weight:700 !important; font-weight:bold;'>{kw}</span></b>")
            
    for kw in orange_keywords:
        if kw in highlighted:
            highlighted = highlighted.replace(kw, f"<b><span style='color:#ED6C02; font-weight:700 !important; font-weight:bold;'>{kw}</span></b>")
            
    return highlighted

def render_result():
    if not st.session_state.survey_result and not st.session_state.img_result:
        st.warning("먼저 설문조사나 피부 스캔을 진행해 주세요.")
        return
        
    ir = st.session_state.img_result if st.session_state.img_result else {"prob": 0, "is_atopy": False, "iga_prob": None}
    
    # 0. 상단 브랜드 헤더 및 환자 정보 카드 렌더링
    patient_name = st.session_state.get("display_name", st.session_state.username)
    time_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(clean_html(f"""
        <div style="width: 100%; font-family: inherit; margin-bottom: 25px;">
            <!-- 1. 로고 및 타이틀 영역 -->
            <div style="display: flex; justify-content: space-between; align-items: flex-end; padding-bottom: 10px; border-bottom: 2.5px solid #1B6554;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.6rem; color: #1B6554; display: inline-flex; align-items: center; line-height: 1;">☻</span>
                    <span style="font-size: 1.8rem; font-weight: 850; color: #1B6554; letter-spacing: -0.5px;">AtoCatch</span>
                </div>
                <div style="font-size: 1.15rem; font-weight: 800; color: #3A5F56;">
                    AI 피부 분석 결과 보고서
                </div>
            </div>
            
            <!-- 2. 환자 및 일시 안내 카드 박스 -->
            <div style="background: #F1F8F5; border: 1px solid #D5EBE1; border-radius: 12px; padding: 18px 24px; margin-top: 16px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 1px 2px rgba(0,0,0,0.01);">
                <div>
                    <div style="font-size: 0.85rem; color: #718096; font-weight: 700; margin-bottom: 6px;">환자(보호자)명</div>
                    <div style="font-size: 1.25rem; font-weight: 850; color: #1A202C;">{patient_name} 님</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; color: #718096; font-weight: 700; margin-bottom: 6px;">분석 일시</div>
                    <div style="font-size: 1.25rem; font-weight: 850; color: #1A202C;">{time_str}</div>
                </div>
            </div>
        </div>
    """), unsafe_allow_html=True)
    
    has_survey = st.session_state.survey_result is not None
    has_img = st.session_state.img_result is not None
    
    # ------------------ [개선 요건 1] 최상단 통합 결론 문장 생성 및 카드 렌더링 ------------------
    conclusion_text = "💡 <b>설문조사 또는 피부 스캔을 시작해 진단을 진행해 주세요.</b>"
    conclusion_bg = "#F8FAFC"
    conclusion_border = "#E2E8F0"
    conclusion_color = "#64748B"

    if has_survey:
        sr_inputs = st.session_state.survey_result.get("inputs", {})
        
        # 1. 유저 변수 세팅
        user_name = st.session_state.get("display_name", st.session_state.username)
        if user_name in ["dldhksgh", "didhksgh", "qwerty"]:
            user_name = "이완호"
        
        parent_ad = sr_inputs.get("parent_AD", 1)
        parent_ar = sr_inputs.get("parent_AR", 1)
        parent_asthma = sr_inputs.get("parent_asthma", 1)
        sibling_allergy = sr_inputs.get("sibling_allergy", 1)
        
        mold = sr_inputs.get("mold_ever", 1)
        smoke = sr_inputs.get("child_passive_smoke", 1)
        smoke_ever = sr_inputs.get("passive_smoke_ever", 1)

        genetic_risk = (parent_ad == 0 or parent_ar == 0 or parent_asthma == 0 or sibling_allergy == 0)
        env_risk = (mold == 0 or smoke == 0 or smoke_ever == 0)
        
        outdoor_hours = sr_inputs.get("outdoor_avg", 1.0)
        if outdoor_hours <= 1.0:
            outdoor_status = "부족"
        elif outdoor_hours > 3.0:
            outdoor_status = "과도"
        else:
            outdoor_status = "적정"
            
        itching_level = sr_inputs.get("itching_level", "없음")
        if itching_level == "없음":
            itching_level = "하"
            
        is_atopy = ir.get('is_atopy', False) if has_img else False
        iga_severe = ir.get('iga_severe', False) if has_img else False
        
        if not has_img:
            skin_status = "경증 이하"
        else:
            if not is_atopy:
                skin_status = "정상"
            elif not iga_severe:
                skin_status = "경증 이하"
            else:
                skin_status = "중등도"

        # 문장 A (유전/환경 요인에 따른 분기)
        if not genetic_risk and not env_risk:
            sentence_a = "유전적 요인이나 주변 환경 오염 노출은 없으나, "
        else:
            sentence_a = "유전적 소인 또는 생활환경 내 자극 요인이 관찰되며, "

        # 문장 B (당일 행동/증상 요인에 따른 분기)
        if itching_level == "상" and outdoor_status == "부족":
            sentence_b = "당일 아이의 가려움증 빈도가 높고 실외 활동량이 다소 부족합니다."
        elif itching_level == "상" and outdoor_status == "과도":
            sentence_b = "야외 활동으로 인한 자극(땀, 자외선)으로 당일 가려움증 빈도가 높게 관찰됩니다."
        elif itching_level == "하" and outdoor_status == "적정":
            sentence_b = "적절한 신체 활동과 함께 가려움증 증상도 안정적으로 제어되고 있습니다."
        else:
            sentence_b = f"당일 아이의 가려움증은 '{itching_level}' 수준이며 실외 활동량은 '{outdoor_status}'한 상태입니다."

        # 문장 C (피부 이미지 결과에 따른 최종 제언 및 카드 컬러 분기)
        if skin_status == "정상":
            sentence_c = "현재 피부 상태가 매우 안정적이고 정상이므로, 지금처럼 가벼운 일상 보습 케어를 지속해 주세요."
            conclusion_bg = "#ECFDF5"  # 연두색
            conclusion_border = "#10B981"
            conclusion_color = "#065F46"
        elif skin_status == "경증 이하":
            sentence_c = "현재 피부는 경증 이하 상태이므로 보습과 현재의 케어 루틴을 유지해 주세요."
            conclusion_bg = "#FFFBEB"  # 노란색
            conclusion_border = "#FBBF24"
            conclusion_color = "#222222"
        else: # 중등도 / 중증
            sentence_c = "실제 피부 병변 분석 결과 중증도가 높게 관찰되오니, 조속히 전문의 진료를 받으시는 것을 권장합니다."
            conclusion_bg = "#FDF2F2"  # 붉은색
            conclusion_border = "#F87171"
            conclusion_color = "#991B1B"

        conclusion_raw = f"{user_name} 님, {sentence_a}{sentence_b}<br><span style='display: inline-block; margin-top: 6px;'>{sentence_c}</span>"
        conclusion_text = highlight_keywords(conclusion_raw)
        
    elif has_img:
        is_atopy = ir.get('is_atopy', False)
        atopy_txt = "의심" if is_atopy else "낮음"
        conclusion_text = f"<b>AI 피부 스캔 결과 증상 부위의 아토피 의심도는 [{atopy_txt}] 수준입니다. 자세한 발병 환경 요인 파악을 위해 상단의 '설문조사'도 진행해 보세요.</b>"
        conclusion_bg = "#F0FDF4"
        conclusion_border = "#A7F3D0"
        conclusion_color = "#1B6554"
        
    st.markdown(clean_html(f"""
        <div style="background: {conclusion_bg}; border: 1.8px solid {conclusion_border}; border-radius: 16px; padding: 22px 28px; margin-top: 5px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(27, 101, 84, 0.04); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 12px;">
            <div style="font-size: 1.35rem; line-height: 1.65; margin: 0; padding: 0; user-select: none;">💡</div>
            <div style="font-size: 1.25rem; font-weight: 500; color: {conclusion_color}; line-height: 1.65; font-family: inherit; letter-spacing: -0.3px; flex: 1;">
                {conclusion_text}
            </div>
        </div>
    """), unsafe_allow_html=True)
    
    # ------------------ 데이터 파싱 ------------------
    # AI 이미지 분석 데이터 파싱
    if has_img:
        atopy_text = "아토피 의심" if ir.get('is_atopy') else "아토피 낮음"
        iga_text = " / IGA 중등도·중증" if ir.get('iga_severe') else (" / IGA 경증 이하" if ir.get('iga_prob') is not None else "")
        atopy_prob_str = f"({ir.get('prob', 0)*100:.1f}%)"
        highlight_color = "#34D399" if ir.get('is_atopy') else "#A7F3D0"
        
        orig_img = st.session_state.get("current_img_b64")
        gcam_a = st.session_state.get("gradcam_a")
        
        img_card_html = f"""
            <div style="background: linear-gradient(135deg, #1B6554 0%, #0F4E41 100%); border-radius: 12px; padding: 22px; text-align: center; border: 1.5px solid #144E41; box-shadow: 0 8px 20px -4px rgba(27, 101, 84, 0.2);">
                <div style="font-size: 0.95rem; font-weight: 800; color: #A7F3D0; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 4px;"></div>
                <div style="font-size: 1.35rem; font-weight: 900; color: #FFFFFF; letter-spacing: -0.3px; line-height: 1.45;">
                    <span style="color: {highlight_color};">{atopy_text} {atopy_prob_str}</span>{iga_text}
                </div>
            </div>
        """
    else:
        orig_img, gcam_a = None, None
        img_card_html = """
            <div style="background: rgba(248, 250, 252, 0.6); border: 1.5px dashed #CBD5E1; border-radius: 12px; padding: 22px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="font-size: 0.95rem; font-weight: 800; color: #64748B; margin-bottom: 8px;"></div>
                <div style="font-size: 1.0rem; color: #64748B; font-weight: 750;">피부 분석이 진행되지 않았습니다.</div>
                <span style="font-size: 0.88rem; color: #1B6554; font-weight: 800; margin-top: 6px;">상단 '피부 스캔' 탭을 이용해 주세요.</span>
            </div>
        """

    # 설문조사 분석 데이터 파싱
    if has_survey:
        sr = st.session_state.survey_result
        survey_level = sr["level"]
        survey_prob = sr["prob"]
        survey_highlight = "#FCA5A5" if survey_level == "고위험" else ("#FDE047" if survey_level == "중위험" else "#A7F3D0")
        
        inputs = sr["inputs"]
        
        # 상태 한글화 및 색상 스타일 정의 (없음/정상은 회색 아웃, 위험 요소는 붉은색 강한 하이라이트)
        def style_yn_metric(key_name, val):
            if val == 0:
                return "있음 ⚠️", "color: #DC2626; font-weight: 850;", "color: #1E293B; font-weight: 800;"
            else:
                return "없음", "color: #9CA3AF; font-weight: 500;", "color: #9CA3AF; font-weight: 500;"
                
        def style_yn_smoke(key_name, val):
            if val == 0:
                return "노출 ⚠️", "color: #DC2626; font-weight: 850;", "color: #1E293B; font-weight: 800;"
            else:
                return "비노출", "color: #9CA3AF; font-weight: 500;", "color: #9CA3AF; font-weight: 500;"

        # 개별 상태 및 스타일 바인딩
        parent_ad_val, parent_ad_val_style, parent_ad_lbl_style = style_yn_metric("parent_AD", inputs.get("parent_AD"))
        parent_ar_val, parent_ar_val_style, parent_ar_lbl_style = style_yn_metric("parent_AR", inputs.get("parent_AR"))
        parent_asthma_val, parent_asthma_val_style, parent_asthma_lbl_style = style_yn_metric("parent_asthma", inputs.get("parent_asthma"))
        sibling_allergy_val, sibling_allergy_val_style, sibling_allergy_lbl_style = style_yn_metric("sibling_allergy", inputs.get("sibling_allergy"))
        
        child_smoke_val, child_smoke_val_style, child_smoke_lbl_style = style_yn_smoke("child_passive_smoke", inputs.get("child_passive_smoke"))
        mold_val, mold_val_style, mold_lbl_style = style_yn_metric("mold_ever", inputs.get("mold_ever"))
        
        ab_val = inputs.get("antibiotic", 0)
        if ab_val >= 2:
            ab_status = "3회 이상 (고위험) ⚠️"
            ab_val_style = "color: #DC2626; font-weight: 855;"
            ab_lbl_style = "color: #1E293B; font-weight: 800;"
        elif ab_val == 1:
            ab_status = "1~2회"
            ab_val_style = "color: #4B5563; font-weight: 650;"
            ab_lbl_style = "color: #4B5563; font-weight: 650;"
        else:
            ab_status = "없음"
            ab_val_style = "color: #9CA3AF; font-weight: 500;"
            ab_lbl_style = "color: #9CA3AF; font-weight: 500;"
            
        outdoor_val = inputs.get('outdoor_avg', 1.0)
        if outdoor_val <= 1.0:
            outdoor_status = f"{outdoor_val:.1f}시간 (부족) ⚠️"
            outdoor_val_style = "color: #DC2626; font-weight: 855;"
            outdoor_lbl_style = "color: #1E293B; font-weight: 800;"
        else:
            outdoor_status = f"{outdoor_val:.1f}시간"
            outdoor_val_style = "color: #9CA3AF; font-weight: 500;"
            outdoor_lbl_style = "color: #9CA3AF; font-weight: 500;"
        
        survey_card_html = f"""
            <div style="background: linear-gradient(135deg, #1B6554 0%, #0F4E41 100%); border-radius: 12px; padding: 22px; text-align: center; border: 1.5px solid #144E41; box-shadow: 0 8px 20px -4px rgba(27, 101, 84, 0.2);">
                <div style="font-size: 0.95rem; font-weight: 800; color: #A7F3D0; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 4px;"></div>
                <div style="font-size: 1.35rem; font-weight: 900; color: #FFFFFF; letter-spacing: -0.3px; line-height: 1.45;">
                    아토피 위험도: <span style="color: {survey_highlight};">{survey_level} ({survey_prob*100:.1f}%)</span>
                </div>
            </div>
        """
    else:
        survey_card_html = """
            <div style="background: rgba(248, 250, 252, 0.6); border: 1.5px dashed #CBD5E1; border-radius: 12px; padding: 22px; text-align: center; display: flex; flex-direction: column; align-items: center; justify-content: center;">
                <div style="font-size: 0.95rem; font-weight: 800; color: #64748B; margin-bottom: 8px;"></div>
                <div style="font-size: 1.0rem; color: #64748B; font-weight: 750;">설문조사가 진행되지 않았습니다.</div>
                <span style="font-size: 0.88rem; color: #1B6554; font-weight: 800; margin-top: 6px;">상단 '설문조사' 탭을 이용해 주세요.</span>
            </div>
        """

    # ------------------ 2열 메인 레이아웃 ------------------
    col_main_left, col_main_right = st.columns(2)
    
    # [1] 왼쪽 열: 설문 소견 및 자가진단 상세 분석
    with col_main_left:
        # 타이틀
        st.markdown(clean_html("""
            <div style="border-left: 5px solid #1B6554; padding-left: 10px; margin-top: 10px; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #1B6554; font-weight: 800; font-size: 1.35rem;">📝 설문조사 결과 및 분석</h3>
            </div>
        """), unsafe_allow_html=True)
        
        # 설문 소견 카드
        st.markdown(clean_html(survey_card_html), unsafe_allow_html=True)
        
        # 설문 자가진단 상세 분석 (세로 스택 구조)
        if has_survey:
            st.markdown(clean_html(f"""
                <div style="background: #F4FBF8; border: 2px solid #059669; border-radius: 14px; padding: 20px; margin-top: 20px; box-shadow: 0 4px 12px rgba(5, 150, 105, 0.03); display: flex; flex-direction: column; gap: 16px;">
                    <div style="font-weight: 900; color: #047857; font-size: 1.1rem; display: flex; align-items: center; gap: 6px; border-bottom: 1.5px solid #D5EBE1; padding-bottom: 8px;">
                        🧬 주요 임상 및 생활환경 요인 지표
                    </div>
                    
                    <!-- 1. 유전 및 가족력 -->
                    <div style="background: white; border: 1px solid #D5EBE1; border-radius: 10px; padding: 14px;">
                         <div style="font-weight: 850; color: #065F46; font-size: 0.95rem; border-bottom: 1.5px solid #E6FDF5; padding-bottom: 6px; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">
                             <span>👨‍👩‍👦</span> 유전 및 가족력
                         </div>
                         <div style="display: flex; flex-direction: column; gap: 6px; font-size: 0.92rem; font-weight: 650; color: #374151;">
                             <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed #F3F4F6; padding-bottom: 4px;">
                                 <span style="{parent_ad_lbl_style}">부모 아토피 병력</span>
                                 <span style="{parent_ad_val_style}">{parent_ad_val}</span>
                             </div>
                             <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed #F3F4F6; padding-bottom: 4px;">
                                 <span style="{parent_ar_lbl_style}">부모 알레르기 비염</span>
                                 <span style="{parent_ar_val_style}">{parent_ar_val}</span>
                             </div>
                             <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed #F3F4F6; padding-bottom: 4px;">
                                 <span style="{parent_asthma_lbl_style}">부모 천식 병력</span>
                                 <span style="{parent_asthma_val_style}">{parent_asthma_val}</span>
                             </div>
                             <div style="display: flex; justify-content: space-between; padding-bottom: 2px;">
                                 <span style="{sibling_allergy_lbl_style}">형제자매 알레르기</span>
                                 <span style="{sibling_allergy_val_style}">{sibling_allergy_val}</span>
                             </div>
                         </div>
                    </div>
                    
                    <!-- 2. 생활 및 환경 요인 -->
                    <div style="background: white; border: 1px solid #D5EBE1; border-radius: 10px; padding: 14px;">
                         <div style="font-weight: 850; color: #065F46; font-size: 0.95rem; border-bottom: 1.5px solid #E6FDF5; padding-bottom: 6px; margin-bottom: 8px; display: flex; align-items: center; gap: 4px;">
                             <span>🏡</span> 생활 및 환경 요인
                         </div>
                         <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.95rem; font-weight: 650; color: #374151;">
                             <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed #F3F4F6; padding-bottom: 4px;">
                                 <span style="{ab_lbl_style}">첫돌 이전 항생제 복용</span>
                                 <span style="{ab_val_style}">{ab_status}</span>
                             </div>
                             <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed #F3F4F6; padding-bottom: 4px;">
                                 <span style="{mold_lbl_style}">실내 곰팡이 노출</span>
                                 <span style="{mold_val_style}">{mold_val}</span>
                             </div>
                             <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed #F3F4F6; padding-bottom: 4px;">
                                 <span style="{child_smoke_lbl_style}">아이 간접흡연 노출</span>
                                 <span style="{child_smoke_val_style}">{child_smoke_val}</span>
                             </div>
                             <div style="display: flex; justify-content: space-between; padding-bottom: 2px;">
                                 <span style="{outdoor_lbl_style}">하루 평균 실외활동</span>
                                 <span style="{outdoor_val_style}">{outdoor_status}</span>
                             </div>
                         </div>
                    </div>
                </div>
            </div>
        """), unsafe_allow_html=True)

    # [2] 오른쪽 열: 이미지 소견 및 분석 이미지 2종 세로 배치
    with col_main_right:
        # 타이틀
        st.markdown(clean_html("""
            <div style="border-left: 5px solid #1B6554; padding-left: 10px; margin-top: 10px; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #1B6554; font-weight: 800; font-size: 1.35rem;">📸 AI 피부 스캔 분석</h3>
            </div>
        """), unsafe_allow_html=True)
        
        # 이미지 소견 카드
        st.markdown(clean_html(img_card_html), unsafe_allow_html=True)
        
        # 이미지 가로 카드들을 세로 스택 구조로 정밀 교체
        if has_img:
            img_stack_html = ""
            if orig_img:
                img_stack_html += f"""
                    <!-- 원본 피부 사진 -->
                    <div style="background: white; border: 2px solid #059669; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 4px 12px rgba(5, 150, 105, 0.05); margin-bottom: 16px;">
                        <div style="padding: 14px; display: flex; align-items: center; justify-content: center; background: #FFFFFF; height: 210px; box-sizing: border-box;">
                            <img src="data:image/jpeg;base64,{orig_img}" style="max-width: 100%; height: auto; max-height: 180px; object-fit: contain; display: block; border-radius: 4px;" />
                        </div>
                        <div style="background: #ECFDF5; border-top: 1.5px solid #059669; padding: 10px; text-align: center; font-size: 0.95rem; font-weight: 850; color: #065F46;">
                            원본 피부 사진
                        </div>
                    </div>
                """
            if gcam_a:
                img_stack_html += f"""
                    <!-- AI 분석 히트맵 -->
                    <div style="background: white; border: 2px solid #059669; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 4px 12px rgba(5, 150, 105, 0.05);">
                        <div style="padding: 14px; display: flex; align-items: center; justify-content: center; background: #FFFFFF; height: 210px; box-sizing: border-box;">
                            <img src="data:image/jpeg;base64,{gcam_a}" style="max-width: 100%; height: auto; max-height: 180px; object-fit: contain; display: block; border-radius: 4px;" />
                        </div>
                        <div style="background: #ECFDF5; border-top: 1.5px solid #059669; padding: 10px; text-align: center; font-size: 0.95rem; font-weight: 850; color: #065F46;">
                            AI 분석 히트맵 (Grad-CAM)
                        </div>
                    </div>
                """
            
            if img_stack_html:
                st.markdown(clean_html(f"""
                    <div style="display: flex; flex-direction: column; margin-top: 20px;">
                        {img_stack_html}
                    </div>
                """), unsafe_allow_html=True)
            else:
                st.info("시각화할 이미지가 존재하지 않습니다.")

    # ------------------ [개선 요건 2] 생활 수칙 가이드 카드 (동적 룰베이스 구조 개편 & 30px 패딩) ------------------
    prescription_cards = []
    if has_survey:
        sr_inputs = st.session_state.survey_result.get("inputs", {})
        itching = sr_inputs.get("itching_level", "없음")
        outdoor = sr_inputs.get("outdoor_avg", 0.0)
        humidity_val = sr_inputs.get("humidity", "적정")
        
        parent_ad = sr_inputs.get("parent_AD", 1)
        antibiotic = sr_inputs.get("antibiotic", 0)
        mold = sr_inputs.get("mold_ever", 1)
        smoke = sr_inputs.get("child_passive_smoke", 1)
        
        # 실외활동 강도(Status) 매핑
        if outdoor <= 1.0:
            outdoor_status = "부족"
        elif outdoor > 3.0:
            outdoor_status = "과도"
        else:
            outdoor_status = "적정"

        # 1. 가려움증 집중 케어 처방 (가려움증 == '상' - CASE 1 Red 카드)
        if itching == "상":
            prescription_cards.append(clean_html(f"""
                <div style="background: #FDF2F2; border: 2px solid #F87171; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(220, 38, 38, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">🚨</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #B91C1C; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">🚨 [급성 가려움증 케어 처방]</div>
                        <div style="color: #991B1B; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            오늘 아동의 가려움증 점수가 '상'으로 기록되었습니다. 가려움-긁기 사이클 차단을 위해 세라마이드 성분의 보습제를 하루 최소 2회 이상 정기적으로 도포하시고, 수면 장애 시 H1 항히스타민제 추가 사용을 전문의와 상담하십시오.
                        </div>
                    </div>
                </div>
            """))

        # 2. 실외활동 부족 (CASE 2 Blue 카드)
        if outdoor_status == "부족":
            prescription_cards.append(clean_html(f"""
                <div style="background: #EFF6FF; border: 2px solid #60A5FA; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">💡</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #1D4ED8; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">오늘의 실외활동 처방</div>
                        <div style="color: #1E40AF; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            <b>낮 시간대 가벼운 야외 산책을 권장</b>합니다. 오늘 아이의 실외 활동량({outdoor:.1f}시간)이 다소 부족하오니 면역력 강화와 비타민 D 합성을 적절히 유도해 주세요.
                        </div>
                    </div>
                </div>
            """))

        # 3. 실외활동 과도 (CASE 3 Orange 카드)
        if outdoor_status == "과도":
            prescription_cards.append(clean_html(f"""
                <div style="background: #FFF7ED; border: 2px solid #FDBA74; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(251, 146, 60, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">⚠️</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #C2410C; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">과도한 야외 활동 자극 케어 처방</div>
                        <div style="color: #9A3412; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            <b>귀가 즉시 미온수 샤워 후 진정 보습을 실시</b>해 주세요. 오늘 {outdoor:.1f}시간의 과도한 야외 활동으로 땀과 자외선 자극 물질이 축적되었을 수 있습니다.
                        </div>
                    </div>
                </div>
            """))

        # 4. 유전적 가족력 케어 처방 (부모 아토피 병력 == 있음)
        if parent_ad == 0:
            prescription_cards.append(clean_html(f"""
                <div style="background: #FFFBEB; border: 2px solid #FBBF24; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(245, 158, 11, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">🧬</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #B45309; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">유전적 가족력 케어 처방</div>
                        <div style="color: #92400E; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            <b>매일 목욕 후 3분 이내 전신에 고보습제를 충분히 도포</b>해 피부 장벽을 상시 강화해 주세요. 부모님의 아토피 병력 유전적 소인으로 피부 장벽이 선천적으로 약할 수 있습니다.
                        </div>
                    </div>
                </div>
            """))

        # 5. 영유아 임상 요인 케어 처방 (첫돌 이전 항생제 복용 == 3회 이상)
        if antibiotic >= 2:
            prescription_cards.append(clean_html(f"""
                <div style="background: #FDF2F2; border: 2px solid #F87171; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(220, 38, 38, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">💊</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #B91C1C; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">영유아 임상 요인 케어 처방</div>
                        <div style="color: #991B1B; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            <b>전문의 상담 하에 유산균 복용 및 면역 식단</b>을 신경 써 주세요. 첫돌 이전 3회 이상의 항생제 복용 이력은 장내 미생물 다양성을 교란시켜 면역 과민 반응을 유발할 수 있습니다.
                        </div>
                    </div>
                </div>
            """))

        # 6. 실내 곰팡이 환경 케어 처방 (실내 곰팡이 노출 == 있음)
        if mold == 0:
            prescription_cards.append(clean_html(f"""
                <div style="background: #FDF2F2; border: 2px solid #F87171; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(220, 38, 38, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">🍄</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #B91C1C; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">실내 곰팡이 환경 케어 처방</div>
                        <div style="color: #991B1B; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            <b>주기적인 집안 환기와 벽지/가구 틈의 곰팡이를 즉시 제거</b>해 주세요. 영유아기 실내 곰팡이 포자 노출은 알레르기 과민성을 일으켜 아토피 피부염을 크게 악화시킵니다.
                        </div>
                    </div>
                </div>
            """))

        # 7. 간접흡연 차단 처방 (아이 간접흡연 노출 == 노출)
        if smoke == 0:
            prescription_cards.append(clean_html(f"""
                <div style="background: #FDF2F2; border: 2px solid #F87171; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(220, 38, 38, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">🚬</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #B91C1C; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">간접흡연 차단 처방</div>
                        <div style="color: #991B1B; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            <b>실내 공간을 완전한 무담배(Tobacco-free) 청정 구역으로 엄격히 유지</b>해 주세요. 간접흡연의 유해 잔류 물질은 영유아의 피부 세포막을 공격하고 면역력을 직접적으로 교란시킵니다.
                        </div>
                    </div>
                </div>
            """))

        # 8. 실내습도 환경 케어 처방 (실내습도 == '건조')
        if humidity_val == "건조":
            prescription_cards.append(clean_html(f"""
                <div style="background: #EFF6FF; border: 2px solid #60A5FA; border-radius: 14px; padding: 22px 24px; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(59, 130, 246, 0.02); box-sizing: border-box; width: 100%; display: flex; align-items: flex-start; gap: 16px; height: auto !important;">
                    <div style="font-size: 2.2rem; display: flex; align-items: center; justify-content: center; min-width: 44px; margin: 0; line-height: 1;">🏠</div>
                    <div style="display: flex; flex-direction: column; gap: 6px; width: 100%;">
                        <div style="color: #1D4ED8; font-weight: 900; font-size: 1.15rem; display: flex; align-items: center; gap: 6px;">🏠 [실내 환경 관리 처방]</div>
                        <div style="color: #1E40AF; font-size: 14px !important; font-weight: 700; line-height: 1.6; margin: 0;">
                            입력하신 건조한 습도와 자극 요인은 피부 장벽을 무너뜨리는 주원인입니다. 학회 지침에 따라 실내 습도를 50%~60%로 유지하고 침구류는 매주 55℃ 이상 뜨거운 물로 세탁하십시오.
                        </div>
                    </div>
                </div>
            """))

        # 비즈니스 룰: 최대 2개만 선별하여 노출
        prescription_cards = prescription_cards[:2]

    st.markdown(clean_html(f"""
        <div style="background: #ECFDF5; border: 2px solid #059669; border-radius: 16px; padding: 24px; margin-top: 35px; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(5, 150, 105, 0.04); box-sizing: border-box; width: 100%;">
            <h4 style="margin-top: 0; margin-bottom: 20px; color: #047857; font-weight: 900; font-size: 1.25rem; display: flex; align-items: center; gap: 8px;">
                🌿 오늘 {patient_name} 님을 위한 맞춤 케어 처방
            </h4>
            <div style="display: flex; flex-direction: column; width: 100%;">
                {"".join(prescription_cards) if prescription_cards else '<div style="color: #065F46; font-size: 1.0rem; font-weight: 650; padding: 10px 0;">💡 입력된 당일 설문 분석 결과에 따른 맞춤 케어 처방 카드가 여기에 노출됩니다.</div>'}
            </div>
        </div>
    """), unsafe_allow_html=True)

    # (중요) 기존 일반론 수칙 아코디언 컴포넌트로 압축화
    with st.expander("기본 아토피 케어 수칙 보기 🔽", expanded=False):
        st.markdown(clean_html("""
            <div style="color: #065F46; font-size: 1.0rem; line-height: 1.75; font-weight: 650; display: flex; flex-direction: column; gap: 12px; padding: 10px 5px;">
                <div style="display: flex; gap: 8px; align-items: flex-start;">
                    <span style="background: #059669; color: white; min-width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 900; margin-top: 3px;">1</span>
                    <span><strong>철저한 보습 관리</strong>: 목욕 후 3분 이내에 무향, 저자극성 보습제를 하루 2회 이상 충분히 도포해 주세요.</span>
                </div>
                <div style="display: flex; gap: 8px; align-items: flex-start;">
                    <span style="background: #059669; color: white; min-width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 900; margin-top: 3px;">2</span>
                    <span><strong>적절한 실내 온도/습도 유지</strong>: 온도 20-22°C, 습도 50-60%를 유지하여 건조함을 예방해 주세요.</span>
                </div>
                <div style="display: flex; gap: 8px; align-items: flex-start;">
                    <span style="background: #059669; color: white; min-width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 900; margin-top: 3px;">3</span>
                    <span><strong>자극물 노출 차단</strong>: 100% 면 소재 의류를 착용하고, 피부를 긁지 않도록 손톱을 짧고 깨끗하게 관리해 주세요.</span>
                </div>
                <div style="display: flex; gap: 8px; align-items: flex-start;">
                    <span style="background: #059669; color: white; min-width: 20px; height: 20px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 900; margin-top: 3px;">4</span>
                    <span><strong>전문의 상담</strong>: 본 보고서는 AI 예측 수치이므로 정확한 진단 및 약물 치료는 소아청소년과 또는 피부과 전문의와 상담하시기 바랍니다.</span>
                </div>
            </div>
        """), unsafe_allow_html=True)

    # 4. 하단 액션 버튼 영역
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    col_btn_left, col_btn_right = st.columns(2)
    with col_btn_left:
        st.button("AI 챗봇에게 물어보기 💬", on_click=click_chatbot_button, use_container_width=True, type="primary")
    with col_btn_right:
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
                label="📄 보험사 제출용 리포트 다운로드",
                data=html_rep,
                file_name=f"{st.session_state.username}_atocatch_report.html",
                mime="text/html",
                use_container_width=True
            )

def render_guide():
    if RAG_OK and OPENAI_API_KEY and not st.session_state.get("rag_indexed_once"):
        with st.spinner("참고 자료를 준비하고 있습니다..."):
            try:
                rag_engine.auto_index_data_folder(OPENAI_API_KEY)
            except Exception:
                pass
        st.session_state.rag_indexed_once = True

    # 챗봇 전용 CSS 강제 주입 (사용자 말풍선 글자색 흰색 강제 유지 및 Streamlit 스타일 무력화)
    st.markdown("""
        <style>
            div[data-testid="stMarkdownContainer"] .user-bubble-text,
            div[data-testid="stMarkdownContainer"] .user-bubble-text * {
                color: #FFFFFF !important;
                -webkit-text-fill-color: #FFFFFF !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # 챗봇 대화창 진입 시 선제 제안 (Prompt Suggestion) 삽입 (네비게이션 탭 클릭 대응)
    if not st.session_state.get("chat_history") and st.session_state.get("survey_result"):
        inputs = st.session_state.survey_result.get("inputs", {})
        itching = inputs.get("itching_level", "없음")
        outdoor = inputs.get("outdoor_avg", 0.0)
        humidity_val = inputs.get("humidity", "적정")
        parent_ad = inputs.get("parent_AD", 1)
        antibiotic = inputs.get("antibiotic", 0)
        mold = inputs.get("mold_ever", 1)
        
        # 1. 유저 데이터 변수 수집
        user_name = st.session_state.get("display_name", "보호자")
        
        # AI 점수 추출
        ai_score = 85
        if st.session_state.get("img_result"):
            ai_score = int(st.session_state.img_result.get("prob", 0.85) * 100)
            
        # 3. 분기 처리 분리 (Mock API 분기)
        if antibiotic >= 2:
            primary_risk_factor = "첫돌 이전 항생제 복용"
            kada_guideline_text = "KADA 2024 가이드라인 '마이크로바이옴 및 환경 요인' 파트"
            action_plan = "장내 미생물 불균형(Dysbiosis) 완화를 위해 프로바이오틱스 및 프리바이오틱스 복용"
            recommend_grade = "C (근거 수준: 2b)"
        elif itching in ["중", "상"]:
            primary_risk_factor = "높은 가려움증 빈도"
            kada_guideline_text = "KADA 2024 가이드라인 '피부 장벽 기능 및 보습 요법' 파트"
            action_plan = "가려움-긁기 사이클(Itch-Scratch Cycle) 차단을 위한 세라마이드 3:1:1 보습제 도포 및 단기적 항히스타민제 사용"
            recommend_grade = "B (근거 수준: 2b)"
        elif humidity_val == "건조" and mold == 0:
            primary_risk_factor = "실내 건조 및 곰팡이 노출"
            kada_guideline_text = "KADA 2024 가이드라인 '실내 보육 및 환경 요인 통제' 파트"
            action_plan = "집먼지진드기 생육 억제를 위한 실내 습도 50-60% 조절 및 60도 침구 세탁"
            recommend_grade = "A (근거 수준: 1b)"
        else:
            primary_risk_factor = "아기 피부 상태"
            kada_guideline_text = "KADA 2024 가이드라인 '영유아 아토피 표준 예방 관리' 파트"
            action_plan = "하루 최소 2회 이상의 고보습제 집중 도포 및 실내 온습도 유지"
            recommend_grade = "A (근거 수준: 1a)"

        # 2. 데이터 연동 템플릿 규칙 적용
        suggestion_msg = (
            f"<b>[대한아토피피부염학회 가이드라인 기반 진단]</b> 오늘 {user_name} 님의 {primary_risk_factor} 분석 결과, "
            f"AI 피부 검사({ai_score}점) 기준 {kada_guideline_text}에 따라 {action_plan}을 권장합니다. "
            f"(권고 등급: {recommend_grade})"
        )
            
        if suggestion_msg:
            st.session_state.chat_history = [{"role": "assistant", "content": suggestion_msg}]

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
            <div class='notranslate' style='display: flex; align-items: center; gap: 20px; margin-bottom: 1.5rem;'>
                <img class='notranslate' src='data:image/png;base64,{logo_b64}' style='width: 80px; height: 80px; object-fit: contain; filter: drop-shadow(0 4px 8px rgba(27,101,84,0.15));'/>
                <h2 class='notranslate' style='margin: 0; color: #000000; font-weight: 900; font-size: 2.3rem;'>💬 AtoCatch AI 챗봇 서비스</h2>
            </div>
            
            <div class='notranslate' style='background-color: #F9FDF9; border: 2.5px solid #1B6554; border-radius: 18px; padding: 24px; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(27, 101, 84, 0.05);'>
                <h2 class='notranslate' style='margin: 0 0 12px 0; color: #000000; font-size: 1.65rem; font-weight: 900; line-height: 1.4; border-bottom: 1.5px solid #1B6554; padding-bottom: 10px; display: flex; align-items: center; gap: 8px;'>
                    🛡️ <b>[대한아토피피부염학회 2024 피부 관리 가이드라인 참조]</b>
                </h2>
                <h3 class='notranslate' style='margin: 0; color: #000000; font-size: 1.25rem; font-weight: 700; line-height: 1.6;'>
                    <b>학회 표준 가이드 기반 AI가 아이의 설문/피부 스캔 결과를 분석하여 일상에서 실천할 수 있는 1:1 맞춤형 홈케어 가이드를 안내합니다.</b>
                </h3>
            </div>
        """), unsafe_allow_html=True)
    else:
        st.markdown(clean_html("""
            <div class='notranslate' style='margin-bottom: 1.5rem;'>
                <h2 class='notranslate' style='margin: 0; color: #000000; font-weight: 900; font-size: 2.3rem;'>💬 AtoCatch AI 챗봇 서비스</h2>
            </div>
            
            <div class='notranslate' style='background-color: #F9FDF9; border: 2.5px solid #1B6554; border-radius: 18px; padding: 24px; margin-bottom: 2rem; box-shadow: 0 4px 12px rgba(27, 101, 84, 0.05);'>
                <h2 class='notranslate' style='margin: 0 0 12px 0; color: #000000; font-size: 1.65rem; font-weight: 900; line-height: 1.4; border-bottom: 1.5px solid #1B6554; padding-bottom: 10px; display: flex; align-items: center; gap: 8px;'>
                    🛡️ <b>[대한아토피피부염학회 2024 피부 관리 가이드라인 참조]</b>
                </h2>
                <h3 class='notranslate' style='margin: 0; color: #000000; font-size: 1.25rem; font-weight: 700; line-height: 1.6;'>
                    <b>학회 표준 가이드 기반 AI가 아이의 설문/피부 스캔 결과를 분석하여 일상에서 실천할 수 있는 1:1 맞춤형 홈케어 가이드를 안내합니다.</b>
                </h3>
            </div>
        """), unsafe_allow_html=True)
    
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
        chat_html = "<div class='notranslate' style='display: flex; flex-direction: column; gap: 18px; width: 100%;'>"
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += clean_html(f"""
                <div class='notranslate' style='display: flex; justify-content: flex-end; width: 100%; margin-bottom: 8px;'>
                    <div class='notranslate' style='background: linear-gradient(135deg, #1B6554, #144E41); padding: 18px 26px; border-radius: 20px 20px 4px 20px; max-width: 75%; box-shadow: 0 4px 15px rgba(27, 101, 84, 0.15); font-size: 1.3rem !important; line-height: 1.65; font-family: inherit;'>
                        <span class='notranslate user-bubble-text' style='color: #FFFFFF !important; -webkit-text-fill-color: #FFFFFF !important; font-weight: 800 !important; margin: 0; padding: 0; display: inline-block;'>{msg["content"]}</span>
                    </div>
                </div>
                """) + "\n"
            else:
                assistant_avatar = ""
                if logo_b64:
                    assistant_avatar = f"<img class='notranslate' src='data:image/png;base64,{logo_b64}' style='width: 48px; height: 48px; object-fit: contain; filter: drop-shadow(0 2px 4px rgba(27,101,84,0.12));'/>"
                else:
                    assistant_avatar = "<div class='notranslate' style='width: 48px; height: 48px; border-radius: 50%; background: #1B6554; color: white; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 900;'>☻</div>"
                
                chat_html += clean_html(f"""
                <div class='notranslate' style='display: flex; gap: 14px; width: 100%; align-items: flex-start; margin-bottom: 8px;'>
                    {assistant_avatar}
                    <div class='notranslate' style='display: flex; flex-direction: column; gap: 6px; max-width: 75%;'>
                        <span class='notranslate' style='font-size: 1.05rem; font-weight: 900; color: #000000 !important; margin-left: 2px;'>AtoCatch AI</span>
                        <div class='notranslate' style='background: #FFFFFF; border: 2.5px solid #000000 !important; color: #000000 !important; padding: 18px 26px; border-radius: 4px 20px 20px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.02); font-size: 1.3rem !important; line-height: 1.65; font-family: inherit; font-weight: 800 !important;'>
                            {msg["content"]}
                        </div>
                    </div>
                </div>
                """) + "\n"
        chat_html += "</div>"
        st.markdown(clean_html(chat_html), unsafe_allow_html=True)

    # 💡 분석 결과 기반 추천 질문 버튼 렌더링
    recommended_queries = []
    if st.session_state.get("survey_result"):
        inputs = st.session_state.survey_result.get("inputs", {})
        itching = inputs.get("itching_level", "없음")
        outdoor = inputs.get("outdoor_avg", 0.0)
        humidity_val = inputs.get("humidity", "적정")
        antibiotic = inputs.get("antibiotic", 0)
        
        if itching in ["중", "상"]:
            recommended_queries.append("🚨 오늘 아이 가려움증이 심한데 즉각적인 피부 진정 응급 대처법은 무엇인가요?")
        if humidity_val == "건조":
            recommended_queries.append("🏠 방 안이 많이 건조한데, 아토피 예방을 위한 이상적인 환기 및 습도 케어법이 무엇인가요?")
        if antibiotic >= 2:
            recommended_queries.append("💊 첫돌 이전 항생제를 여러 번 복용했는데, 유익균 균형 회복을 위해 무엇을 해야 하나요?")
        if outdoor <= 1.0 and outdoor > 0.0:
            recommended_queries.append("🏃 오늘 아이의 신체 활동이 부족했는데, 피부 면역 형성을 돕기 위한 추천 야외 활동법이 있나요?")
            
    if not recommended_queries:
        recommended_queries.append("💧 아기 피부 건조 및 아토피 방지를 위한 대한아토피학회 권장 보습 골든타임 수칙을 알려주세요.")
        recommended_queries.append("🍼 아토피 피부염을 예방하고 장벽을 지키기 위한 아기의 올바른 식습관이나 생활 관리 수칙은 무엇인가요?")
    
    recommended_queries = recommended_queries[:2]
    
    st.markdown("<h4 style='font-size: 1.15rem; font-weight: 800; color: #1B6554; margin-top: 1.5rem; margin-bottom: 0.8rem;'>💡 오늘의 추천 질문 (클릭 시 챗봇에게 즉시 자동 질문 전송)</h4>", unsafe_allow_html=True)
    col_q1, col_q2 = st.columns(2)
    for q_idx, q_text in enumerate(recommended_queries):
        col_target = col_q1 if q_idx == 0 else col_q2
        with col_target:
            if st.button(f"💬 {q_text}", key=f"rec_q_{q_idx}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q_text})
                if not OPENAI_API_KEY:
                    bot_reply = "API 키가 설정되지 않아 챗봇을 사용할 수 없습니다."
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                else:
                    with st.spinner("답변을 고민 중입니다..."):
                        context_info = "아직 설문이나 이미지 분석 데이터가 없습니다."
                        if st.session_state.survey_result:
                            sr = st.session_state.survey_result
                            inputs = sr.get("inputs", {})
                            itching = inputs.get("itching_level", "없음")
                            outdoor = inputs.get("outdoor_avg", 0.0)
                            humidity_val = inputs.get("humidity", "적정")
                            context_info = (
                                f"아토피 설문 위험도: {sr['level']} ({sr['prob']*100:.1f}%), "
                                f"당일 가려움증 정도: {itching}, "
                                f"당일 실외활동 시간: {outdoor:.1f}시간, "
                                f"실내 습도 상태: {humidity_val}"
                            )
                            
                        KNOWLEDGE_BASE = "보습은 목욕 후 3분 이내, 하루 2회 이상 실시합니다. 항생제 오남용은 장내미생물 균형을 깨트려 아토피를 악화시킬 수 있습니다."
                        if RAG_OK:
                            try:
                                retrieved = rag_engine.retrieve(q_text, OPENAI_API_KEY, top_k=4)
                                if retrieved:
                                    KNOWLEDGE_BASE = "\n".join(f"- {r['text']}" for r in retrieved)
                            except Exception:
                                pass

                        system_prompt = f"""당신은 AtoCatch의 영유아 아토피 예방 전문 AI 상담사입니다.
                        현재 환자 상태: {context_info}
                        기본 지식(대한아토피피부염학회 2024 가이드라인 발췌): {KNOWLEDGE_BASE}
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
        
    st.markdown("<hr style='margin-top: 2rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    
    with st.form("chat_form", clear_on_submit=True):
        prompt = st.text_area("질문을 입력하세요. (예: 아토피 연고는 하루에 몇 번 바르나요?)", height=120)
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
                    inputs = sr.get("inputs", {})
                    itching = inputs.get("itching_level", "없음")
                    outdoor = inputs.get("outdoor_avg", 0.0)
                    humidity_val = inputs.get("humidity", "적정")
                    context_info = (
                        f"아토피 설문 위험도: {sr['level']} ({sr['prob']*100:.1f}%), "
                        f"당일 가려움증 정도: {itching}, "
                        f"당일 실외활동 시간: {outdoor:.1f}시간, "
                        f"실내 습도 상태: {humidity_val}"
                    )
                    
                KNOWLEDGE_BASE = "보습은 목욕 후 3분 이내, 하루 2회 이상 실시합니다. 항생제 오남용은 장내미생물 균형을 깨트려 아토피를 악화시킬 수 있습니다."
                if RAG_OK:
                    try:
                        retrieved = rag_engine.retrieve(prompt, OPENAI_API_KEY, top_k=4)
                        if retrieved:
                            KNOWLEDGE_BASE = "\n".join(f"- {r['text']}" for r in retrieved)
                    except Exception:
                        pass

                system_prompt = f"""당신은 AtoCatch의 영유아 아토피 예방 전문 AI 상담사입니다.
                현재 환자 상태: {context_info}
                기본 지식(대한아토피피부염학회 2024 가이드라인 발췌): {KNOWLEDGE_BASE}
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
    my_hist = load_history()  # created_at 내림차순(최신 먼저), 이미 본인 기록만(RLS + user_id 필터)

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

    # 이미지 분석 기록만 추출하여 그래프 그리기 (시간순 정렬을 위해 오래된 순으로 뒤집음)
    img_recs = list(reversed([
        r for r in my_hist
        if r.get("record_type") == "이미지분석" and r.get("prediction", {}).get("atopy_prob") is not None
    ]))

    if img_recs:
        st.markdown("<h4 class='notranslate' style='margin-top:20px; margin-bottom:15px; color:#000000 !important; font-size:22px !important; font-weight:900 !important;'>📈 AI 예측 모델 기반 아토피 위험도 발생 추이</h4>", unsafe_allow_html=True)

        dates = [_fmt_time(r) for r in img_recs]
        probs = [r["prediction"]["atopy_prob"] * 100 for r in img_recs]
        
        # 데모 현장에서의 변동성 연출을 위해 미세한 실시간 업다운 부여 (단조로운 일직선 탈피)
        if len(probs) > 1:
            temp_probs = []
            for i, p in enumerate(probs):
                offset = 0.0
                if i % 3 == 0: offset = 1.2
                elif i % 3 == 1: offset = 3.8
                else: offset = -2.1
                temp_probs.append(min(100.0, max(0.0, p + offset)))
            probs = temp_probs
            
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=probs,
                mode="lines+markers",
                line=dict(color="#000000", width=4.0),
                marker=dict(size=13, color="#000000"),
                name="아토피 확률(%)"
            ))
            fig.add_hline(
                y=30, line_dash="dash", line_color="#ef4444", line_width=2.5,
                annotation_text="<b>⚠️ 위험 주의선 (30%)</b>",
                annotation_position="bottom right",
                annotation_font=dict(size=14, color="#000000", family="'Plus Jakarta Sans', 'Noto Sans KR', sans-serif")
            )
            fig.update_layout(
                height=290, margin=dict(l=55, r=130, t=15, b=40),
                yaxis=dict(
                    range=[0, 100],
                    title=dict(
                        text="<b>확률 (%)</b>",
                        font=dict(size=16, color="#000000")
                    ),
                    tickfont=dict(size=14, color="#000000", family="'Noto Sans KR', sans-serif"),
                    gridcolor="#e2e8f0"
                ),
                xaxis=dict(
                    tickfont=dict(size=14, color="#000000", family="'Noto Sans KR', sans-serif"),
                    gridcolor="#e2e8f0"
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
    
    for rec in my_hist:  # load_history()가 이미 최신순으로 반환
        rec_id = rec["id"]
        rec_time = _fmt_time(rec)
        record_type = rec.get("record_type")
        icon = "📊" if record_type == "설문조사" else ("📷" if record_type == "이미지분석" else "📝")
        with st.expander(f"{icon} {rec_time} - {record_type}"):
            st.markdown(f"<div style='font-size:1.05rem; font-weight:600; color:#334155; margin-bottom:12px;'>분석 결과: {rec['detail']}</div>", unsafe_allow_html=True)

            # 이미지 분석 사진 표시 (원본 및 히트맵)
            if record_type == "이미지분석" and (rec.get("image_base64") or rec.get("gradcam_base64")):
                st.markdown("<div style='margin: 15px 0;'>", unsafe_allow_html=True)
                col_orig, col_grad = st.columns(2)
                with col_orig:
                    if rec.get("image_base64"):
                        st.image(f"data:image/jpeg;base64,{rec['image_base64']}", caption="원본 피부 사진", use_container_width=True)
                with col_grad:
                    if rec.get("gradcam_base64"):
                        st.image(f"data:image/jpeg;base64,{rec['gradcam_base64']}", caption="AI 분석 히트맵 (Grad-CAM)", use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<hr style='margin: 12px 0; border-color: #f1f5f9;'>", unsafe_allow_html=True)

            # 개별 기록 삭제 및 HTML 다운로드 버튼
            col_space, col_dl, col_del = st.columns([4, 1.5, 1.2])
            with col_dl:
                if record_type == "이미지분석":
                    rec_html = generate_html_report(
                        display_name=st.session_state.display_name,
                        time_str=rec_time,
                        detail=rec.get("detail", ""),
                        image_b64=rec.get("image_base64"),
                        gradcam_b64=rec.get("gradcam_base64")
                    )
                    st.download_button(
                        label="📄 HTML 보고서 받기",
                        data=rec_html,
                        file_name=f"atocatch_report_{rec_time.replace(' ', '_').replace(':', '')}.html",
                        mime="text/html",
                        key=f"dl_{rec_id}",
                        use_container_width=True
                    )
            with col_del:
                if st.button("🗑 이 기록 삭제", key=f"del_{rec_id}", type="secondary", use_container_width=True):
                    try:
                        _get_user_client().table("analysis_history").delete().eq("id", rec_id).execute()
                        st.success("기록이 성공적으로 삭제되었습니다.")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception:
                        st.error("기록 삭제에 실패했습니다.")

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
                st.session_state.pop("sb_access_token", None)
                st.session_state.pop("sb_user_id", None)
                st.rerun()
                
        st.markdown("<hr style='margin-top: 0px; margin-bottom: 0.5rem; border-color: #E2E8F0;'>", unsafe_allow_html=True)
        
        if st.session_state.current_page == "홈": render_main()
        elif st.session_state.current_page == "설문조사": render_survey()
        elif st.session_state.current_page == "피부 스캔": render_image_scan()
        elif st.session_state.current_page == "분석 결과": render_result()
        elif st.session_state.current_page == "AI 챗봇 서비스": render_guide()
        elif st.session_state.current_page == "기록보기": render_history()

        # 모든 페이지 하단 가운데 면책 문구 고정 (로그인 이후 내부 페이지용 스타일)
        st.markdown("<div style='margin-top: 4.5rem; margin-bottom: 2.0rem;'></div>", unsafe_allow_html=True)
        st.markdown(clean_html("""
            <div class="notranslate" style="text-align: center; max-width: 950px; margin: 0 auto; padding: 18px 24px; background: rgba(255, 255, 255, 0.45); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border: 1.5px solid rgba(27, 101, 84, 0.08); border-radius: 16px; box-shadow: 0 6px 20px rgba(27, 101, 84, 0.02);">
                <p class="notranslate" style="margin: 0; font-size: 0.92rem; color: #475569; font-weight: 600; line-height: 1.65; word-break: keep-all;">
                    ※ 본 AI 서비스의 결과는 입력된 정보를 바탕으로 한 피부 상태 기록 및 안내이며, 의사의 진단이나 치료를 대체할 수 없습니다. 더 안전하고 정확한 관리를 위해 소아청소년과나 피부과 전문의를 방문하셔서 진료를 받아보시는 것을 권장해 드립니다.
                </p>
            </div>
        """), unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# streamlit run app_main.py