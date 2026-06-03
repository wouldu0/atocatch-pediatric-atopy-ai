import streamlit as st
import base64, io, os
from PIL import Image

def _get_login_img_b64():
    try:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "design", "realmain.png")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((800, int(img.height * 800 / img.width)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except:
        return ""

def set_custom_css():
    st.markdown("""
    <style>
        /* 기본 여백 제거 및 전체 화면 사용 */
        .block-container {
            padding: 0rem !important;
            max-width: 100% !important;
        }
        header {
            visibility: hidden !important;
        }
        
        /* 컬럼 레이아웃 설정 (vh 단위 사용으로 뷰포트 전체 높이 차지) */
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100vh;
        }
        
        /* 왼쪽 컬럼 배경(그라데이션) 및 패딩 */
        div[data-testid="column"]:nth-of-type(1) {
            background: linear-gradient(135deg, #e4f0ee 0%, #edf1e9 100%);
            padding: 0 10% !important;
        }
        
        /* 오른쪽 컬럼 배경 및 패딩 */
        div[data-testid="column"]:nth-of-type(2) {
            background-color: white;
            padding: 0 15% !important;
        }

        /* 왼쪽 텍스트 스타일 */
        .left-badge {
            display: inline-block;
            background-color: #D1E5DE;
            color: #1B6554;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .left-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1A1A1A;
            line-height: 1.3;
            margin-bottom: 1rem;
        }
        .left-subtitle {
            font-size: 1rem;
            color: #555555;
            line-height: 1.6;
        }
        
        /* 오른쪽 로고 및 타이틀 스타일 */
        .logo-container {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
            font-weight: 700;
            color: #1B6554;
        }
        .logo-icon {
            background-color: #1B6554;
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
        }
        .right-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
            color: #222;
        }
        .right-subtitle {
            font-size: 0.95rem;
            color: #666;
            margin-bottom: 2rem;
        }

        /* 입력 폼 스타일 */
        .stTextInput > div > div > input {
            background-color: #F5F7F9 !important;
            border: none !important;
            border-radius: 8px !important;
            padding: 12px 15px !important;
        }
        .stTextInput label {
            font-size: 0.9rem !important;
            color: #333 !important;
            font-weight: 600 !important;
            margin-bottom: 5px !important;
        }

        /* 버튼 기본 스타일 */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            height: 50px;
            font-weight: 600;
            font-size: 1rem;
        }
        
        /* 로그인(기본) 버튼 (오른쪽 컬럼의 첫 번째 버튼) */
        div[data-testid="column"]:nth-of-type(2) .stButton:first-of-type > button {
            background-color: #1B6554 !important;
            color: white !important;
            border: none !important;
            margin-top: 10px;
        }
        
        /* 게스트 / QR코드 버튼 컨테이너 내부 버튼 */
        .social-btn-container button {
            background-color: white !important;
            color: #444 !important;
            border: 1px solid #E0E0E0 !important;
        }

        /* 구분선 "또는 간편 로그인" */
        .divider {
            display: flex;
            align-items: center;
            text-align: center;
            color: #999;
            font-size: 0.85rem;
            margin: 2rem 0;
        }
        .divider::before, .divider::after {
            content: '';
            flex: 1;
            border-bottom: 1px solid #EAEAEA;
        }
        .divider:not(:empty)::before {
            margin-right: .5em;
        }
        .divider:not(:empty)::after {
            margin-left: .5em;
        }
        
        /* 푸터 영역 */
        .footer {
            display: flex;
            justify-content: space-between;
            color: #A0A0A0;
            font-size: 0.75rem;
            margin-top: 4rem;
        }
        
        /* Forgot Password 텍스트 레이아웃 보정 */
        .forgot-pw {
            position: absolute;
            right: 0;
            top: -28px;
            font-size: 0.8rem;
            color: #1B6554;
            cursor: pointer;
            z-index: 10;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    # 사이드바를 숨기고 전체 너비를 사용합니다.
    st.set_page_config(layout="wide", page_title="AtoCatch Login", initial_sidebar_state="collapsed")
    set_custom_css()

    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        # --------------------------
        # 왼쪽 UI: 배너, 이미지, 텍스트
        # --------------------------
        _img_b64 = _get_login_img_b64()
        st.markdown(f"""
            <div style="background-color: #E2ECFC; border-radius: 20px; overflow: hidden; position: relative; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 10px;">
                <!-- 실제 앱에서는 적절한 이미지 링크나 로컬 이미지 경로를 넣어서 변경해 주세요 -->
                <img src="data:image/jpeg;base64,{{_img_b64}}" style="width: 100%; height: auto; display: block;" alt="Mother and baby">
                
                <div style="position: absolute; top: 15px; right: 15px; font-size: 1.8rem; color: #AAC6C1;">♡</div>
                
                <!-- 하단 오버레이 카드(초기 예측 서비스) -->
                <div style="background: white; border-radius: 12px; padding: 25px 20px; text-align: center; margin: -50px 30px 30px 30px; position: relative; box-shadow: 0 5px 20px rgba(0,0,0,0.08); z-index: 2;">
                    <div style="color: #1B6554; font-weight: 700; font-size: 1.1rem; margin-bottom: 5px;">신생아 아토피 피부염</div>
                    <div style="color: #4A83DA; font-weight: 800; font-size: 1.6rem; margin-bottom: 10px;">초기 예측 서비스</div>
                    <div style="color: #888; font-size: 0.8rem; margin-bottom: 15px; line-height: 1.4;">가장 소중한 우리 아기, 건강한 피부를 위한 스마트 솔루션.<br>지금 확인하세요.</div>
                    <div style="display: inline-block; padding: 8px 16px; border: 1px solid #E0E0E0; border-radius: 6px; color: #333; font-size: 0.85rem; font-weight: 600; cursor: pointer;">
                        🔗 지금 예측 시작하기
                    </div>
                </div>
            </div>
            
            <div class="left-badge">⛨ AI Infant Healthcare</div>
            <div class="left-title">
                우리 아이 아토피,<br>
                <span style="color: #1B6554;">미리 알고 막아주세요</span>
            </div>
            <div class="left-subtitle">
                AtoCatch AI는 아기의 미세한 피부 변화를 감지하여 부모님께 안심을 드립니다.
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # --------------------------
        # 오른쪽 UI: 폼 구성
        # --------------------------
        st.markdown("""
            <div class="logo-container">
                <div class="logo-icon">☻</div>
                AtoCatch
            </div>
            <div class="right-title">AtoCatch에 오신 것을 환영합니다</div>
            <div class="right-subtitle">아이의 피부 건강을 위한 첫 걸음</div>
        """, unsafe_allow_html=True)
        
        # 1. 이메일 입력창
        email = st.text_input("Email Address", placeholder="✉  name@example.com")
        
        # 2. 비밀번호 입력창 & Forgot Password? 링크
        st.markdown('<div style="position: relative;">', unsafe_allow_html=True)
        st.markdown('<div class="forgot-pw">Forgot?</div>', unsafe_allow_html=True)
        password = st.text_input("Password", type="password", placeholder="🔒  ••••••••")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 3. 아이디 저장하기 체크박스
        save_id = st.checkbox("아이디 저장하기")
        
        # 4. 로그인 버튼
        if st.button("로그인 →", use_container_width=True):
            st.success(f"로그인 시도 - Email: {email}")
        
        # 5. 구분선
        st.markdown('<div class="divider">또는 간편 로그인</div>', unsafe_allow_html=True)
        
        # 6. 소셜 / 간편 로그인 버튼
        st.markdown('<div class="social-btn-container">', unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            st.button("👤 Guest", use_container_width=True)
        with col_btn2:
            st.button("📱 QR Code", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
            
        # 7. 회원가입 문구
        st.markdown('<div style="text-align: center; font-size: 0.9rem; color: #666; margin-top: 30px;">아직 회원이 아니신가요? <span style="color: #1B6554; font-weight: 700; cursor: pointer;">회원가입 하기</span></div>', unsafe_allow_html=True)
        
        # 8. 푸터 영역
        st.markdown("""
            <div class="footer">
                <div>© 2024 AtoCatch AI</div>
                <div>
                    <span style="margin-right: 15px; cursor: pointer;">Privacy Policy</span>
                    <span style="cursor: pointer;">Support</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    
    