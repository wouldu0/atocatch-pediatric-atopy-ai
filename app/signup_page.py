import streamlit as st

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
        
        /* 컬럼 레이아웃 설정 */
        div[data-testid="column"] {
            display: flex;
            flex-direction: column;
            justify-content: center;
            height: 100vh;
        }
        
        /* 왼쪽 컬럼 배경 및 패딩 */
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

        /* 가입하기 버튼 스타일 */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            height: 50px;
            font-weight: 600;
            font-size: 1rem;
            background-color: #1B6554 !important;
            color: white !important;
            border: none !important;
            margin-top: 15px;
        }
        
        /* 약관 동의 체크박스 텍스트 */
        .stCheckbox label {
            font-size: 0.85rem !important;
            color: #555 !important;
        }

        /* 푸터 영역 */
        .footer {
            display: flex;
            justify-content: space-between;
            color: #A0A0A0;
            font-size: 0.75rem;
            margin-top: 3rem;
        }
    </style>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(layout="wide", page_title="AtoCatch Sign Up", initial_sidebar_state="collapsed")
    set_custom_css()

    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        # 왼쪽 UI: 이미지 및 안내 문구
        st.markdown("""
            <div style="background-color: #E2ECFC; border-radius: 20px; overflow: hidden; position: relative; box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 10px;">
                <!-- 회원가입 분위기에 맞는 샘플 이미지 -->
                <img src="https://images.unsplash.com/photo-1519689680058-324335c77eba?ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=80" style="width: 100%; height: auto; display: block;" alt="Happy baby">
                
                <div style="position: absolute; top: 15px; right: 15px; font-size: 1.8rem; color: #AAC6C1;">♡</div>
                
                <!-- 하단 오버레이 카드 -->
                <div style="background: white; border-radius: 12px; padding: 25px 20px; text-align: center; margin: -50px 30px 30px 30px; position: relative; box-shadow: 0 5px 20px rgba(0,0,0,0.08); z-index: 2;">
                    <div style="color: #1B6554; font-weight: 700; font-size: 1.1rem; margin-bottom: 5px;">신생아 아토피 피부염</div>
                    <div style="color: #4A83DA; font-weight: 800; font-size: 1.6rem; margin-bottom: 10px;">초기 예측 서비스</div>
                    <div style="color: #888; font-size: 0.8rem; margin-bottom: 15px; line-height: 1.4;">가장 소중한 우리 아기, 건강한 피부를 위한 스마트 솔루션.<br>지금 가입하고 시작하세요.</div>
                </div>
            </div>
            
            <div class="left-badge">⛨ AI Infant Healthcare</div>
            <div class="left-title">
                우리아이 피부건강,<br>
                <span style="color: #1B6554;">지금 바로 시작하세요</span>
            </div>
            <div class="left-subtitle">
                AtoCatch 회원가입을 통해 아기의 미세한 피부 변화를 체계적으로 관리하고, 전문가 수준의 AI 분석 결과를 받아보세요.
            </div>
        """, unsafe_allow_html=True)

    with col2:
        # 오른쪽 UI: 회원가입 폼 구성
        st.markdown("""
            <div class="logo-container">
                <div class="logo-icon">☻</div>
                AtoCatch
            </div>
            <div class="right-title">회원가입</div>
            <div class="right-subtitle">AtoCatch의 새로운 가족이 되어주세요</div>
        """, unsafe_allow_html=True)
        
        # 1. 정보 입력창
        name = st.text_input("Full Name", placeholder="👤  이름을 입력하세요")
        email = st.text_input("Email Address", placeholder="✉  name@example.com")
        password = st.text_input("Password", type="password", placeholder="🔒  비밀번호 (8자리 이상)")
        password_confirm = st.text_input("Confirm Password", type="password", placeholder="🔒  비밀번호 확인")
        
        # 2. 약관 동의 체크박스
        st.write("") # 약간의 여백 추가
        agree = st.checkbox("이용약관 및 개인정보 처리방침에 동의합니다.")
        
        # 3. 가입하기 버튼 및 로직
        if st.button("가입하기 →", use_container_width=True):
            if not name or not email or not password:
                st.error("모든 필드를 입력해 주세요.")
            elif password != password_confirm:
                st.error("비밀번호가 일치하지 않습니다.")
            elif not agree:
                st.warning("약관에 동의해 주세요.")
            else:
                st.success(f"🎉 회원가입 완료! 환영합니다, {name}님.")
                st.balloons()
        
        # 4. 로그인 화면 이동 링크
        st.markdown('<div style="text-align: center; font-size: 0.9rem; color: #666; margin-top: 25px;">이미 계정이 있으신가요? <span style="color: #1B6554; font-weight: 700; cursor: pointer;">로그인 하기</span></div>', unsafe_allow_html=True)
        
        # 5. 푸터 영역
        st.markdown("""
            <div class="footer">
                <div>© 2024 AtoCatch AI</div>
                <div>
                    <span style="margin-right: 15px; cursor: pointer;">Privacy Policy</span>
                    <span style="cursor: pointer;">Terms of Service</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()