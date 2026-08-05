"""
DermNet NZ 피부질환 이미지 크롤러 v2
- 갤러리 페이지 + 토픽 본문 페이지 이미지 모두 수집
- 하위 유형/관련 질환 갤러리 대폭 추가

사용법:
    python dermnet_crawler_v2.py --output_dir E:\skin\dermnet_images

필요 패키지:
    conda install requests beautifulsoup4 tqdm
"""

import os
import time
import argparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

# ──────────────────────────────────────────────
# 크롤링 대상: 갤러리 페이지 + 토픽(본문) 페이지
# ──────────────────────────────────────────────
DISEASE_PAGES = {
    "atopic_dermatitis": [
        # 갤러리 페이지
        "https://dermnetnz.org/images/atopic-dermatitis-images",
        "https://dermnetnz.org/images/atopic-flexural-eczema-images",
        "https://dermnetnz.org/images/atopic-hand-dermatitis-images",
        # 토픽 페이지 (본문 내 이미지)
        "https://dermnetnz.org/topics/atopic-dermatitis",
        "https://dermnetnz.org/cme/dermatitis/atopic-dermatitis",
        "https://dermnetnz.org/topics/atopic-dermatitis-complications",
    ],
    "psoriasis": [
        # 갤러리 페이지
        "https://dermnetnz.org/images/chronic-plaque-psoriasis-images",
        "https://dermnetnz.org/images/scalp-psoriasis-images",
        "https://dermnetnz.org/images/nail-psoriasis-images",
        "https://dermnetnz.org/images/palmoplantar-psoriasis-images",
        "https://dermnetnz.org/images/flexural-psoriasis-images",
        "https://dermnetnz.org/images/erythrodermic-psoriasis-images",
        "https://dermnetnz.org/images/generalised-pustular-psoriasis-images",
        "https://dermnetnz.org/images/psoriasis-affecting-the-penis-images",
        # 토픽 페이지
        "https://dermnetnz.org/topics/psoriasis",
        "https://dermnetnz.org/topics/chronic-plaque-psoriasis",
        "https://dermnetnz.org/topics/guttate-psoriasis",
        "https://dermnetnz.org/topics/scalp-psoriasis",
        "https://dermnetnz.org/topics/nail-psoriasis",
        "https://dermnetnz.org/topics/flexural-psoriasis",
    ],
    "acne": [
        # 갤러리 페이지
        "https://dermnetnz.org/images/acne-vulgaris-images",
        "https://dermnetnz.org/images/acne-scarring-images",
        "https://dermnetnz.org/images/acne-affecting-the-back-images",
        "https://dermnetnz.org/images/comedonal-acne-images",
        "https://dermnetnz.org/images/nodulocystic-acne-images",
        "https://dermnetnz.org/images/infantile-acne-images",
        "https://dermnetnz.org/images/acne-affecting-the-trunk-images",
        # 토픽 페이지
        "https://dermnetnz.org/topics/acne",
        "https://dermnetnz.org/topics/acne-vulgaris",
        "https://dermnetnz.org/topics/comedonal-acne",
        "https://dermnetnz.org/topics/nodulocystic-acne",
        "https://dermnetnz.org/topics/acne-affecting-the-back",
    ],
    "rosacea": [
        # 갤러리 페이지
        "https://dermnetnz.org/images/rosacea-images",
        "https://dermnetnz.org/images/steroid-rosacea-images",
        "https://dermnetnz.org/images/rhinophyma-images",
        "https://dermnetnz.org/images/ocular-rosacea-images",
        "https://dermnetnz.org/images/papulopustular-rosacea-images",
        "https://dermnetnz.org/images/erythematotelangiectatic-rosacea-images",
        # 토픽 페이지
        "https://dermnetnz.org/topics/rosacea",
        "https://dermnetnz.org/topics/papulopustular-rosacea",
        "https://dermnetnz.org/topics/rhinophyma",
        "https://dermnetnz.org/topics/ocular-rosacea",
    ],
    "seborrheic_dermatitis": [
        # 갤러리 페이지
        "https://dermnetnz.org/images/seborrhoeic-dermatitis-images",
        "https://dermnetnz.org/images/seborrhoeic-dermatitis-of-the-scalp-images",
        "https://dermnetnz.org/images/cradle-cap-images",
        "https://dermnetnz.org/images/dandruff-images",
        # 토픽 페이지
        "https://dermnetnz.org/topics/seborrhoeic-dermatitis",
        "https://dermnetnz.org/topics/seborrhoeic-dermatitis-of-the-face-and-body",
        "https://dermnetnz.org/topics/seborrhoeic-dermatitis-of-the-scalp",
        "https://dermnetnz.org/topics/cradle-cap",
    ],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_image_urls(page_url: str) -> list[str]:
    """페이지에서 /assets/collection/ 경로의 이미지 URL 추출"""
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [SKIP] {page_url} → {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    image_urls = []

    for img in soup.find_all("img"):
        src = img.get("src", "")
        # DermNet 질환 이미지는 /assets/collection/ 경로에 저장됨
        if "/assets/collection/" in src:
            full_url = urljoin(page_url, src)
            image_urls.append(full_url)

    # 중복 제거 (순서 유지)
    return list(dict.fromkeys(image_urls))


def download_image(url: str, save_path: str) -> bool:
    """단일 이미지 다운로드"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
        resp.raise_for_status()

        # 이미지인지 확인
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not url.endswith((".jpg", ".jpeg", ".png", ".webp")):
            return False

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException:
        return False


def crawl_disease(disease_name: str, page_urls: list[str], output_dir: str) -> int:
    """특정 질환의 모든 페이지에서 이미지 수집"""
    disease_dir = os.path.join(output_dir, disease_name)
    os.makedirs(disease_dir, exist_ok=True)

    all_urls = []
    for page_url in page_urls:
        print(f"    스캔: {page_url}")
        urls = get_image_urls(page_url)
        if urls:
            print(f"      → {len(urls)}장 발견")
        all_urls.extend(urls)

    # 전체 중복 제거
    all_urls = list(dict.fromkeys(all_urls))
    print(f"    총 {len(all_urls)}장 (중복 제거 후)")

    if not all_urls:
        return 0

    downloaded = 0
    for url in tqdm(all_urls, desc=f"    다운로드", ncols=80):
        # 파일명 생성: 원본 파일명 사용
        filename = os.path.basename(url).split("?")[0]  # 쿼리스트링 제거
        if not filename:
            continue

        save_path = os.path.join(disease_dir, filename)

        # 이미 존재하면 스킵
        if os.path.exists(save_path):
            downloaded += 1
            continue

        if download_image(url, save_path):
            downloaded += 1

        # 서버 부담 방지
        time.sleep(0.5)

    return downloaded


def main():
    parser = argparse.ArgumentParser(description="DermNet NZ 피부질환 이미지 크롤러 v2")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dermnet_images",
        help="이미지 저장 경로 (기본값: ./dermnet_images)",
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=None,
        help="크롤링할 질환 목록. "
             "선택: atopic_dermatitis, psoriasis, acne, rosacea, seborrheic_dermatitis",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="요청 간 대기 시간(초) (기본값: 0.5)",
    )
    args = parser.parse_args()

    if args.diseases:
        targets = {k: v for k, v in DISEASE_PAGES.items() if k in args.diseases}
        unknown = set(args.diseases) - set(DISEASE_PAGES.keys())
        if unknown:
            print(f"[WARNING] 알 수 없는 질환명: {unknown}")
    else:
        targets = DISEASE_PAGES

    print("=" * 60)
    print("DermNet NZ 피부질환 이미지 크롤러 v2")
    print(f"저장 경로: {args.output_dir}")
    print(f"대상 질환: {list(targets.keys())}")
    print(f"요청 딜레이: {args.delay}초")
    print("=" * 60)

    summary = {}
    for disease_name, page_urls in targets.items():
        print(f"\n[{disease_name}] ({len(page_urls)}개 페이지)")
        count = crawl_disease(disease_name, page_urls, args.output_dir)
        summary[disease_name] = count

    # 결과 요약
    print("\n" + "=" * 60)
    print("크롤링 완료 요약")
    print("=" * 60)
    total = 0
    for disease, count in summary.items():
        print(f"  {disease:30s}: {count:4d}장")
        total += count
    print(f"  {'합계':30s}: {total:4d}장")
    print(f"\n저장 위치: {args.output_dir}")

    # 이미 받은 v1 이미지와 중복될 수 있으므로 안내
    print("\n※ 이전 크롤링(v1)과 같은 output_dir을 쓰면 자동으로 중복 스킵됩니다.")
    print("※ 이 이미지는 DermNet NZ 저작권 하에 있습니다. 출처를 명시하세요.")


if __name__ == "__main__":
    main()