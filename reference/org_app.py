#!/usr/bin/env python3
"""
Streamlit Web App for Barcode Recognition POC
Provides a web UI to upload images and recognize barcodes.
Backend logic is separated into barcode_engine.py.
"""

import os
import io
import csv
import base64
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from streamlit_paste_button import paste_image_button
import cv2
import numpy as np

# Import barcode engine (backend logic)
from barcode_engine import (
    PREPROCESS_CATEGORIES,
    RECOMMENDED_METHODS,
    RECOMMENDED_SET,
    PREPROCESSOR_CLASSIC,
    PREPROCESSOR_VE2_BALANCED,
    PREPROCESSOR_VE2_AGGRESSIVE,
    PREPROCESSOR_ALL,
    PREPROCESSOR_OPTIONS,
    DECODER_ZXINGCPP,
    DECODER_PYLIBDMTX,
    DECODER_OPTIONS,
    get_all_methods_with_recommended,
    get_all_method_ids,
    get_recommended_method_ids,
    get_available_decoders,
    decode_image_from_array,
    image_to_base64,
    results_to_csv_rows,
    csv_rows_to_stringio,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="2D Barcode Scanner",
    page_icon="🔌",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    # Session state initialization
    if "results" not in st.session_state:
        st.session_state.results = []
    if "selected_files" not in st.session_state:
        st.session_state.selected_files = []
    if "selected_files_data" not in st.session_state:
        st.session_state.selected_files_data = {}  # filename -> bytes
    if "csv_data" not in st.session_state:
        st.session_state.csv_data = None
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0  # 파일 업로더 리셋을 위한 키
    if "paste_key" not in st.session_state:
        st.session_state.paste_key = 0  # paste 버튼 리셋을 위한 키

    # Handle individual file delete from HTML button
    if "delete" in st.query_params:
        try:
            delete_idx = int(st.query_params["delete"])
            if 0 <= delete_idx < len(st.session_state.selected_files):
                removed_file = st.session_state.selected_files.pop(delete_idx)
                st.session_state.selected_files_data.pop(removed_file, None)
                st.rerun()
        except (ValueError, IndexError):
            pass

    # Title
    st.subheader("2D Barcode Scanner")

    # Sidebar - Settings
    with st.sidebar:
        st.header("🎛️ 설정")

        # --- Preprocessor Selection ---
        st.subheader("전처리 엔진")
        
        preprocessor_options_map = {            
            "Fast (기본)": PREPROCESSOR_CLASSIC,
            "Balanced (고급 기법 (~50-100종))": PREPROCESSOR_VE2_BALANCED,
            "Quality (Balanced + 회전 (~500-650종))": PREPROCESSOR_VE2_AGGRESSIVE,            
            "All (모든 전처리 순차 시도)": PREPROCESSOR_ALL,
        }
        
        selected_preprocessor_names = st.selectbox(
            "전처리 엔진 선택",
            options=list(preprocessor_options_map.keys()),
            index=0,  # Default: Classic
            # help="",
        )
        selected_preprocessor = preprocessor_options_map[selected_preprocessor_names]

        # --- Decoder Selection ---
        st.subheader("디코더")
        
        available_decoders = get_available_decoders()
        decoder_options_map = {
            "ZXingCPP(기본)": DECODER_ZXINGCPP,
            "PyLibDMTX": DECODER_PYLIBDMTX,
        }
        
        # Filter available decoders
        available_decoder_options = {k: v for k, v in decoder_options_map.items() if v in available_decoders}
        
        if available_decoder_options:
            selected_decoder_name = st.selectbox(
                "디코더 선택",
                options=list(available_decoder_options.keys()),
                index=0,
                help="ZXingCPP: 일반 바코드 인식\nPyLibDMTX: DataMatrix 특화",
            )
            selected_decoder = available_decoder_options[selected_decoder_name]
        else:
            st.error("사용 가능한 디코더가 없습니다.")
            selected_decoder = DECODER_ZXINGCPP

        # --- Classic Method Selection (only for Classic preprocessor) ---
        if selected_preprocessor == PREPROCESSOR_CLASSIC:
            st.markdown("---")
            st.subheader("전처리 방법")

            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ 전체선택", use_container_width=True):
                    st.session_state.all_methods = get_all_method_ids()
            with col2:
                if st.button("❌ 전체해제", use_container_width=True):
                    st.session_state.all_methods = []
            with col3:
                if st.button("⭐ 추천선택", use_container_width=True):
                    st.session_state.all_methods = get_recommended_method_ids()

            # Initialize session state for methods
            if "all_methods" not in st.session_state:
                st.session_state.all_methods = get_all_method_ids()

            # Multiselect for preprocessing methods
            all_methods = get_all_method_ids()
            selected_methods = st.multiselect(
                "적용할 전처리 방법",
                options=all_methods,
                default=st.session_state.all_methods,
                help="선택한 전처리 방법만 적용됩니다. 비워두면 모든 방법이 적용됩니다.",
            )
            st.session_state.all_methods = selected_methods

            # Selected count
            st.metric("선택된 방법 수", len(selected_methods))

            # Info
            st.markdown("---")
            st.info("""
**선택 가이드:**
- **추천선택**: 최적의 성능을 위한 기본 추천 방법
- **전체선택**: 모든 방법 적용 (시간 소요)
- **비워두기**: 모든 방법 적용
            """)
        else:
            # VE2 mode info
            st.markdown("---")
            mode_info = {
                PREPROCESSOR_VE2_BALANCED: "🔧 **VE2 Balanced 모드**\n\n채널 분리 (RGB/HSV/LAB), 감마/대비 보정, 이진화\n\n약 50-100개 전처리 생성\n소요 시간: 약 5-10초",
                PREPROCESSOR_VE2_AGGRESSIVE: "🚀 **VE2 Aggressive 모드**\n\nBalanced 모든 기법 + 다양한 각도 회전 (-14°~+14°)\n\n약 500-650개 전처리 생성\n소요 시간: 약 10-12초",
            }
            st.info(mode_info.get(selected_preprocessor, ""))

        # --- Paste button in sidebar (placeholder, actual button is in main area) ---
        st.markdown("---")
        st.caption("아래 메인 영역에서 '📋 클립보드에서 붙여넣기' 버튼 클릭")

    # --- Paste button in main area ---
    st.markdown("---")
    st.subheader("📋 클립보드에서 붙여넣기")
    st.info("""💡 "Ctrl+C"로 복사한 이미지가 있다면 아래 버튼을 클릭하세요. 브라우저가 클립보드 접근을 허용하도록 '허용'을 클릭하세요.""")
    
    paste_result = paste_image_button(
        "📋 클립보드에서 붙여넣기",
        key=f"paste_btn_{st.session_state.paste_key}",
        text_color="#ffffff",
        background_color="#27ae60",
        hover_background_color="#2ecc71",
    )

    if paste_result and paste_result.image_data is not None:
        # PIL 이미지를 numpy 배열로 변환
        pil_img = paste_result.image_data
        img_np = np.array(pil_img)

        # RGB를 BGR로 변환하여 OpenCV 형식으로 만듦
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # 이미지를 JPEG로 인코딩하여 저장
        _, buffer = cv2.imencode('.jpg', img_bgr)
        file_data = buffer.tobytes()

        # 클립보드 이미지는 "clipboard_" 접두사 파일명으로 저장
        filename = f"clipboard_{uuid.uuid4().hex[:8]}.jpg"

        if filename not in st.session_state.selected_files:
            st.session_state.selected_files.append(filename)
            st.session_state.selected_files_data[filename] = file_data
            st.success(f"이미지가 클립보드에서 추가되었습니다: {filename}")
            # paste 버튼 리셋
            st.session_state.paste_key = st.session_state.paste_key + 1
        st.rerun()

    # Main content area
    # Stats row
    total_images = len(st.session_state.selected_files)
    success_count = sum(1 for r in st.session_state.results if r["status"] == "success")
    failed_count = sum(1 for r in st.session_state.results if r["status"] != "success")

    # Upload section
    st.markdown("---")
    st.subheader("📄 이미지 업로드")

    # 붙여넣기 안내
    # st.info("💡 **클립보드에서 이미지를 복사하고 위 'Ctrl+V 붙여넣기' 버튼을 클릭하세요.** (브라우저가 클립보드 접근을 허용하도록 '허용'을 클릭하세요)")

    # uploader_key를 동적으로 변경하여 파일 업로더 리셋
    uploaded_files = st.file_uploader(
        "이미지를 선택하거나 드래그하여 업로드하세요",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.selected_files:
                st.session_state.selected_files.append(file.name)
                # Read and store file data immediately (file object becomes invalid on rerun)
                file.seek(0)
                st.session_state.selected_files_data[file.name] = file.read()

    # Display selected files as grid thumbnails
    # Build base64 images for display
    file_images_b64 = {}
    file_sizes = {}
    if st.session_state.selected_files:
        for filename in st.session_state.selected_files:
            file_data = st.session_state.selected_files_data.get(filename, b'')
            if file_data:
                img = cv2.imdecode(
                    np.frombuffer(file_data, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w = img.shape[:2]
                    file_sizes[filename] = f"{w}×{h}"
                    # Resize to 50% for thumbnail
                    new_w, new_h = w // 2, h // 2
                    img_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    # Convert to base64 for display
                    _, buffer = cv2.imencode('.jpg', img_small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    file_images_b64[filename] = base64.b64encode(buffer).decode('utf-8')

        # Remove button (placed before HTML to avoid rendering issues)
        if st.button("🗑️ 모든 이미지 삭제", use_container_width=True):
            st.session_state.selected_files = []
            st.session_state.selected_files_data = {}
            st.session_state.results = []
            st.session_state.uploader_key = st.session_state.uploader_key + 1  # 업로더 리셋을 위한 키 증가
            st.rerun()

    if file_images_b64:
        # Create HTML grid layout with delete buttons
        grid_html = """
<style>
.uploaded-image-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
    padding: 10px;
}
.uploaded-image-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px;
    background: #fafafa;
    text-align: center;
    transition: box-shadow 0.2s;
    position: relative;
}
.uploaded-image-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.uploaded-image-card img {
    width: 100%;
    height: 120px;
    object-fit: contain;
    border-radius: 4px;
    background: #fff;
}
.uploaded-image-card .filename {
    font-size: 12px;
    font-weight: 600;
    margin: 6px 0 2px 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.uploaded-image-card .meta {
    font-size: 11px;
    color: #666;
}
.uploaded-image-card .delete-btn {
    position: absolute;
    top: 4px;
    right: 4px;
    background: rgba(255, 0, 0, 0.8);
    color: white;
    border: none;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    cursor: pointer;
    font-size: 14px;
    line-height: 24px;
    text-align: center;
    display: none;
    z-index: 10;
}
.uploaded-image-card .delete-btn:hover {
    background: rgba(255, 0, 0, 1);
    transform: scale(1.1);
}
</style>
<div class="uploaded-image-grid">
"""
        for idx, filename in enumerate(st.session_state.selected_files):
            if filename in file_images_b64:
                grid_html += f"""
<div class="uploaded-image-card">
<img src="data:image/jpeg;base64,{file_images_b64[filename]}" alt="{filename}">
<div class="filename" title="{filename}">{filename}</div>
<div class="meta">번호: {idx + 1} | {file_sizes.get(filename, 'N/A')}</div>
<form method="GET" action="?delete={idx}" style="position:absolute;top:4px;right:4px;margin:0;">
<button class="delete-btn" type="submit" title="삭">×</button>
</form>
</div>
"""
        grid_html += """
</div>
"""
        st.components.v1.html(grid_html, height=280, scrolling=True)

    # Scan button
    scan_clicked = st.button("🔍 바코드 인식", type="primary", use_container_width=True)

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    if scan_clicked and st.session_state.selected_files:
        # Get selected methods for Classic preprocessor
        if selected_preprocessor == PREPROCESSOR_CLASSIC:
            selected_methods_set = set(selected_methods) if selected_methods else None
        else:
            selected_methods_set = None  # VE2 ignores this

        all_results = []
        total = len(st.session_state.selected_files)

        for idx, filename in enumerate(st.session_state.selected_files):
            status_text.text(f"처리 중: {filename} ({idx + 1}/{total})")
            progress_bar.progress((idx + 1) / total)

            file_data = st.session_state.selected_files_data.get(filename, b'')

            try:
                img = cv2.imdecode(
                    np.frombuffer(file_data, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                result = decode_image_from_array(
                    img,
                    selected_methods=selected_methods_set,
                    preprocessor_type=selected_preprocessor,
                    decoder_type=selected_decoder,
                )
                result["image_file"] = filename
                result["image_b64"] = image_to_base64(img) if img is not None else ""
            except Exception as e:
                result = {
                    "image_file": filename,
                    "barcodes": [],
                    "status": f"error: {str(e)}",
                    "image_b64": "",
                }

            all_results.append(result)

        status_text.text("처리 완료!")
        st.session_state.results = all_results

        # Convert results to CSV
        csv_rows = results_to_csv_rows(all_results)
        st.session_state.csv_data = csv_rows_to_stringio(csv_rows)

        # Save to file as well
        with open("barcode_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image_file", "barcode_index", "decoded_text", "format",
                "preprocessing_method", "status"
            ])
            writer.writeheader()
            writer.writerows(csv_rows)

    # Results section
    st.markdown("---")

    if st.session_state.results:
        # Display results as table
        st.markdown("### 📊 인식 결과 테이블")

        # CSV download button
        col1, col2 = st.columns([5, 1])

        with col2:
            if st.session_state.csv_data:
                st.download_button(
                    label="📊 CSV 다운로드",
                    data=st.session_state.csv_data.encode('utf-8'),
                    file_name=f"barcode_results_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # Prepare table data
        table_rows = []
        for result in st.session_state.results:
            image_file = result.get('image_file', 'Unknown')
            image_b64_raw = result.get('image_b64', '')
            barcodes = result.get('barcodes', [])
            status = result.get('status', 'failed')
            
            # st.column_config.ImageColumn이 인식할 수 있는 full base64 URI로 변환
            # image_to_base64()는 base64 문자열만 반환하므로 프리픽스 추가
            image_b64_uri = f"data:image/png;base64,{image_b64_raw}" if image_b64_raw else ''
            
            if barcodes:
                for bc in barcodes:
                    table_rows.append({
                        '이미지': image_b64_uri,
                        '파일명': image_file,
                        '바코드 값': bc.get('text', ''),
                        '포맷': bc.get('format', 'N/A'),
                        '전처리 방법': bc.get('preprocessing_method', 'N/A'),
                        '상태': '✅ 성공'
                    })
            else:
                # Failed result
                table_rows.append({
                    '이미지': image_b64_uri,
                    '파일명': image_file,
                    '바코드 값': '-',
                    '포맷': '-',
                    '전처리 방법': '-',
                    '상태': f'❌ {status}'
                })
        
        if table_rows:
            df = __import__('pandas', fromlist=['DataFrame']).DataFrame(table_rows)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "이미지": st.column_config.ImageColumn(
                        "미리보기",
                        help="이미지 미리보기",
                        width=100,
                    ),
                    "파일명": st.column_config.TextColumn(
                        "파일명",
                        help="업로드된 파일명",
                        width="medium",
                    ),
                    "바코드 값": st.column_config.TextColumn(
                        "바코드 값",
                        help="인식된 바코드 데이터",
                        width="large",
                    ),
                    "포맷": st.column_config.TextColumn(
                        "포맷",
                        help="바코드 포맷 타입",
                        width="small",
                    ),
                    "전처리 방법": st.column_config.TextColumn(
                        "전처리 방법",
                        help="사용된 전처리 방법",
                        width="medium",
                    ),
                    "상태": st.column_config.TextColumn(
                        "인식 상태",
                        help="인식 성공/실패 상태",
                        width="small",
                    ),
                },
                row_height=80
            )
    elif st.session_state.selected_files:
        st.info("결과가 없습니다. '바코드 인식' 버튼을 클릭하여 스캔하세요.")
    else:
        st.info("결과가 없습니다. 이미지를 업로드하세요.")


if __name__ == "__main__":
    main()
