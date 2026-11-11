from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from streamlit_paste_button import paste_image_button

from reader import ReaderOptions, ReadResult, pil_to_png, read_datamatrix, read_image_bytes


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
CASE_DIRS = {
    "test_cases": ROOT / "test_cases",
    "test_cases2": ROOT / "test_cases2",
}


st.set_page_config(page_title="2D Data Matrix Reader", page_icon="[]", layout="wide")


def main() -> None:
    st.title("2D Data Matrix Reader")

    options = render_controls()

    upload_tab, sample_tab = st.tabs(["업로드", "샘플"])

    with upload_tab:
        pasted = st.session_state.setdefault("clipboard-images", {})
        paste_revision = st.session_state.setdefault("clipboard-revision", 0)
        paste_result = paste_image_button("클립보드 이미지 붙여넣기", key=f"clipboard-{paste_revision}")
        if paste_result and paste_result.image_data is not None:
            # Keep pasted code images lossless; JPEG artifacts can erase modules.
            data = pil_to_png(paste_result.image_data.convert("RGB"))
            name = f"clipboard_{hashlib.sha256(data).hexdigest()[:12]}.png"
            pasted[name] = data
        if pasted:
            st.caption(f"붙여넣은 이미지 {len(pasted)}개")
            if st.button("붙여넣은 이미지 지우기"):
                st.session_state["clipboard-images"] = {}
                st.session_state["clipboard-revision"] += 1
                st.rerun()
        uploaded = st.file_uploader(
            "이미지",
            type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        ) or []
        files = [(file.name, file.getvalue()) for file in uploaded] + list(pasted.items())
        render_batch(files, options, "uploaded")

    with sample_tab:
        sample_name = st.selectbox("폴더", list(CASE_DIRS), index=1 if CASE_DIRS["test_cases2"].exists() else 0)
        sample_dir = CASE_DIRS[sample_name]
        sample_files = sorted(sample_dir.glob("*"), key=natural_key)
        image_files = [path for path in sample_files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        if not image_files:
            st.info("샘플 폴더가 비어 있습니다. 이미지를 업로드하거나 로컬 샘플을 추가하세요.")
            return
        selected_names = st.multiselect(
            "샘플",
            [path.name for path in image_files],
            default=[path.name for path in image_files],
        )
        selected = [(path.name, path.read_bytes()) for path in image_files if path.name in selected_names]
        render_batch(selected, options, f"samples-{sample_name}")


def natural_key(path: Path) -> tuple[int, str]:
    return (int(path.stem), path.name) if path.stem.isdigit() else (10**9, path.name)


def render_controls() -> ReaderOptions:
    with st.sidebar:
        st.header("스캔")
        mode_label = st.radio("강도", ["빠름", "균형", "강함"], index=2, horizontal=True)
        mode = {"빠름": "fast", "균형": "balanced", "강함": "aggressive"}[mode_label]
        max_seconds = st.slider("이미지당 로컬 판독 제한(초)", 2, 30, 12, 1)
        st.header("GLM 전처리 제안")
        configured = bool(os.getenv("GLM_API_KEY") or os.getenv("ZAI_API_KEY") or os.getenv("GLM_BASE_URL"))
        use_glm = st.checkbox("GLM으로 이미지 분석", value=configured,
                              help="원본 판독에 실패하면 GLM이 전처리를 제안합니다. 최종 값은 바코드 디코더로 확인합니다.")
        glm_timeout_seconds = 20.0
        if use_glm:
            st.caption(f"모델: {os.getenv('GLM_MODEL') or '미설정'}")
            glm_timeout_seconds = st.slider("GLM 응답 제한(초)", 5, 60, 20, 5)
            st.caption("GLM 응답 시간은 로컬 판독 제한에 추가됩니다.")

        st.header("화면 조정")
        brightness = st.slider("밝기", -80, 80, 0, 5)
        contrast = st.slider("대비", 0.4, 3.0, 1.0, 0.1)
        saturation = st.slider("채도", 0.0, 2.5, 1.0, 0.1)
        sharpness = st.slider("선명도", 0.0, 4.0, 1.0, 0.1)
        use_libdmtx = st.checkbox("libdmtx 보조", value=True)

    return ReaderOptions(
        mode=mode,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        sharpness=sharpness,
        max_seconds=float(max_seconds),
        use_libdmtx=use_libdmtx,
        use_glm=use_glm,
        glm_timeout_seconds=float(glm_timeout_seconds),
    )


def render_batch(files: list[tuple[str, bytes]], options: ReaderOptions, namespace: str) -> None:
    if not files:
        st.session_state.pop(f"results-{namespace}", None)
        st.session_state.pop(f"files-{namespace}", None)
        st.info("이미지를 선택하세요.")
        return

    file_names = [(name, hashlib.sha256(data).hexdigest()) for name, data in files]
    signature = (file_names, asdict(options), os.getenv("GLM_MODEL"), os.getenv("GLM_BASE_URL"))
    if st.session_state.get(f"files-{namespace}") != signature:
        st.session_state.pop(f"results-{namespace}", None)

    run = st.button("읽기", key=f"run-{namespace}", type="primary")
    if not run and f"results-{namespace}" not in st.session_state:
        return

    if run:
        results: list[tuple[str, bytes, ReadResult]] = []
        progress = st.progress(0.0)
        status = st.empty()
        for index, (name, data) in enumerate(files, start=1):
            status.write(f"{name} ({index}/{len(files)})")
            # A previous API failure must not be cached as a permanent result.
            try:
                if options.use_glm:
                    result = read_datamatrix(read_image_bytes(data), options)
                else:
                    result = cached_read(name, data, asdict(options))
            except (OSError, ValueError) as exc:
                result = ReadResult(False, [], 0, 0.0, "invalid-image", None, error=str(exc))
            results.append((name, data, result))
            progress.progress(index / len(files))
        status.empty()
        st.session_state[f"results-{namespace}"] = results
        st.session_state[f"files-{namespace}"] = signature

    results = st.session_state.get(f"results-{namespace}", [])
    if not results:
        return

    render_summary(results)
    render_table(results)
    render_details(results, namespace)


@st.cache_data(show_spinner=False)
def cached_read(name: str, data: bytes, options_dict: dict[str, object]) -> ReadResult:
    image = read_image_bytes(data)
    options = ReaderOptions(**options_dict)
    return read_datamatrix(image, options)


def render_summary(results: list[tuple[str, bytes, ReadResult]]) -> None:
    total = len(results)
    decoded = sum(result.ok for _, _, result in results)
    avg_time = sum(result.elapsed_seconds for _, _, result in results) / total
    col1, col2, col3 = st.columns(3)
    col1.metric("전체", total)
    col2.metric("성공", decoded)
    col3.metric("평균 초", f"{avg_time:.2f}")


def render_table(results: list[tuple[str, bytes, ReadResult]]) -> None:
    rows = []
    for name, _, result in results:
        hits = result.hits
        rows.append(
            {
                "파일": name,
                "상태": "성공" if result.ok else "미검출",
                "값": "\n".join(hit.text for hit in hits),
                "엔진": "\n".join(hit.engine for hit in hits),
                "전처리": result.best_variant,
                "시도": result.attempts,
                "초": round(result.elapsed_seconds, 2),
            }
        )
    frame = pd.DataFrame(rows)
    st.dataframe(frame, width="stretch", hide_index=True)
    st.download_button(
        "CSV",
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name="datamatrix_results.csv",
        mime="text/csv",
    )


def render_details(results: list[tuple[str, bytes, ReadResult]], namespace: str) -> None:
    for index, (name, data, result) in enumerate(results):
        status = "성공" if result.ok else "미검출"
        with st.expander(f"{status} · {name}", expanded=index == 0):
            left, right = st.columns(2)
            with left:
                if result.error:
                    st.error("이미지 파일을 열 수 없습니다.")
                else:
                    st.image(data, caption="원본", width="stretch")
            with right:
                if result.preview_png:
                    st.image(result.preview_png, caption="처리", width="stretch")
                else:
                    st.write("처리 이미지 없음")

            if result.ok:
                for hit_index, hit in enumerate(result.hits, start=1):
                    st.text_area(
                        f"값 {hit_index}",
                        hit.text,
                        height=80,
                        key=f"value-{namespace}-{hashlib.sha256(data + hit.text.encode()).hexdigest()}-{index}-{hit_index}",
                        disabled=True,
                    )
                    st.caption(f"{hit.engine} · {hit.variant}")
            else:
                st.warning("Data Matrix가 읽히지 않았습니다.")
                st.caption(f"{result.best_variant} · {result.attempts}회 · {result.elapsed_seconds:.2f}초")
            if result.assistance:
                info = result.assistance
                if info.get("status") == "not-needed":
                    st.caption("원본 판독 성공: GLM 호출 생략")
                elif info.get("status") == "ok":
                    st.caption("GLM 제안으로 판독" if info.get("used") else "GLM 제안 후 로컬 추가 탐색으로 판독 시도")
                    st.json(info.get("hints", []), expanded=False)
                else:
                    st.caption(f"GLM 연결 결과: {info.get('error', info.get('status'))}. 로컬 판독을 실행했습니다.")


if __name__ == "__main__":
    main()
