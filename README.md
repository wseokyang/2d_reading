# 2D Data Matrix Reader

`app.py`를 Streamlit으로 실행하면 이미지 업로드·클립보드 붙여넣기·샘플 일괄 판독·결과표·CSV 다운로드를 사용할 수 있습니다. GLM은 이미지의 전처리 방법을 제안하고, 실제 문자열은 ZXing/libdmtx의 판독 결과로만 확정합니다.

## 실행

Python 3.12 환경 기준:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

브라우저에서 `http://localhost:8501`을 엽니다. API 설정이 없어도 로컬 판독은 동작합니다. `샘플` 탭에서 26개 이미지를 선택하고 `읽기`를 누르면 됩니다.

클립보드는 브라우저가 허용하는 localhost 또는 HTTPS 환경에서 사용합니다. 붙여넣은 이미지는 PNG로 보관해 재압축으로 코드가 손상되지 않게 했습니다.

## 사내 GLM 연결

`.env.example`을 `.env`로 복사하고 아래 빈칸을 채웁니다. 이 작업 폴더에는 빈 `.env`도 준비되어 있습니다.

```dotenv
GLM_BASE_URL=
GLM_MODEL=
GLM_API_KEY=
GLM_TIMEOUT_SECONDS=20
GLM_HEADERS_JSON='{}'
GLM_REASONING_EFFORT=
```

- `GLM_BASE_URL`: 사내 OpenAI 호환 API의 기본 주소. 예: `https://your-internal-api.example/v1`. `/chat/completions`까지 포함된 주소도 지원합니다.
- `GLM_MODEL`: 사내 게이트웨이의 정확한 모델 ID. 대상 모델의 예는 `glm-5.3-flash`이며 자동으로 다른 모델을 선택하지 않습니다.
- `GLM_API_KEY`: Bearer 인증값. 인증이 필요 없는 내부 API에서는 비워둡니다.
- `GLM_HEADERS_JSON`: 별도 인증 헤더가 필요한 경우 JSON 객체로 지정합니다.
- `GLM_REASONING_EFFORT`: 게이트웨이가 지원할 때만 지정합니다. 기본 요청에는 포함하지 않습니다.

앱과 `--glm` 벤치마크는 `.env`를 읽으며, 이미 설정된 환경변수가 우선합니다. `.env`는 Git과 Docker 빌드 대상에서 제외됩니다. 주소와 모델명이 비어 있으면 외부 API를 호출하지 않습니다.

설정 후 앱을 다시 실행하고 `GLM으로 이미지 분석`을 선택합니다. 일반 OpenAI 호환 `/chat/completions` 요청의 `messages[].content`에 PNG Base64 `image_url`을 보냅니다. 서버가 해당 모델의 이미지 입력을 지원해야 합니다. 대상 모델의 공개 명세는 [GLM-5.3-Flash 문서](https://docs.z.ai/guides/vlm/glm-5.3-flash), 메시지 형식은 [Chat Completion 문서](https://docs.z.ai/api-reference/llm/chat-completion)를 참고하세요.

처리 흐름:

1. 원본에서 바로 판독되면 API 호출을 생략합니다.
2. GLM이 ROI, 색상 채널, 블러, 임계값, 격자 크기를 JSON으로 제안합니다.
3. 허용된 범위의 전처리만 실행하고 실제 바코드 디코더로 확인합니다.
4. 제안으로 읽히지 않거나 API가 실패하면 로컬 색상 분리·잡음 제거·격자 복원 탐색을 이어갑니다.

GLM이 반환한 문자열이나 코드를 판독값으로 채택하거나 실행하지 않습니다. API 오류와 적용된 제안은 이미지별 상세 결과에서 확인할 수 있습니다. 로컬 판독 제한과 GLM HTTP 응답 제한은 별도이며, API 요청은 재시도하지 않습니다. 벤치마크에는 프로세스 단위 제한시간도 있습니다.

## 파일 역할

| 파일 | 역할 |
| --- | --- |
| `app.py` | 실제 Streamlit 진입점, 업로드·붙여넣기·결과 화면 |
| `reader.py` | 전처리와 디코더 실행 순서, GLM 제안 적용 및 로컬 fallback |
| `glm_assist.py` | API 호출·응답 검증·전처리 제안 실행 |
| `grid_recovery.py` | 불균일한 배경 제거, 관측한 테두리에 맞춰 격자 복원 |
| `backend.py` | OpenCV BGR 배열 입력과 기존 결과 딕셔너리 형태 연결 |
| `org_app.py` | `app.py`를 실행하는 호환 진입점 |
| `org_code_2.py` | `reader.py`의 공개 API를 제공하는 호환 진입점 |
| `org_code.py` | 기존 기본 함수명 `decode_image_from_array()`를 개선 엔진에 연결 |
| `reference/` | 제공한 세 파일의 수정 전 원본과 SHA256 목록 |

실행 시 필요한 엔진은 한 벌입니다. `reference/`는 앱에서 가져오지 않습니다. 호환 함수는 Data Matrix 판독을 대상으로 하며, 과거의 수동 전처리 목록·특정 디코더 강제 선택 인자는 지원하지 않는 경우 명시적인 오류를 반환합니다.

기존 BGR 호출 방식에 연결하려면:

```python
import cv2
from backend import decode_datamatrix_from_array

image = cv2.imread("sample.png")
result = decode_datamatrix_from_array(image, use_glm=True, max_seconds=12)
print(result["status"], result["barcodes"])
```

`barcodes`, `status`, `debug_variants`, `preprocessing_engine`, `decoder_engine` 키를 제공합니다. 숫자 코드는 문자열로 유지하므로 앞자리 0을 보존합니다.

## 검증

기존 저장소의 24개 이미지와 이번에 제공된 2개 이미지를 포함해 총 26개입니다.

| 실행 | 판독 및 기준 문자열 일치 |
| --- | --- |
| 수정 전 코드, 이미지당 30초 | 23/26 |
| 개선 코드, 이미지당 12초 | 26/26 |
| 모의 GLM API + 실제 로컬 디코더 | 26/26 |

`test_cases/manifest.json`의 기준값은 과거 판독 결과 또는 별도 전처리/격자 복원으로 교차 확인한 회귀 검사 기준입니다. 실물에서 독립적으로 제공된 정답 인증과는 구분합니다. 런타임 판독 코드는 이 파일이나 이미지 파일명으로 값을 조회하지 않습니다.

```bash
# 실제 이미지 전체 판독 및 문자열 일치 검사 (native libdmtx 없이도 가능)
.venv/bin/python scripts/benchmark_cases.py --max-seconds 12 --no-libdmtx

# 사내 API 없이 GLM 연결부 검증
.venv/bin/python scripts/check_glm_flow.py --max-seconds 12 --no-libdmtx

# 사내 설정 후 실제 GLM 호출을 포함한 검사
.venv/bin/python scripts/benchmark_cases.py --glm --max-seconds 12

# 자동 회귀검사
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

`check_glm_flow.py`는 localhost의 모의 API가 모든 이미지에 동일한 수동 전처리 제안을 반환하도록 합니다. 실제 HTTP 메시지 전달·JSON 처리·전처리·ZXing 판독·fallback을 검사합니다. **실제 GLM 추론이나 GLM의 이미지 판단 정확도를 측정한 결과는 아닙니다.** 사내 서버 연결·인증·모델 동작은 배포 환경에서 마지막 명령으로 확인해야 합니다.

검증 기록:

- `reports/current_baseline.json`: 수정 전 코드의 26개 결과
- `reports/improved_local.json`: 개선된 로컬 판독 결과
- `reports/improved_local_environment.json`: 실행 환경과 소스 해시
- `reports/glm_simulated_flow.json`: 모의 API 사용 여부와 경로별 결과

테스트는 새로 생성한 코드, 회전·반전, 빈 이미지·노이즈 거부, API 오류·시간 초과·잘못된 응답, BGR/RGB 채널 순서, 화면에서 이미지 교체 시 결과 갱신도 확인합니다.

## 선택 사항: native libdmtx

현재 26개 회귀 이미지는 ZXing과 격자 복원만으로 판독됩니다. 추가 이미지에서 libdmtx 보조 디코더를 사용하려면 OS 라이브러리를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y libdmtx0t64
```

배포판에 따라 패키지명은 `libdmtx0b`일 수 있습니다. 설치 후 `.venv/bin/python scripts/check_runtime.py`로 확인합니다. Java는 기존 `pyzxing` 연구용 스크립트에만 필요하며 기본 앱의 판독 경로에는 필요하지 않습니다.
