# Decision Head Training — 학습 구조와 실무 노트

> 이 프로젝트의 decision head(결정 헤드) 학습 방식과, 실제 학습/디버깅 과정에서
> 발견한 함정들을 정리한 문서. (2026-10-09, Breakout paddle-direction 작업 중 검증)

## 1. 전체 구조: 백본은 얼리고, 헤드만 학습

```
Qwen3.5-0.8B (GGUF, llama-server)          ← 절대 수정 안 됨 (read-only)
   │  임베딩 추출 (1024-dim, last-token pooling)
   ▼
data/features_*.npz  (frozen 상태 임베딩 캐시)
   │
   ▼
Decision Head (MLP, 학습 대상)
   ├─ TypedDecisionHead   head_trained.pt  — 질문별 고정 출력 헤드
   └─ DynamicDecisionHead head_dynamic.pt  — option을 입력으로 받는 어텐션 헤드
```

- 백본은 그래디언트를 전혀 받지 않는다. 학습은 사전 추출된 **frozen 임베딩 위에서만** 일어난다.
- 헤드 파라미터는 작다 (~100K–400K). 그래서 학습이 빠르고, 백본 교체 없이 실험 반복이 가능하다.
- 대가: 헤드는 백본 임베딩이 담지 못하는 정보를 만들어낼 수 없다. held-out 템플릿 성능 하락의 근본 원인 (README 일반화 섹션 참고).

### DynamicDecisionHead의 어텐션 방식

option을 아키텍처에 박지 않고 **입력으로** 받는다:

```
상태 임베딩 → trunk → query (d_k=128)
option 텍스트들 → 백본 임베딩 → key projection → keys (n_opts × d_k)
logits = query · keys^T / √d_k  →  softmax → argmax
```

- option 개수가 학습과 무관하게 자유로움 (재학습 없이 새 선택지 추가 가능).
- 대신 option **문자열 자체**가 학습 어휘에 없으면 스코어가 무의미해진다
  → "left/right/stay" 편향 사건의 원인 (§5 참고).
- 온도(T)는 학습 후 캘리브레이션 홀드아웃에서 LBFGS로 전역 1개를 피팅해 체크포인트에 저장.

### v2: 필드 세트 cross-attention (2026-10-09)

v1의 근본 한계 — 상태 텍스트 전체가 **풀링된 단일 벡터** 1개로만 헤드에 들어감 — 을
풀기 위해 상태를 **필드/문장 단위 임베딩 세트**로 바꾸고 option이 그 세트에
cross-attention하게 재구성했다 (checkpoint arch `state_set: true`):

```
필드 세트 (B, M, 1024) ──field_enc (공유)──→ F (B, M, H)
option (N, 1024) ──opt_query──→ q (B, N, d_k)
필드 ──field_key──→ K (B, M, d_k)
α_ij = softmax_i(q_j·k_i/√d_k)   ← 필드 축, 마스크된 패딩 필드 제외
z_j = Σ_i α_ij·F_i → score_head(z_j) → logits (B, N)
```

- **오염 국소화**: OOD 토큰 하나가 상태 벡터 전체가 아니라 그 토큰이 속한 필드만
  오염시킨다. JSON/로그 같은 비-prose 포맷은 필드 단위로 쪼개면 문법 토큰(`{`, `"`,
  `=`)에 기하 정보가 희석되지 않는다.
- **직접 매칭**: 헤드가 "공이 오른쪽이라는 문장" ↔ "right" 옵션을 필드 수준에서
  매칭할 수 있다.
- **요약 필드**: 전체 텍스트 임베딩을 field 0으로 항상 prepend
  (`dynamic_head.include_summary_field: true`). M≥1 보장 + 전역 컨텍스트.
- **noul 정리**: v1에 있던 전용 `noul_head`는 한 번도 학습되지 않은 죽은 경로였다
  (학습/서빙은 둘 다 attention 경로). v2에서 noul은 `["false","true"]` 옵션 어텐션
  경로로 통일 — 학습·서빙·벤치마크가 같은 경로를 쓴다.
- **청킹 일관성 불변식 (가장 중요)**: 학습 시 필드 분할과 추론 시 필드 분할이
  같아야 한다. 단일 진실원천은 `states/fields.py`의 `state_fields` /
  `split_state_fields` (JSON→leaf, 줄바꿈, `KEY=VALUE` 토큰, 문장 순으로 시도,
  16필드 컷)이고, 피처 추출(`backbone/features.py:extract_field_features`)과
  서빙(`webapp/agents.py:_state_tensor`)이 둘 다 이 함수를 경유한다. Breakout
  렌더러는 필드를 직접 내보내지만(`(text, fields)`), **휴리스틱 분할 결과가
  렌더러 필드와 정확히 일치함을 테스트로 강제한다**
  (`tests/test_fields.py::TestBreakoutChunkingConsistency`) — 그래서 HTTP
  `custom_fields`를 생략한 caller도 학습과 같은 청킹을 얻는다.
- **캐시 형식**: `features_{split}_fields.npz` — ragged 저장
  (`features (ΣM_i, 1024)` + `field_counts (N,)`). 기존 pooled
  `features_*.npz`는 typed 헤드가 그대로 쓴다.
- **하위 호환**: v1 체크포인트는 그대로 로드된다 (arch 플래그 분기,
  `models/head_dynamic_v1.pt`에 백업 보관). v1 헤드에 3D 필드 세트를 넣으면
  field 0(=요약=전체 텍스트 임베딩)을 읽어 구 동작과 동일하다.

## 2. Last-token pooling — 가장 중요한 함정

Qwen 같은 causal 모델의 문장 임베딩은 **마지막 토큰의 hidden state** 하나다.

```
"The ball is LEFT of the paddle."   (causal LM)

토큰:    [The] [ball] [is] [LEFT] [of] ... [paddle] [.]
hidden:  h1     h2    h3   h4    h5   ...  h9     h10  ← 이 것만 사용
```

가능한 이유: causal 어텐션에서 마지막 토큰은 앞의 모든 토큰을 이미 봤기 때문에
문장 전체 정보가 거기에 누적되어 있다. `configs/default.yaml`의 `model.pooling: "last"`가 이 방식.

### 대안 비교

| 방식 | 방법 | 특징 |
|---|---|---|
| **last** | 마지막 토큰 hidden state | causal 모델에 자연스러움, 패딩 불필요. **끝 정보가 가장 강함** |
| mean | 전체 토큰 평균 | 고르게 섞이지만 위치 편향 가능 |
| BERT류 | [CLS] 토큰 | 양방향 어텐션이라 첫 토큰에 정보가 모임 |

### 왜 문제가 되었나 (실제 사례)

마지막 토큰도 결국 **하나의 위치**라서, 앞 정보를 어텐션으로 담는 정도가
균등하지 않고 직전 내용이 상대적으로 강하게 남는다.

1차 Breakout 학습 때 템플릿을 "게임 상태 먼저, 기하학(볼 위치) 마지막"으로
만들었더니 헤드가 **텍스트 끝에서 볼 위치를 읽도록** 학습됨. 실제 게임이 보내는
상태는 기하학이 앞에 있고 끝에 motion 구문이 오는 배치라 → right/stay 상태를
못 맞춤. 템플릿이 문장 순서를 랜덤화하고 상태 문장을 생략하는 경우도 포함하도록
고친 뒤 95.5% 도달.

**교훈**: last-token pooling으로 학습할 때는 "결정적 정보가 텍스트 어디에
오는가"가 학습 성패를 가린다. 학습 데이터의 문장 배치가 실제 서비스 입력의
배치와 다르면 조용히 실패한다. 반대 방향(학습 데이터를 실제 입력에 맞추는 것)도
한 가지 레이아웃에만 맞추면 서로 다른 클라이언트에서 깨진다 — **둘 다 커버하도록
학습 데이터를 다양화**하는 것이 답.

## 3. 학습 데이터 설계 원칙

생성기는 라틴트(latent) 변수에서 문서를 렌더링하며, **라틴트가 곧 정답(gold)이다** —
별도 라벨링 과정이 없다 (`states/generator.py`, `states/breakout.py`).

### Decorrelation (지름길 차단)

표면 단어가 정답을 힌트하지 않도록 변수 간 독립성을 강제한다. Breakout 예:

- 볼의 속도 방향("moving right", "away from the paddle")을 볼의 위치와
  **독립적으로** 샘플링 → gold=left 상태에도 "moving right"가 등장.
- bricks/score/lives는 순수 잡음(디스트랙터).
- gold 라벨은 샘플링 시점에 균등하게(1/3씩) 선택 → 클래스 균형이 구조적으로 보장.

### 템플릿 커버리지

- 하나의 표준 레이아웃만 쓰면 위치 편향이 생긴다 (§2 사례).
- 방향 단어가 전혀 없는 템플릿(JSON 좌표, 부호 있는 로그)을 포함해
  "좌표 → 방향" 추론을 강제한다.
- 학습 전 테스트로 검증: `tests/test_breakout.py`의 decorrelation /
  no-direction-word / 문장 순서 커버리지 테스트.

## 4. 동적 헤드 학습 파이프라인

```bash
python -m decision_lab generate      # 문서 + breakout JSONL (동일 seed면 문서는 byte-identical)
python -m decision_lab extract       # llama-server로 상태 임베딩 → features_*.npz 캐시
python -m decision_lab train-dynamic # 샘플 생성 + 커리큘럼 학습 + 온도 캘리브레이션
python -m decision_lab eval && python -m decision_lab report
```

- **Curriculum (anchor → generalize)**: 앞 구간은 기본 질문 뱅크(anchor)만,
  뒤 구간부터 option 변형(shuffle/동의어)을 섞어 permutation 불변성을 학습.
- **메모리**: 샘플을 한 번만 패킹하고 단계별로 인덱싱 (3번 패킹하던 구조는
  198k 샘플에서 ~8GB 피크). 현재 ~5GB (MPS).
- **Split 위생**: 학습은 train split에서만. 예전 코드는 generalize 단계에서
  val/calibration 샘플까지 학습해 val_acc를 부풀렸다 — 수정됨.
  이 저장소는 "정직한 숫자"를 원칙으로 한다 (벤치마크 벽 버그 이력 참고).

### Mixed-domain 학습 (Breakout 추가 방식)

- Breakout은 **동적 헤드에만** 혼합 (`breakout_weight: 3`). Typed 헤드는
  `encode_labels`가 문서 상태에 없는 qid에서 KeyError를 내므로 절대 섞지 않는다.
- 문서 JSONL은 동일 seed에서 byte-identical (breakout은 `Random(seed + 1)`) →
  문서 특징 캐시 유효 유지, `head_trained` 수치 불변.
- 회귀 게이트: 혼합 후 in-dist 정확도가 사전 값 ±2pp 이내, 표당 손실 없음.

## 5. 운영 gotchas

- **서버는 체크포인트를 시작 시 캐시한다.** `python -m decision_lab ui`가
  `models/head_dynamic.pt`를 한 번 로드함 (`webapp/agents.py`의
  `_load_dynamic_agent`). 재학습 후 **서버를 재시작하지 않으면** HTTP 요청이
  조용히 stale 헤드를 서빙한다.
- **`question` 문자열은 메타데이터** — 모델에는 상태 텍스트와 option 문자열만
  전달됨 (`docs/http-api.md`). 질문별 의미는 상태 텍스트에 담아야 한다.
  질문 조건화 임베딩은 양쪽(학습 캐시 재추출 + 추론)을 바꿔야 해서 연기됨.
- **학습 안 본 도메인의 option은 신뢰 불가** — README의 novel-option 일반화
  표 (별 태그 42%, 은유 라벨 7%).

## 6. 진단 도구

- `scripts/quick_breakout_test.py` — 사용자의 실패 상태 재현 + 셔플된 option
  순서 강건성 + test_breakout 랜덤 20개 눈으로 확인.
- `scripts/quick_one_state_test.py` — 타입 헤드 + 프롬프트 LM 1-state 점검.
- 혼동 행렬 / 라벨별 정확도: `results/report.md`의 Breakout 섹션
  ("항상 right" 같은 붕괴는 left/stay 라벨 정확도 0으로 바로 드러난다).

## 7. 파일 지도

| 역할 | 파일 |
|---|---|
| 문서 상태 생성기 | `src/decision_lab/states/generator.py` |
| Breakout 상태 생성기 | `src/decision_lab/states/breakout.py` |
| 필드 분할 (청킹 단일 진실원천) | `src/decision_lab/states/fields.py` |
| 동적 헤드 모델/디코딩 (v1/v2) | `src/decision_lab/head/dynamic_model.py` |
| 동적 헤드 학습 (커리큘럼, 온도) | `src/decision_lab/head/dynamic_train.py` |
| 피처 추출 (pooled + ragged 필드) | `src/decision_lab/backbone/features.py` |
| 동적 헤드 서빙 | `src/decision_lab/webapp/agents.py` |
| 벤치마크 (혼동 행렬 포함) | `src/decision_lab/eval/benchmark.py` |
| HTTP API 문서 | `docs/http-api.md` |
