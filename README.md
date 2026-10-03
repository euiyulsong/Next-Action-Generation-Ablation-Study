# Next Action 생성 방식 비교 — 50개 샘플 결과 정리

## 1. 실험 목적

이 실험의 목적은 **LLM 답변 뒤에 다음 행동을 어떻게 추천해야 사용자 입장에서 자연스럽고 유용한가**를 비교하는 것이다.

핵심 비교축은 3개다.

- **언제 next action을 생성하는가**
  - 답변과 동시에 생성
  - 답변을 다 만든 뒤 생성
  - 답변을 보지 않고 query만 보고 생성
- **action space를 얼마나 제한하는가**
  - 완전 자유 생성
  - fixed taxonomy
  - fixed coarse action + free detail
- **추천 여부 판단을 별도 gate로 분리하는가**

---

## 2. 각 method가 정확히 뭔가

| Method | 구조 | 설명 |
|---|---|---|
| `joint_free` | `Query → Answer + Next Action` | 답변과 next action을 **한 번의 generation에서 같이 생성** |
| `cont_free` | `Query → Answer → Next Action` | 답변을 완성한 뒤, **그 답변을 읽고 자유롭게 next action 생성** |
| `query_only_free` | `Query → Next Action` | 답변을 보지 않고 **원 질문만 보고 next action 생성** |
| `cont_fixed` | `Answer → Fixed Action → Transition` | 답변 후 `SEARCH / DEBUG / COMPARE ...` 같은 **고정 action taxonomy 중 하나 선택** |
| `cont_hybrid` | `Answer → Coarse Action + Free Detail → Transition` | coarse action은 고정하되 실제 행동 설명은 자유 생성 |
| `two_stage_free` | `Answer → Should Suggest? → Transition` | 먼저 별도 gate가 추천 여부를 판단하고, YES일 때만 문구 생성 |

조금 더 직관적으로 보면:

```text
joint_free
사용자 질문
   ↓
LLM
   ↓
답변 + "다음엔 이것도 해볼 수 있음"
```

```text
cont_free
사용자 질문
   ↓
답변
   ↓
답변을 다시 보고
   ↓
"아직 자연스럽게 이어질 다음 단계가 있나?"
```

```text
query_only_free
사용자 질문
   ↓
답변을 보지 않고
   ↓
"다음엔 뭐 하면 좋지?"
```

```text
cont_fixed
답변
   ↓
SEARCH / DEBUG / COMPARE / ...
   ↓
사용자용 문구
```

```text
cont_hybrid
답변
   ↓
DEBUG
+
"GPU idle time과 DataLoader wait 확인"
   ↓
자연어 문구
```

```text
two_stage_free
답변
   ↓
추천할 필요 있음?
   ↓
NO → 끝
YES → 자연어 next action 생성
```

---

# 3. 전체 수치

| Method | Suggestion Rate | 추천 개수 | Action 다양성 | Action-stage Latency |
|---|---:|---:|---:|---:|
| `joint_free` | 10% | 5/50 | 6 | 3.94s |
| `cont_free` | **28%** | **14/50** | 15 | 1.90s |
| `query_only_free` | **44%** | **22/50** | **23** | 14.22s |
| `cont_fixed` | 4% | 2/50 | 2 | 1.81s |
| `cont_hybrid` | 2% | 1/50 | 2 | 1.81s |
| `two_stage_free` | 0% | 0/50 | 1 | 1.90s |

추천 빈도만 보면:

```text
query_only_free   44%
cont_free         28%
joint_free        10%
cont_fixed         4%
cont_hybrid        2%
two_stage_free     0%
```

하지만 **추천을 많이 한다고 좋은 게 아니다.**

중요한 건:

> 필요한 곳에서는 추천하고, 필요 없는 곳에서는 조용히 끝내는가?

이다.

---

# 4. 실제 출력 기준으로 보면 `cont_free`가 가장 균형이 좋음

`cont_free`가 추천한 14개를 보면 실제로 대화가 이어질 이유가 있는 케이스가 많았다.

대표적인 예:

| 상황 | 생성된 next action 성격 | 평가 |
|---|---|---|
| Amalfi 5일 여행 일정 | 여행 월/숙소를 받아 ferry·crowd 기준으로 개인화 | 매우 좋음 |
| Europe 여행 planning | 날짜/기간/예산/선호지를 받아 다음 planning 진행 | 매우 좋음 |
| Wales 음식 추천 | 어느 지역인지 받아 현지 음식/장소 좁힘 | 매우 좋음 |
| haircut 고민 | guard length / grow-out plan으로 구체화 | 매우 좋음 |
| cookie recipe | 몇 명 먹을지 받아 batch 계산 | 좋음 |
| hexdump 분석 | 더 긴 raw bytes를 받아 byte order 검증 | 매우 좋음 |
| sleep 문제 | 깨어 있는 원인을 받아 다음 advice를 개인화 | 매우 좋음 |
| crypto system design | 다음 단계로 measurable spec/budget 구체화 | 좋음 |

즉 `cont_free`는 단순히:

> "더 알려드릴까요?"

를 붙이는 게 아니라,

**현재 답변 이후 실제로 남아 있는 정보 gap이나 실행 단계**를 잘 잡는 편이었다.

---

# 5. `query_only_free`는 recall은 높아 보이지만 과잉 추천이 많음

`query_only_free`는 22/50으로 가장 많이 추천했다.

문제는 **답변을 못 보고 next action을 생성한다는 것**이다.

그래서 실제 결과에서 이런 현상이 생겼다.

### 예: juggling

사용자:

> How can I learn to juggle?

`query_only_free`가 생성한 내용은:

> soft balls로 연습하고 한 개씩 arc로 던져라...

이런 식이었다.

이건 next action이 아니라 **원래 answer 본문에 들어가야 할 내용**이다.

---

### 예: Zelda

사용자:

> 첫 Zelda 게임은 어떤 내용이야?

`query_only_free`는 Ganon, Hyrule 같은 내용을 추가 설명했다.

이 역시:

```text
next action
```

이라기보다

```text
answer continuation
```

에 가깝다.

---

### 예: guitar vs bass

`query_only_free`:

> 밴드에서 어떻게 들리는지나 어느 게 입문하기 쉬운지도 비교해드릴 수 있다.

문장 자체는 자연스럽지만, 사용자는 단순히 차이를 물었다.

답변이 이미 충분했다면 굳이 추가 제안을 붙이지 않아도 된다.

---

그래서 실제 패턴은:

```text
query_only_free
→ Recall 높음
→ Precision 낮음
→ 대화를 너무 자주 이어가려는 경향
```

으로 보인다.

---

# 6. `joint_free`는 반대 문제

`joint_free`는 5/50만 추천했다.

좋은 점은 **생성된 문장 자체는 자연스럽다**는 것.

답변과 같은 generation에서 만들기 때문에 tone이나 흐름이 잘 맞을 수 있다.

하지만 실제 결과를 보면:

- 여행 planning
- haircut
- sleep
- technical debugging
- personalization

같이 **자연스러운 다음 턴이 있는 경우에도 NONE**이 많았다.

즉:

```text
joint_free
→ Precision 높아 보임
→ Recall 낮음
```

이다.

대화형 assistant 입장에서는 조금 **너무 조용한 assistant**가 될 수 있다.

---

# 7. `cont_fixed` / `cont_hybrid`는 거의 collapse

결과:

```text
cont_fixed   2 / 50
cont_hybrid  1 / 50
```

즉 action taxonomy가 들어가자 거의 대부분:

```text
NONE
```

으로 빠졌다.

이건 꽤 중요한 결과다.

예를 들어 free 상태에서는:

> 여행 월과 숙소 알려주면 ferry 일정에 맞춰 조정 가능

같은 자연스러운 continuation이 나오는데,

fixed taxonomy를 주면 모델이 내부적으로:

> 이건 PLAN인가? CLARIFY인가? TOOL인가? 굳이 하나를 고를 만큼 명확한가?

를 판단하면서 threshold가 높아진다.

그래서:

```text
자연스러운 conversation continuation
```

을 만들고 싶은 기능에서는 **action enum을 먼저 강제하는 게 오히려 recall을 죽일 수 있다.**

---

# 8. `two_stage_free`는 현재 gate가 너무 강함

결과:

```text
0 / 50
```

전부 NONE.

즉 architecture 문제라기보다 현재 prompt가:

```text
HIGH PRECISION
clear
materially useful
non-redundant
```

같은 조건을 너무 강하게 줘서 모델이 전부 suppress한 것이다.

따라서 이번 결과로:

> two-stage가 나쁘다

라고 볼 수는 없고,

정확히는:

> **현재 gate threshold는 production에 쓰기엔 지나치게 conservative하다**

가 맞다.

---

# 9. 사용자 입장에서 보면 어떻게 느껴지나

| Method | 사용자 체감 |
|---|---|
| `query_only_free` | 친절하지만 자꾸 말을 걸려고 함 |
| `cont_free` | **필요할 때만 자연스럽게 다음 턴을 열어줌** |
| `joint_free` | 깔끔하지만 대화를 이어갈 기회를 자주 놓침 |
| `cont_fixed` | 너무 보수적 |
| `cont_hybrid` | 거의 기능이 안 보임 |
| `two_stage_free` | 아예 추천 기능이 없는 것과 비슷 |

대화 UX에서 가장 중요한 건:

```text
추천 횟수 최대화
```

가 아니라

```text
자연스럽게 이어질 때만 이어주는 것
```

이다.

이 기준에서는 실제 50개 출력상 `cont_free`가 가장 자연스럽다.

---

# 10. Precision / Recall 관점

정확한 Recall은 gold label이 없어서 아직 계산할 수 없다.

하지만 정성적으로 보면:

```text
                 Precision      Recall
query_only_free      중간↓        높음
cont_free            높음         높음
joint_free           매우 높음     낮음
fixed/hybrid         높음         매우 낮음
two_stage            -            0
```

정도로 보인다.

특히 `cont_free`가 잡은 14개 중 대략 10~12개는 실제 서비스에서도 충분히 노출할 만한 continuation으로 보였다.

거칠게 정성 precision proxy를 잡으면:

```text
10~12 / 14
≈ 71~86%
```

정도.

반면 `query_only_free`는 더 많은 positive를 잡지만 closed QA에도 자주 추천을 붙였다.

---

# 11. Action diversity 결과도 의미 있음

| Method | Unique action metadata |
|---|---:|
| `query_only_free` | 23 |
| `cont_free` | 15 |
| `joint_free` | 6 |
| `cont_fixed` | 2 |
| `cont_hybrid` | 2 |
| `two_stage_free` | 1 |

Free generation에서는 다양한 next action을 잘 발견한다.

하지만 이걸 그대로 routing key로 쓰면:

```text
RUN_TEST
RUN_EVALUATION
EVALUATE_RESULT
COMPARE_RESULT
VERIFY_RESULT
```

같이 taxonomy가 난립할 수 있다.

그래서 production에서는:

```text
User-facing transition
→ 자유 생성
```

하면서도 내부 metadata만 나중에:

```text
CLARIFY
EXPLORE
EXECUTE
COMPARE
SEARCH
```

정도로 coarse하게 normalize하는 구조가 더 적합해 보인다.

---

# 12. Production 구조로 가져간다면

이번 결과에서 가장 합리적인 baseline은:

```text
User Query
    ↓
Main Answer
    ↓
Free Continuation Policy
    ↓
should_suggest?
   /       \
 NO        YES
 ↓          ↓
END     natural transition
```

즉 `cont_free`.

출력은 예를 들면:

```json
{
  "should_suggest": true,
  "transition": "여행 월과 숙소가 정해졌으면 ferry 일정과 혼잡도까지 반영해서 더 현실적인 일정으로 다듬을 수 있어."
}
```

정도로만 두고,

필요하면 뒤에서:

```text
transition
   ↓
internal classifier
   ↓
CLARIFY / PLAN / SEARCH / ...
```

를 붙이는 게 좋아 보인다.

---

# 13. 최종 결론

이번 50개 실제 결과에서 가장 중요한 발견은 3개다.

### ① Answer를 보고 next action을 만드는 게 중요함

```text
query_only_free 44%
→ 많이 추천하지만 과잉/중복이 많음

cont_free 28%
→ 답변에서 이미 해결된 것은 제거하고
   남은 task만 추천
```

그래서 완전 병렬 생성보다는 **answer-conditioned continuation**이 더 적합해 보인다.

### ② Action taxonomy를 먼저 강제하면 conversational recall이 크게 떨어짐

```text
cont_fixed   4%
cont_hybrid  2%
```

자연스러운 follow-up 기능에는 너무 restrictive했다.

### ③ 현재 가장 좋은 UX candidate는 `cont_free`

```text
Answer
↓
Answer를 보고
↓
필요할 때만
↓
자유로운 자연어 continuation
```

이 구조가 실제 출력에서 **precision, recall, 자연스러운 대화 흐름의 균형이 가장 좋았다.**

한 줄 요약하면:

> **이번 데이터에서는 `query_only_free`는 말을 너무 많이 하고, `joint_free`는 말을 너무 적게 하며, `cont_free`가 “필요할 때만 자연스럽게 한 번 더 이어주는” 가장 균형 잡힌 방식으로 보인다.**
