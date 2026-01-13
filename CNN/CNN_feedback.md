# CNN CIFAR-10 정확도 개선 피드백

## 정확도가 올라간 이유

epoch를 20에서 50으로 늘렸을 때 정확도가 **오히려 떨어졌던 문제**가 수정된 후, 정확도가 **엄청 올라갔습니다**. 
그 핵심 이유는 **2가지**입니다:

---

## 1️⃣ 정규화(Normalization) 적용

### 이전 코드
```python
X_test = torch.tensor(cifar_test.data).permute(0, 3, 1, 2).float().to(device)
```
- `cifar_test.data`는 **원본 이미지**(픽셀값 0~255) 그대로 사용
- 정규화 전처리(`ToTensor`, `Normalize`) **미적용**

### 새로운 코드
```python
test_loader = DataLoader(cifar_test, batch_size=256, shuffle=False)
```
- `DataLoader`가 `transform_test` 자동 적용
- `ToTensor`: 픽셀값을 0~1로 변환
- `Normalize`: 평균 0.49, 표준편차 0.20으로 정규화

### 영향
- **학습 때**: 정규화된 입력(값 범위 -1.5~2.0)으로 학습
- **평가 때 (이전)**: 정규화되지 않은 입력(값 범위 0~255) → **스케일 불일치** → 모델이 제대로 인식 못함
- **평가 때 (새로운)**: 정규화된 입력 → **학습과 동일한 스케일** → 모델이 정확하게 인식

---

## 2️⃣ Dropout 비활성화 (`model.eval()`)

### 이전 코드
```python
# model.eval() 없음
with torch.no_grad():
    prediction = model(X_test)
```
- 모델이 **학습 모드 유지**
- Dropout이 **활성화됨** → 30% 뉴런을 무작위로 제거하며 실행

### 새로운 코드
```python
model.eval()
with torch.no_grad():
    for X, Y in test_loader:
        logits = model(X)
        preds = logits.argmax(1)
model.train()  # 평가 후 학습 모드 복귀
```
- `model.eval()` 호출 → 평가 모드 전환
- Dropout **비활성화** → **모든 뉴런(100%) 사용**
- 최종 학습된 모델의 완전한 성능 발휘

### 영향
- **학습 중**: Dropout으로 과적합 방지 (뉴런 일부 무작위 제거)
- **평가 때 (이전)**: Dropout이 여전히 실행 → 불완전한 네트워크로 평가 → 낮은 정확도
- **평가 때 (새로운)**: 모든 뉴런 활용 → 최고 성능 발휘 → 높은 정확도

---

## 비교 요약

| 항목 | 이전 코드 | 새로운 코드 |
|------|---------|----------|
| **입력 정규화** | ❌ 안 됨 (0~255 범위) | ✅ 됨 (정규화 범위) |
| **Dropout 상태** | ❌ 활성화 (30% 제거) | ✅ 비활성화 (100% 사용) |
| **학습-평가 일관성** | ❌ 낮음 | ✅ 높음 |
| **정확도** | 🔴 낮음 | 🟢 높음 |

---

## 핵심 교훈

1. **입력 전처리 일관성이 중요**: 학습과 평가의 입력 분포가 다르면 정확도를 제대로 평가할 수 없습니다.
2. **`model.eval()` 필수**: 평가할 때는 반드시 평가 모드로 전환해 최종 학습된 모델의 성능을 테스트해야 합니다.
3. **이 두 가지 조합**: 정규화 + 평가 모드가 함께 작용하면서 정확도 개선이 눈에 띄게 나타났습니다.

---

## 다음 개선 방향

- Learning Rate Scheduling 추가 (예: `StepLR`, `ReduceLROnPlateau`)
- Early Stopping 구현 (validation 기반 최적 epoch 선택)
- 데이터 증강 강도 조정
- Weight Decay 값 최적화
- Batch Normalization 추가 고려
