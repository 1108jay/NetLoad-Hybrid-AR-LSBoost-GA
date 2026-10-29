# NetLoad-Hybrid-AR-LSBoost-GA
Hybrid AR-Lasso-LSBoost Net Load Forecasting with Genetic Algorithm in Jeju


# ⚡ 제주 지역 순부하(Net Load) 예측 융합 모델
> **Hybrid AR-Lasso-LSBoost 아키텍처 및 유전 알고리즘(GA) 하이퍼파라미터 최적화**

신재생에너지 침투율이 높은 제주 전력망의 '덕 커브(Duck Curve)' 현상에 대응하기 위해, 전통적 계량 시계열 모형(AR)과 기계학습(LSBoost)을 결합하여 순부하(Net Load)를 정밀 예측한 프로젝트입니다.

---

## 📌 핵심 성과 (Key Highlights)
* **15.44%p 오차율 급감:** 단일 AR 베이스라인(MAPE 21.70%) 대비 하이브리드 제안 모형(MAPE 6.26%)으로 예측 정확도 대폭 개선
* **경량화된 연산 효율성 입증:** 직접 구현한 유전 알고리즘(GA)을 통해 내장 베이지안 최적화 대비 절반 이하의 트리(499개 → 233개)로 오차 추가 개선 (RMSE 54.44 MW)
* **이중 과적합 방지 설계:** 무작위 분할을 배제한 시계열 순차 분할(2022~2024 Train, 2025 Test) 및 GA 내부 독립 검증셋(2024) 격리

---

## 🛠 아키텍처 구조 (3-Phase Hybrid)
1. **Phase 1 (선형 관성 통제):** ACF/PACF 분석 기반 지연 변수($Y_{t-24}, Y_{t-168}$)를 활용한 AR 베이스라인 적합 및 비선형 잔차(Residual) 추출
2. **Phase 2 (변수 선택 및 노이즈 제어):** 
   * 푸리에 변환 기반 24시간 주기 파동 변수(`SinHour`, `CosHour`)
   * 3시간 이동평균 기반 기상 평활화 변수(`Temp_MA3`, `Insol_MA3`)
   * 10-Fold 교차 검증 Lasso 정규화를 통한 17개 입력 변수 유의성 확정 (17/17 선정)
3. **Phase 3 (비선형 잔차 학습 & GA 최적화):**
   * LSBoost 앙상블 트리 모형으로 기상 이변에 따른 비선형 스파이크 예측
   * 룰렛 휠 선택 및 돌연변이 연산이 적용된 유전 알고리즘(GA) 직접 구현

---

## 📊 모델 성능 비교 (2025 Out-of-Sample)

| Model | Hyperparameters (Trees / LearnRate) | RMSE (MW) | MAPE (%) |
| :--- | :---: | :---: | :---: |
| **Benchmark (AR 단일)** | - | 159.67 | 21.70% |
| **Model A (Bayesian Hybrid)** | 499 Trees / 0.3600 | 56.90 | 6.57% |
| **Model B (GA Hybrid)** | **233 Trees / 0.2426** | **54.44** | **6.26%** |

---

## 📂 파일 구성
* `Final_Assignment2.m`: 전체 분석 파이프라인 및 GA 직접 구현 MATLAB 원본 코드
* `Final_mlx_ver_Assignment2.mlx`: 연산 결과 및 시각화 그래프가 포함된 MATLAB Live Script
* `Final_Master_Table_for_MATLAB.csv`: 2022~2025년 4개년 시계열 통합 마스터 데이터셋 (69,958 rows)
