# 선행연구와 구현을 구분한 기록

| 연구 | 실제 출판처·연도 | 이번 구현에서 참고한 것 | v1 구현 여부 |
|---|---|---|---|
| A cost-sensitive transformer model for prognostics under highly imbalanced industrial data | Cluster Computing, 2026 | 미탐·오탐 비용이 비대칭이라는 문제 정의, SECOM 평가 | 비용 시나리오만 자체 구현, 논문 모델 미재현 |
| Accurate predictions on small data with a tabular foundation model | Nature, 2025 | 작은 표형 데이터의 최신 비교 모델 후보 | TabPFN 미실행 |
| A multi-objective feature selection and self-paced ensemble framework for semiconductor defect detection | APEM, 2025 | 고차원·불균형·중첩을 함께 보는 관점 | MOFS-SPE 미재현 |
| From synthetic data to actionable insights: fault detection in industry | Complex & Intelligent Systems, 2026 | 소수 클래스 생성과 설명 가능성의 확장 방향 | VAE–SHAP 미재현 |
| TabM: Advancing tabular deep learning with parameter-efficient ensembling | ICLR, 2025 | 효율적 신경망 앙상블 비교 후보 | TabM 미실행 |

공식 원문:

- https://doi.org/10.1007/s10586-026-06080-2
- https://doi.org/10.1038/s41586-024-08328-6
- https://doi.org/10.14743/apem2025.4.552
- https://doi.org/10.1007/s40747-026-02323-2
- https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html

공개 원문/초록을 통해 연구 주제와 적용 방향을 확인한 참고 목록이다. 모든 결과표·부록을 재현 검증한 문헌 고찰은 아니다.
최신 논문의 높은 점수를 v1의 목표치나 성능 보장으로 사용하지 않는다. 분할, 양성 클래스, 전처리 위치, 지표 정의,
튜닝 예산이 다르면 수치를 직접 비교할 수 없다.
