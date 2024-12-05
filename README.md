# Spotify 음악 추천 시스템

---

## **팀 구성**  
| 팀원 | 역할 |  
|------|------|  
| 조성지 | 팀장, mlflow 환경설정, 데이터 크롤링, 프론트엔드제작 |  
| 안서인 | 모델링 및 평가지표제작 |  
| 김태환 | 모델링 및 서빙 |  
| 조혜인 | 모델링 |  

---

## **프로젝트 개요**  
Spotify 오디오 특성 데이터를 기반으로 사용자 맞춤형 음악을 추천하는 콘텐츠 기반 추천 시스템을 개발합니다.

### **개발 환경**  
- **언어 및 도구:**  
  - Python 3.9  
  - Jupyter Notebook / VSCode 
- **라이브러리:**  
  - NumPy, pandas, scikit-learn, TensorFlow, Spotipy, mlflow, streamlit, fastapi, apache-airflow 

### **요구 사항**  
- Spotify API 액세스 키  
- Python 필수 라이브러리 설치  
- Spotify 오디오 특성 데이터셋  

---

## **타임라인**  
- **시작일:** 2024년 11월 25일  
- **최종 제출 마감일:** 2024년 12월 7일  

---

## **프로젝트 디렉토리 구조**  
```plaintext
├── code  
│   ├── jupyter_notebooks  
│   │   └── model_train.ipynb  
│   └── train.py  
├── docs  
│   ├── pdf  
│   │   └── (템플릿) [패스트캠퍼스] Upstage AI Lab 1기_그룹 스터디.pptx  
│   └── paper  
└── input  
    └── data  
        ├── eval  
        └── train  
```
---

## 3. 데이터 설명

### 데이터 개요

- 출처: Spotify API
- 구성: 약 10,000곡 이상의 오디오 특성 데이터 (Danceability, Energy, Tempo 등).

### EDA

1. 결측값 탐지 및 제거.
2. 오디오 특성 간 상관관계 분석.
3. 장르별 오디오 특성 분포 시각화.

### Data Processing

- 중복 제거 및 결측값 처리.
- 주요 특성을 기반으로 데이터 라벨링.
- 오디오 특성 값 표준화/정규화

## 4. 모델링

### 선택모델

- KNN
- 컨텐츠 기반 유사도
- TF-IDF 코사인 유사도

### 모델링 프로세스 

1. 음악 선택/아티스트 선택/현재 유저의 플레이리스트 선택
2. 추천 모델 학습 : 오디오 특성 기반 유사도 계산
3. 결과 검증 : 추천 음악의 유사성평가

## 5. 결과

### 리더보드

- feature 유사도
- f1 score 
- Human Evaluation

### 발표자료
: 링크

## 기타

### 회의록
: 링크

### 참고자료
- Spotify API 문서
- 추천 시스템 관련 논문 및 튜토리얼
