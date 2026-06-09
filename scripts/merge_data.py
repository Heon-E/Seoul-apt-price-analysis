"""
기준금리 vs 서울 아파트 매매가격지수 - 데이터 병합 스크립트

이 스크립트가 하는 일:
1. data/base_rate.csv (한국은행 기준금리, 월별)
2. data/seoul_apt_price_index.csv (서울 아파트 매매가격지수, 월별)
두 파일을 'date'(YYYY-MM) 기준으로 합쳐서 data/merged.csv로 저장한다.
"""

import pandas as pd

# 1. 두 CSV 파일 불러오기
base_rate = pd.read_csv("data/base_rate.csv")
price_index = pd.read_csv("data/seoul_apt_price_index.csv")

# 2. 'date' 컬럼을 기준으로 병합 (inner join: 양쪽에 다 있는 날짜만)
merged = pd.merge(base_rate, price_index, on="date", how="inner")

# 3. 날짜순 정렬 (혹시 순서가 섞여있을 경우 대비)
merged = merged.sort_values("date").reset_index(drop=True)

# 4. 결과 확인 (터미널에 출력)
print("병합된 데이터 미리보기:")
print(merged.head())
print("...")
print(merged.tail())
print(f"\n총 {len(merged)}개월 데이터가 병합되었습니다.")

# 5. 새 파일로 저장
merged.to_csv("data/merged.csv", index=False)
print("\n저장 완료: data/merged.csv")
