"""
ECOS CSV 파일을 파싱하여 다중회귀분석용 데이터셋 생성
"""
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")


def parse_ecos_wide(filepath, data_row=1, meta_cols=4, value_name="value"):
    """
    ECOS에서 다운받은 가로형(wide) CSV를 세로형(long) Series로 변환합니다.
    - data_row  : 데이터가 있는 행 인덱스 (0=헤더, 1=첫번째 데이터행)
    - meta_cols : 날짜 열 이전의 메타데이터 열 수
    """
    df = pd.read_csv(filepath, header=0)
    row = df.iloc[data_row - 1]          # 원하는 데이터 행 선택
    date_cols = df.columns[meta_cols:]   # 날짜 열만 추출
    series = pd.to_numeric(
        row[date_cols].astype(str).str.replace(",", ""), errors="coerce"
    )
    series.index = pd.to_datetime(series.index, format="%Y/%m")
    series.name = value_name
    return series


def make_monthly_from_cumulative(series):
    """
    연간 누계 데이터를 월별 데이터로 변환합니다.
    1월 값은 그대로 사용, 2~12월은 전월 대비 차분
    """
    monthly = series.diff()
    jan_mask = series.index.month == 1
    monthly[jan_mask] = series[jan_mask]
    return monthly


def make_gov_dummy(date_index):
    df = pd.DataFrame(index=date_index)
    df["park_dummy"] = ((df.index >= "2015-01") & (df.index <= "2017-03")).astype(int)
    df["moon_dummy"] = ((df.index >= "2017-04") & (df.index <= "2022-03")).astype(int)
    df["yoon_dummy"] = ((df.index >= "2022-04") & (df.index <= "2025-10")).astype(int)
    return df


def main():
    print("=== 데이터 병합 시작 ===\n")

    # 1. 기존 데이터 (기준금리 + 서울 집값)
    print("① 기존 데이터 로드...")
    base = pd.read_csv(
        os.path.join(data_dir, "merged.csv"),
        parse_dates=["date"], index_col="date"
    )
    print(f"   → {len(base)}행\n")

    # 2. 소비자심리지수 (CSI)
    # csi.csv: 메타 컬럼 5개 (통계표, CSI코드, CSI분류코드, 단위, 변환)
    print("② 소비자심리지수(CSI) 파싱...")
    csi = parse_ecos_wide(
        os.path.join(data_dir, "csi.csv"),
        data_row=1, meta_cols=5, value_name="csi"
    )
    print(f"   → {len(csi)}개 관측치\n")

    # 3. 주택담보대출 금리
    # mortgage_rate.csv: 메타 컬럼 4개, 첫 번째 행이 주택담보대출 전체
    print("③ 주택담보대출 금리 파싱...")
    mortgage_rate = parse_ecos_wide(
        os.path.join(data_dir, "mortgage_rate.csv"),
        data_row=1, meta_cols=4, value_name="mortgage_rate"
    )
    print(f"   → {len(mortgage_rate)}개 관측치\n")

    # 4. 주택 인허가 (서울, 연간 누계 → 월별 변환)
    print("④ 주택 인허가(서울) 파싱 및 월별 변환...")
    permit_cum = parse_ecos_wide(
        os.path.join(data_dir, "house_permit.csv"),
        data_row=1, meta_cols=4, value_name="housing_permit"
    )
    housing_permit = make_monthly_from_cumulative(permit_cum)
    print(f"   → {len(housing_permit)}개 관측치\n")

    # 5. 정권 더미변수
    print("⑤ 정권 더미변수 생성...")
    gov_dummy = make_gov_dummy(base.index)
    print(f"   → 박근혜: {gov_dummy['park_dummy'].sum()}개월")
    print(f"   → 문재인: {gov_dummy['moon_dummy'].sum()}개월")
    print(f"   → 윤석열: {gov_dummy['yoon_dummy'].sum()}개월\n")

    # 6. 병합
    print("⑥ 전체 데이터 병합...")
    df = base.copy()
    df = df.join(csi, how="left")
    df = df.join(mortgage_rate, how="left")
    df = df.join(housing_permit, how="left")
    df = df.join(gov_dummy, how="left")

    print(f"   → 최종: {df.shape[0]}행 x {df.shape[1]}열")
    print(f"   → 컬럼: {list(df.columns)}\n")

    # 결측치 확인
    print("=== 결측치 현황 ===")
    print(df.isnull().sum())

    # 저장
    out_path = os.path.join(data_dir, "merged_multi.csv")
    df.to_csv(out_path)
    print(f"\n저장 완료: {out_path}")
    print("다음 단계: python scripts/regression_multi.py")


if __name__ == "__main__":
    main()
