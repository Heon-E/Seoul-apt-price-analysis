"""
주택준공 CSV 3개 파일 파싱 → 서울 민간 월별 준공 건수 추출
(민간임대 + 민간분양 합산)
"""
import pandas as pd
import re
import os

DATA_DIR = os.path.expanduser(
    "~/Desktop/개인 프로젝트 폴더/analysis/seoul house pricing/data"
)
FILES = [
    "주택건설실적통계(준공)_주택건설 준공실적(월계) (201007 ~ 202604).csv",
    "주택준공.csv",
    "주택완공.csv",
]

def parse_file(filepath):
    rows = []
    with open(filepath, encoding="cp949") as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            date   = re.sub(r"[^0-9\-]", "", parts[0])
            gubun  = parts[1]
            sector = parts[2]
            region = parts[3]
            num_str = "".join(parts[4:]).replace(",", "")
            try: value = int(num_str)
            except: value = None
            if len(date) == 7:
                rows.append({"date": date, "gubun": gubun, "sector": sector,
                             "region": region, "completion": value})
    return pd.DataFrame(rows)

dfs = []
for fname in FILES:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        dfs.append(parse_file(fpath))
        print(f"{fname} 파싱 완료")

combined = pd.concat(dfs, ignore_index=True)

# 서울 + 민간(임대+분양) 필터링 후 월별 합산
seoul = combined[
    (combined["region"] == "서울") &
    (combined["sector"].isin(["민간임대", "민간분양"]))
].copy()

seoul["date"] = pd.to_datetime(seoul["date"])
seoul = (
    seoul.groupby("date")["completion"]
    .sum()
    .reset_index()
    .sort_values("date")
    .drop_duplicates("date")
    .set_index("date")
)
seoul = seoul.loc["2015-01":"2025-10"]
seoul.columns = ["house_completion"]

print(f"\n서울 민간 준공: {len(seoul)}개월")
print(f"기간: {seoul.index.min().date()} ~ {seoul.index.max().date()}")
print(f"평균: {seoul['house_completion'].mean():.0f}호/월")
print(f"최대: {seoul['house_completion'].max()}호  최소: {seoul['house_completion'].min()}호")
print(seoul.head(6))

out_path = os.path.join(DATA_DIR, "house_completion.csv")
seoul.to_csv(out_path)
print(f"\n저장 완료: {out_path}")
