"""extract_ba_subset.py - cut the Kaggle download down to the BA ads, without unzipping.

Reads the Kaggle zip in chunks (low memory), keeps only postings whose job_title
contains "business analyst" plus their skills, and writes two small files:

    data/linkedin_job_postings.csv   (~4,500 rows instead of 1.35 million)
    data/job_skills.csv              (skills for those ads only)

prep.py and analysis_rq1_rq2.py then run on these small files and give the same
RQ1/RQ2 results, because they only use BA-titled ads.

Run:  python extract_ba_subset.py path/to/archive.zip
"""
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

BA_RULE = re.compile(r'\bbusiness analyst\b', re.IGNORECASE)
CHUNK = 100_000


def member(zf, name):
    """Find a file inside the zip, whatever folder it sits in."""
    for n in zf.namelist():
        if n.lower().endswith(name):
            return n
    sys.exit(f'{name} not found in the zip. Files in it: {zf.namelist()}')


def main():
    if len(sys.argv) != 2:
        sys.exit('usage: python extract_ba_subset.py path/to/archive.zip')
    out = Path('data')
    out.mkdir(exist_ok=True)
    zf = zipfile.ZipFile(sys.argv[1])

    # 1. postings: keep BA titles only
    parts, total = [], 0
    with zf.open(member(zf, 'linkedin_job_postings.csv')) as f:
        for i, chunk in enumerate(pd.read_csv(f, chunksize=CHUNK, dtype=str)):
            total += len(chunk)
            parts.append(chunk[chunk.job_title.fillna('').str.contains(BA_RULE)])
            print(f'  postings read: {total:,}', end='\r')
    ba = pd.concat(parts, ignore_index=True)
    ba.to_csv(out / 'linkedin_job_postings.csv', index=False)
    print(f'\npostings: {total:,} read, {len(ba):,} BA ads kept')   # report: 1,348,454 / 4,546

    # 2. skills: keep rows for those ads only
    links = set(ba.job_link)
    parts, total = [], 0
    with zf.open(member(zf, 'job_skills.csv')) as f:
        for chunk in pd.read_csv(f, chunksize=CHUNK, dtype=str):
            total += len(chunk)
            parts.append(chunk[chunk.job_link.isin(links)])
            print(f'  skill rows read: {total:,}', end='\r')
    sk = pd.concat(parts, ignore_index=True)
    sk.to_csv(out / 'job_skills.csv', index=False)
    print(f'\nskills: {total:,} read, {len(sk):,} kept')

    print('Done. Upload data/linkedin_job_postings.csv and data/job_skills.csv, '
          'or run prep.py and analysis_rq1_rq2.py next.')


if __name__ == '__main__':
    main()
