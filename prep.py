"""prep.py - build the analytical populations used in the Group 2 report.

Input (Kaggle: asaniczka, '1.3M LinkedIn Jobs & Skills (2024)'), placed in data/:
    linkedin_job_postings.csv   (job_link, job_title, company, job_location, first_seen,
                                 search_country, job_level, job_type, ...)
    job_skills.csv              (job_link, job_skills)

Output (written to output/):
    ba_validated.csv   one row per validated US explicit-BA ad (3,723 in the report)
    ba_skills_long.csv one row per ad-skill pair (skill_clean)

Run:  python prep.py
"""
import re
from pathlib import Path

import pandas as pd

DATA = Path('data')
OUT = Path('output')
OUT.mkdir(exist_ok=True)

US_STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
    'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
    'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
    'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
    'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
}
STATE_NAMES = {v.lower(): v for v in US_STATES.values()}

# Single-state metro / area labels -> state. Extend this list after checking
# the 'unmapped_locations.csv' file that this script writes.
METRO_MAP = {
    'greater boston': 'Massachusetts',
    'new york city metropolitan area': 'New York',
    'san francisco bay area': 'California',
    'los angeles metropolitan area': 'California',
    'greater chicago area': 'Illinois',
    'dallas-fort worth metroplex': 'Texas',
    'greater houston': 'Texas',
    'atlanta metropolitan area': 'Georgia',
    'miami-fort lauderdale area': 'Florida',
    'greater seattle area': 'Washington',
    'denver metropolitan area': 'Colorado',
    'greater philadelphia': 'Pennsylvania',
    'detroit metropolitan area': 'Michigan',
    'greater minneapolis-st. paul area': 'Minnesota',
    'greater phoenix area': 'Arizona',
    'charlotte metro': 'North Carolina',
    'raleigh-durham-chapel hill area': 'North Carolina',
    'greater tampa bay area': 'Florida',
    'greater orlando': 'Florida',
    'austin, texas metropolitan area': 'Texas',
}
# Metro areas that span several states: kept nationally, excluded from state counts.
MULTI_STATE = {'washington dc-baltimore area', 'kansas city metropolitan area'}
NON_US_COUNTRIES = {'canada', 'mexico', 'mexico city metropolitan area'}

BA_RULE = re.compile(r'\bbusiness analyst\b', re.IGNORECASE)


def parse_location(loc):
    """Return (loc_status, state_clean) for one job_location string."""
    if not isinstance(loc, str) or not loc.strip():
        return 'missing', None
    s = loc.strip()
    low = s.lower()
    parts = [p.strip() for p in s.split(',')]
    last = parts[-1]
    # 1. "City, TX"  (checked first so that "Canada, KY" stays in Kentucky)
    if len(parts) >= 2 and last.upper() in US_STATES and len(last) == 2:
        return 'state', US_STATES[last.upper()]
    # 2. "Texas, United States"
    if len(parts) >= 2 and last.lower() == 'united states':
        name = parts[-2].lower()
        if name in STATE_NAMES:
            return 'state', STATE_NAMES[name]
        if name in METRO_MAP:
            return 'state', METRO_MAP[name]
        if name in MULTI_STATE:
            return 'multi_state', None
        return 'national', None
    # 3. "United States" only
    if low == 'united states':
        return 'national', None
    # 4. multi-state metros
    if low in MULTI_STATE:
        return 'multi_state', None
    # 5. single-state metro labels
    if low in METRO_MAP:
        return 'state', METRO_MAP[low]
    # 6. advertised location outside the US
    if last.lower() in NON_US_COUNTRIES or low in NON_US_COUNTRIES:
        return 'non_us', None
    if low in STATE_NAMES:
        return 'state', STATE_NAMES[low]
    return 'unmapped', None


def main():
    jobs = pd.read_csv(DATA / 'linkedin_job_postings.csv')
    jobs['explicit_ba_title'] = jobs.job_title.fillna('').str.contains(BA_RULE)
    ba = jobs[jobs.explicit_ba_title].copy()
    print('explicit BA title:', len(ba))                         # report: 4,546

    ba = ba[ba.search_country == 'United States'].copy()
    print('US search context:', len(ba))                         # report: 3,732

    ba[['loc_status', 'state_clean']] = ba.job_location.apply(
        lambda x: pd.Series(parse_location(x)))
    ba['location_country_issue'] = ba.loc_status.eq('non_us')
    print(ba.loc_status.value_counts().to_string())

    unmapped = ba[ba.loc_status == 'unmapped'].job_location.value_counts()
    if len(unmapped):
        unmapped.to_csv(OUT / 'unmapped_locations.csv')
        print(f'WARNING: {len(unmapped)} unmapped location labels -> extend METRO_MAP '
              '(see output/unmapped_locations.csv)')

    val = ba[~ba.location_country_issue].copy()
    print('validated US:', len(val))                             # report: 3,723
    print('state-identifiable:', val.state_clean.notna().sum())  # report: 3,707

    skills = pd.read_csv(DATA / 'job_skills.csv')
    sk = skills[skills.job_link.isin(val.job_link) & skills.job_skills.notna()].copy()
    sk['skill'] = sk.job_skills.str.split(',')
    sk = sk.explode('skill')
    sk['skill_clean'] = (sk.skill.str.strip().str.lower()
                         .str.replace(r'\s+', ' ', regex=True))
    sk = sk[sk.skill_clean != '']
    val['has_skills'] = val.job_link.isin(sk.job_link)
    print('with usable skills:', val.has_skills.sum())            # report: 3,607

    val.to_csv(OUT / 'ba_validated.csv', index=False)
    sk[['job_link', 'skill_clean']].to_csv(OUT / 'ba_skills_long.csv', index=False)


if __name__ == '__main__':
    main()
