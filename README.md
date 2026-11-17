# Group 2 – Business Analyst Advertising Analysis (Assignment 2)

Python code for the Group 2 analytic report on US LinkedIn Business Analyst (BA)
advertisements, January 2024 (one CRISP-DM iteration: data preparation, modelling,
evaluation). This repository holds the data-preparation code shared by all research
questions and the RQ1 and RQ2 analysis. RQ3, RQ4 and the segmentation model are added
to this repository by the members responsible for them.

| RQ | Question | Owner | Script |
|----|----------|-------|--------|
| – | Shared populations (title rule, US filter, location validation, skills join) | Mandar Abhay Patne (a3221269) | `prep.py` |
| RQ1 | How are BA ads distributed across US states, raw and relative to labour force? | Mandar Abhay Patne (a3221269) | `analysis_rq1_rq2.py` (RQ1 block) |
| RQ2 | Which organisations advertise BA roles, how concentrated is it, what type of advertiser? | Harshit Jain (a3191318) | `analysis_rq1_rq2.py` (RQ2 block), `data/advertiser_types.csv` |
| RQ3 | Which capabilities are most prevalent and how do they combine? | Bhuvan, Yifu, Aditya | added by owners |
| RQ4 | Which capabilities are distinctive for BA vs non-BA ads? | Nishanth, Sarvesh | added by owners |

## Repository layout

```
extract_ba_subset.py         optional: cut the Kaggle zip to BA ads (low memory)
prep.py                      populations used by every RQ  -> output/ba_validated.csv, ba_skills_long.csv
analysis_rq1_rq2.py          RQ1 + RQ2 tables, tests, Figures 2-4, check against report values
data/
  bls_labour_force_jan2024.csv   BLS civilian labour force by state incl. DC, January 2024 (USDL-24-0482)
  advertiser_types.csv           manual advertiser-type coding of the top-50 advertisers (RQ2)
  linkedin_job_postings.csv      (download, git-ignored)
  job_skills.csv                 (download, git-ignored)
output/                      generated, git-ignored
```

## Data

Download *1.3M LinkedIn Jobs & Skills (2024)* (asaniczka 2024) from Kaggle:
<https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024>
and put `linkedin_job_postings.csv` and `job_skills.csv` in `data/`. They are too large
for GitHub and are listed in `.gitignore`.

## Low-memory option

The full files are large. To skip unzipping and loading 1.3 million rows, cut the Kaggle
zip down to the BA ads first (reads in chunks, writes small files into `data/`):

```bash
python extract_ba_subset.py path/to/archive.zip
```

RQ1/RQ2 results are the same, because `prep.py` only uses BA-titled ads.

## How to run

```bash
pip install -r requirements.txt
python prep.py               # populations: 4,546 -> 3,732 -> 3,723 validated -> 3,707 by state
python analysis_rq1_rq2.py   # RQ1 + RQ2 tables, tests and Figures 2-4 -> output/
```

`prep.py` writes `output/unmapped_locations.csv` if any `job_location` label is not
recognised. Add those labels to `METRO_MAP` (single-state metros) or `MULTI_STATE`, then re-run.

## Outputs

| File | Content | Report |
|------|---------|--------|
| `rq1_states.csv` | ads, share, ads per 100k labour force, LQ, LQ excl. LATICRETE, exact Poisson p, Holm p, raw/rate ranks | Section 4.1, Appendix C.1 |
| `fig2_rq1.png` | top-12 states by count; significant location quotients | Figure 2 |
| `rq2_all_advertisers.csv` | ads, titles, locations, states, share and cumulative share per advertiser | Section 4.2 |
| `rq2_top50_advertisers.csv` | top-50 advertisers with advertiser type | Appendix C.2 |
| `rq2_type_skill_compare.csv` | skill prevalence, intermediaries vs employers/consultancies, chi-square, Holm p | Figure 4, Appendix C.2 |
| `fig3_rq2.png`, `fig4_rq2.png` | concentration curve and advertiser types; skill dumbbell | Figures 3 and 4 |
| `results_rq1_rq2.json` | every headline statistic quoted in the report | Sections 4.1-4.2 |

## Methods (summary)

- **Population:** `job_title` matches `\bbusiness analyst\b` (case-insensitive), US search
  context, advertised location validated (9 non-US ads removed; "Canada, KY" kept).
- **RQ1:** ads per state, ads per 100,000 labour force, location quotient (LQ); Spearman
  correlation; chi-square goodness of fit; exact Poisson test per state with Holm correction;
  sensitivity without LATICRETE's multi-location ads; commuting check pooling DC, Maryland
  and Virginia.
- **RQ2:** ads per advertiser, top-N shares, Herfindahl–Hirschman index (HHI), Gini; HHI on
  unique company–title pairs; top-50 advertiser typing; chi-square tests (Holm) of skill mix
  for intermediaries vs employers/consultancies (ads with usable skills, exact skill labels).

## Checking the numbers

The last block of `analysis_rq1_rq2.py` compares the script's results with the values
printed in the report and prints `OK` or `DIFF` for each. The advertiser-type file must
cover all 50 top advertisers for the RQ2 type figures to match (it does); the script lists any
top-50 advertiser that has no type. Coded totals in the report:

| Type | Top-50 advertisers | Top-50 ads |
|------|-------------------|-----------|
| Staffing / recruitment | 32 | 672 |
| Direct employer | 7 | 404 |
| Job board / aggregator | 4 | 250 |
| IT services & consulting | 7 | 165 |
| **Total** | **50** | **1,491** |

Verified run (17 Nov 2026, full Kaggle data via `extract_ba_subset.py`): all 15 RQ1/RQ2
checks print `OK`.

Figure 1 (pipeline) is a diagram and has no script.

Environment: Python 3.11 or later; pandas, NumPy, SciPy, statsmodels, matplotlib and
scikit-learn (see `requirements.txt`). No random step is used in RQ1/RQ2.

## Data and ethics

The dataset is public and holds no candidate information; results are reported at state
and organisation level only. Advertiser types were coded by hand from company names and
public company profiles and describe how an organisation advertises, not its conduct.

## Reference

asaniczka 2024, *1.3M LinkedIn Jobs & Skills (2024)*, Kaggle.
US Bureau of Labor Statistics 2024, *State Employment and Unemployment – January 2024*, USDL-24-0482.
