Source pool - MVEB(extended) tasks: 184

Loading model results for correlation analysis...
Found 591 models
Models with metadata: 584
Models after NaN filter (<=10 NaN): 13
Tasks with results: 183
Tasks with results: 183/184
Computed 183x183 correlation matrix

Analysis written to: scripts/mveb_paper/mveb_task_selection_analysis.md

================================================================================
SUMMARY
================================================================================

Source: MVEB(extended) with 184 tasks
Working pool: 184 tasks

Eval times for working pool:
  ebind-av: 122h 40m (183/184 tasks)
  pe-av-small: 159h 40m (183/184 tasks)
  LCO-Embedding-Omni-7B: 302h 39m (183/184 tasks)
  Qwen2.5-Omni-7B: 267h 15m (183/184 tasks)

Protected tasks: 1

Results by threshold:
  Thresh  Tasks  Retr  Cls  Clu  MLC  Pair  ZS  QA  Spearman  Pearson
  ------- ------ ----- ---- ---- ---- ----- --- --- --------- --------
  0.95    79     34    15   6    0    8     12  4   0.9945    0.9993
  0.93    66     28    14   6    0    7     8   3   0.9890    0.9979
  0.9     55     21    12   5    0    7     8   2   0.9890    0.9986
  0.85    41     14    9    5    0    5     6   2   0.9890    0.9980
  0.8     30     10    8    3    0    3     4   2   0.9615    0.9968
  0.7     21     9     4    2    0    3     1   2   0.9945    0.9938
  0.6     20     8     4    2    0    3     1   2   0.9890    0.9887
  0.5     18     8     3    1    0    3     1   2   0.9890    0.9869
  0.4     18     8     3    1    0    3     1   2   0.9890    0.9869

Recommended (threshold=0.85): 41 tasks
  Correlation with MVEB(extended): Spearman=0.9890, Pearson=0.9980

Full analysis (with per-model eval times) saved to: scripts/mveb_paper/mveb_task_selection_analysis.md
