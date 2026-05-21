Source pool - MVEB(extended) tasks: 184

Loading model results for correlation analysis...
Found 591 models
Models with metadata: 584
Models after NaN filter (<=10 NaN): 13
Tasks with results: 183
Tasks with results: 183/184
Computed 183x183 correlation matrix
Note: 1 task(s) dropped (no model results): ['VideoConPairClassification']

Analysis written to: scripts/mveb_paper/mveb_task_selection_analysis.md

================================================================================
SUMMARY
================================================================================

Source: MVEB(extended) with 184 tasks
Working pool: 183 tasks

Eval times for working pool:
  ebind-av: 122h 40m (183/183 tasks)
  pe-av-small: 159h 40m (183/183 tasks)
  LCO-Embedding-Omni-7B: 302h 39m (183/183 tasks)
  Qwen2.5-Omni-7B: 267h 15m (183/183 tasks)

Protected tasks: 1

Results by threshold:
  Thresh  Tasks  Retr  Cls  Clu  MLC  Pair  ZS  QA  Spearman  Pearson
  ------- ------ ----- ---- ---- ---- ----- --- --- --------- --------
  0.95    55     34    8    3    0    3     3   4   0.9780    0.9978
  0.93    47     27    7    3    0    2     4   4   0.9615    0.9974
  0.9     38     19    9    1    0    3     3   3   0.9821    0.9974
  0.88    35     18    7    1    0    3     4   2   0.9546    0.9958
  0.87    32     16    7    1    0    2     4   2   0.9656    0.9973
  0.85    27     13    7    1    0    1     3   2   0.9601    0.9963
  0.84    23     10    6    1    0    2     2   2   0.9821    0.9965
  0.83    22     10    6    1    0    1     2   2   0.9821    0.9965
  0.82    21     9     6    1    0    1     2   2   0.9931    0.9970
  0.81    21     9     7    1    0    0     2   2   0.9931    0.9973
  0.8     21     9     7    1    0    0     2   2   0.9931    0.9965
  0.7     14     7     3    0    0    1     1   2   0.9381    0.9924
  0.6     13     7     2    0    0    1     1   2   0.9381    0.9908
  0.5     13     7     2    0    0    1     1   2   0.9381    0.9908

Recommended (threshold=0.85): 27 tasks
  Correlation with MVEB(extended): Spearman=0.9601, Pearson=0.9963

Full analysis (with per-model eval times) saved to: scripts/mveb_paper/mveb_task_selection_analysis.md
