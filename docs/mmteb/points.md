# Points

**Note**: The points have been moved to [`points`](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb/points) folder to avoid merge conflicts. To add points you will now have to add a jsonl file to the folder. An example could looks like so:

```
{"GitHub": "GitHubUser1", "New dataset": 6}
{"GitHub": "GitHubUser2", "Review PR": 2}
```

The file should be named after the PR number. E.g. `438.jsonl`, where 438 is the PR number.

The possible keys to include is: 

```{python}
{
    "GitHub": "GitHubUser1",
    "New dataset": 2-6,  # 2 points for the dataset and 4 points for the task
    "New task": 2, # e.g. a new style of task (e.g. classification, or retrieval)
    "Dataset annotations": 1, # 1 point for each full dataset annotation
    "Bug fixes": 2-10, # depends on the complexity of the fix
    "Running Models": 1, # pr model run
    "Review PR": 2, # two points pr. reviewer, can be given to multiple reviewers
    "Paper Writing": NA, 
    "Ideation": NA,
    "Coordination": NA
}
```

**Note**: The points have been moved to points folder to avoid merge conflicts.

Note that coordination and ideation is not included in the points yet, but is used to determine first and last authors.

# Contributor Informations

Please also add your first name and last name are as you want them to appear in a publication. If you do not with to have your name in a publication, please add a note to that effect.

| GitHub            | First name | Last name  | Email                        | User on openreview   | Affiliations                                          |
| ----------------- | ---------- | ---------- | ---------------------------- | -------------------- | ----------------------------------------------------- |
| KennethEnevoldsen | Kenneth    | Enevoldsen | kennethcenevoldsen@gmail.com | ~Kenneth_Enevoldsen1 | Aarhus University, Denmark                            |
| x-tabdeveloping   | Márton     | Kardos     | martonkardos@cas.au.dk       | ~Márton_Kardos1      | Aarhus University, Denmark                            |
| imenelydiaker     | Imene      | Kerboua    |                              |                      | Esker, Lyon, France && INSA Lyon, LIRIS, Lyon, France |
| wissam-sib        | Wissam     | Siblini    | wissamsiblini92@gmail.com    |                      | N/A                                                   |
| GabrielSequeira   | Gabriel    | Sequeira   |                              |                      | N/A                                                   |
| schmarion         | Marion     | Schaeffer  |                              |  ~Marion_Schaeffer1  |  Wikit, Lyon, France                                  |
| MathieuCiancone   | Mathieu    | Ciancone   |                              |                      |  Wikit, Lyon, France                                  |
| MartinBernstorff  | Martin     | Bernstorff | martinbernstorff@gmail.com   | ~Martin_Bernstorff1  |  Aarhus University, Denmark                           |
| staoxiao          | Shitao     | Xiao       | 2906698981@qq.com            | ~Shitao_Xiao1        |  Beijing Academy of Artificial Intelligence           |
| achibb            | Aaron      | Chibb      |                              |                      | N/A                                                   |
| cassanof          | Federico   | Cassano    | federico.cassanno@federico.codes | ~Federico_Cassano1 | Northeastern University, Boston, USA                |
| taidnguyen        | Nguyen     | Tai        | taing@seas.upenn.edu         | ~Nguyen_Tai1         |  University of Pennsylvania                           |
| xu3kev            | Wen-Ding   | Li         | wl678@cornell.edu            | ~Wen-Ding_Li1        |  Cornell University                                   |
| taeminlee         | Taemin     | Lee        | taeminlee@korea.ac.kr        | ~Taemin_Lee1         | Korea University Human-Inspired AI Research           |
| izhx              | Xin        | Zhang      | zhangxin2023@stu.hit.edu.cn  |                      |  Harbin Institute of Technology, Shenzhen             |
| orionw            | Orion      | Weller     | oweller@cs.jhu.edu           | ~Orion_Weller1       |  Johns Hopkins University                             |
| slvnwhrl          | Silvan     | Wehrli     | wehrlis@rki.de               | ~Silvan_Wehrli1      | Robert Koch Institute, Berlin, Germany                |
| manandey          | Manan      | Dey        | manandey1@gmail.com          | ~Manan_Dey2          | Salesforce, India                                     |
| isaac-chung       | Isaac      | Chung      | chungisaac1217@gmail.com     | ~Isaac_Kwan_Yin_Chung1 | N/A                                                 |
| asparius          | Ömer       | Çağatan    | ocagatan19@ku.edu.tr         | ~Ömer_Veysel_Çağatan1 | Koç University,Turkey                                |
| rafalposwiata     | Rafał      | Poświata   | rposwiata@opi.org.pl         | ~Rafał_Poświata1     | National Information Processing Institute, Warsaw, Poland |
| rbroc             | Roberta    | Rocca      | roberta.rocca@cas.au.dk      | ~Roberta_Rocca1      | Aarhus University, Denmark                            |

