# Points


**Note**: The points have been moved to [`points`](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb/points) folder to avoid merge conflicts. To add points you will now have to add a jsonl file to the folder. An example could looks like so:

```
{"GitHub": "GitHubUser1", "New dataset": 6}
{"GitHub": "GitHubUser2",  "Review PR": 2}
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

**Note**: The points have been moved to the points folder to avoid merge conflicts.

Note that coordination and ideation are not included in the points yet, but are used to determine first and last authors.

# Contributor Informations

Please also add your first name and last name are as you want them to appear in a publication. If you do not wish to have your name in a publication, please add a note to that effect.

| GitHub            | First name | Last name  | Email                        | User on openreview   | Affiliations                                          |
| ----------------- | ---------- | ---------- | ---------------------------- | -------------------- | ----------------------------------------------------- |
| KennethEnevoldsen | Kenneth    | Enevoldsen | kennethcenevoldsen@gmail.com | ~Kenneth_Enevoldsen1 | Aarhus University, Denmark                            |
| x-tabdeveloping   | Márton     | Kardos     | martonkardos@cas.au.dk       | ~Márton_Kardos1      | Aarhus University, Denmark                            |
| imenelydiaker     | Imene      | Kerboua    |                              |                      | Esker, Lyon, France && INSA Lyon, LIRIS, Lyon, France |
| wissam-sib        | Wissam     | Siblini    |  wissam.siblini92@gmail.com    |   ~Wissam_Siblini1                   | N/A                                                   |
| GabrielSequeira   | Gabriel    | Sequeira   |                              |                      | N/A                                                   |
| schmarion         | Marion     | Schaeffer  |                              |  ~Marion_Schaeffer1  |  Wikit, Lyon, France                                  |
| MathieuCiancone   | Mathieu    | Ciancone   |                              |                      |  Wikit, Lyon, France                                  |
| MartinBernstorff  | Martin     | Bernstorff | martinbernstorff@gmail.com   | ~Martin_Bernstorff1  |  Aarhus University, Denmark                           |
| staoxiao          | Shitao     | Xiao       | 2906698981@qq.com            | ~Shitao_Xiao1        |  Beijing Academy of Artificial Intelligence           |
| ZhengLiu101       | Zheng     | Liu         | zhengliu1026@gmail.com       | ~Zheng_Liu4          |  Beijing Academy of Artificial Intelligence           |
| achibb            | Aaron      | Chibb      |                              |                      | N/A                                                   |
| cassanof          | Federico   | Cassano    | federico.cassanno@federico.codes | ~Federico_Cassano1 | Northeastern University, Boston, USA                |
| taidnguyen        | Nguyen     | Tai        | taing@seas.upenn.edu         | ~Nguyen_Tai1         |  University of Pennsylvania                           |
| xu3kev            | Wen-Ding   | Li         | wl678@cornell.edu            | ~Wen-Ding_Li1        |  Cornell University                                   |
| Rysias            | Jonathan   | Rystrøm    | jonathan.rystroem@gmail.com  |                      | University of Oxford, UK                              |
| taeminlee         | Taemin     | Lee        | taeminlee@korea.ac.kr        | ~Taemin_Lee1         | Korea University Human-Inspired AI Research           |
| izhx              | Xin        | Zhang      | zhangxin2023@stu.hit.edu.cn  |                      |  Harbin Institute of Technology, Shenzhen             |
| orionw            | Orion      | Weller     | oweller@cs.jhu.edu           | ~Orion_Weller1       |  Johns Hopkins University                             |
| slvnwhrl          | Silvan     | Wehrli     | wehrlis@rki.de               | ~Silvan_Wehrli1      | Robert Koch Institute, Berlin, Germany                |
| manandey          | Manan      | Dey        | manandey1@gmail.com          | ~Manan_Dey2          | Salesforce, India                                     |
| isaac-chung       | Isaac      | Chung      | chungisaac1217@gmail.com     | ~Isaac_Kwan_Yin_Chung1 | N/A                                                 |
| asparius          | Ömer       | Çağatan    | ocagatan19@ku.edu.tr         | ~Ömer_Veysel_Çağatan1 | Koç University,Turkey                                |
| rafalposwiata     | Rafał      | Poświata   | rposwiata@opi.org.pl         | ~Rafał_Poświata1     | National Information Processing Institute, Warsaw, Poland |
| rbroc             | Roberta    | Rocca      | roberta.rocca@cas.au.dk      | ~Roberta_Rocca1      | Aarhus University, Denmark                            |
| awinml            | Ashwin     | Mathur     | ashwinxmathur@gmail.com      |                      | N/A                                                   |
| guangyusong       | Guangyu    | Song       | guangysong@gmail.com         | ~Guangyu_Song1       | Tano Labs                            |
| davidstap        | David      | Stap       | dd.stap@gmail.com            | ~David_Stap          | University of Amsterdam.                         |
| HLasse            | Lasse      | Hansen     | lasseh0310@gmail.com         | ~Lasse_Hansen2       | Aarhus University, Denmark                            |
| jaygala24         | Jay        | Gala       | jaygala24@gmail.com          | ~Jay_Gala1           | MBZUAI                          |
| digantamisra98      | Diganta    | Misra      | diganta.misra@mila.quebec    | ~Diganta_Misra1       | Mila - Quebec AI Institute                           |
| PranjalChitale    | Pranjal    | Chitale    | cs21s022@smail.iitm.ac.in    | ~Pranjal_A_Chitale1       | Indian Institute of Technology Madras            |
| Akash190104       | Akash      | Kundu      | akashkundu2xx4@gmail.com      |~Akash_Kundu2             | Heritage Institute of Technology, Kolkata && Apart Research |
| dwzhu-pku         | Dawei      | Zhu        | dwzhu@pku.edu.cn             | ~Dawei_Zhu2       | Peking University            |
| ljvmiranda921     | Lester James | Miranda  | ljm@allenai.org              | ~Lester_James_Validad_Miranda1 | Allen Institute for AI |
| Sakshamrzt        | Saksham    | Thakur     | sthakur5@alumni.ncsu.edu     | ~Saksham_Thakur1     | N/A                                                   |
| Andrian0s     | Andrianos | Michail  | andrianos.michail@cl.uzh.ch         | ~Andrianos_Michail1 | University of Zurich|
| simon-clematide     | Simon | Clematide  | simon.clematide@cl.uzh.ch         | ~Simon_Clematide1 | University of Zurich|
| SaitejaUtpala     | Saiteja | Utpala  | saitejautpala@gmail.com         | ~Saiteja_Utpala1 | Microsoft Research|
| mmhamdy     | Mohammed | Hamdy  | mhamdy.res@gmail.com         | ~Mohammed_Hamdy1 | Cohere For AI Community|
| jupyterjazz       | Saba         | Sturua     | saba.sturua@jina.ai              |     ~Saba_Sturua1      | Jina AI                                                     |
| Ruqyai       | Ruqiya         | Bin Safi     | ruqiya.binsafi@libfstudy.ac.uk           |    ~Ruqiya_Bin_Safi1       | LIBF : The London Institute of Banking and Finance                                     
| kranthigv     | Kranthi Kiran | GV  | kranthi.gv@nyu.edu         | ~Kranthi_Kiran_GV1 | New York University|
| shreeya-dhakal            | Shreeya     | Dhakal     | ssdhakal57@gmail.com      |                      | Individual Contributor                                                   |
| dipam7 | Dipam | Vasani | dipam44@gmail.com | ~Dipam_Vasani1 | Individual Contributor                                                  |
| Art3mis07 | Gayatri | K | gayatrikrishnakumar0707@gmail.com | ~Gayatri_K1 | R. V. College of Engineering, Bengaluru | 
| jankounchained    | Jan        | Kostkan    | jan.kostkan@cas.au.dk | ~Jan_Kostkan1        | Aarhus University, Denmark                            |
| bp-high           | Bhavish       | Pahwa      | t-bpahwa@microsoft.com           | ~Bhavish_Pahwa1 | Microsoft Research                              |
| rasdani           | Daniel     | Auras      | daniel@ellamind.com          |   ~Daniel_Auras1     | ellamind, Germany                                   |
| ShawonAshraf      | Shawon     | Ashraf     | shawon@ellamind.com          |   ~Shawon_Ashraf1    | ellamind, Germany                                   |
| bjoernpl          | Björn      | Plüster    | bjoern@ellamind.com          |  ~Björn_Plüster1     | ellamind, Germany                                   |
| jphme             | Jan Philipp| Harries    | jan@ellamind.com             |~Jan_Philipp_Harries1 | ellamind, Germany                                   |
| malteos           | Malte       | Ostendorff      | malte@occiglot.eu           | ~Malte_Ostendorff1| Occiglot                             |
| ManuelFay         | Manuel        | Faysse     | manuel.faysse@centralesupelec.fr |              ~Manuel_Faysse1        | CentraleSupélec && Illuin Technology                  |
| hgissbkh          | Hippolyte     | Gisserot-Boukhlef    | hippolyte.gisserot-boukhlef@centralesupelec.fr        |   ~Hippolyte_Gisserot-Boukhlef1    | CentraleSupélec && Artefact Research Center   |
| sted97          | Simone     | Tedeschi    | tedeschi@diag.uniroma1.it        |   ~Simone_Tedeschi1                   | Sapienza University of Rome   |
| gentaiscool          | Genta Indra     | Winata    | gentaindrawinata@gmail.com        |   ~Genta_Indra_Winata1                   | N/A   | 
| henilp105 | Henil | Panchal | henilp105@gmail.com | ~Henil_Shalin_Panchal1 | Nirma University |
| ABorghini          | Alessia     | Borghini    | borghini.alessia99@gmail.com        |   ~Alessia_Borghini1            | Sapienza University of Rome   |
| jordiclive          | Jordan     | Clive    | jordan.clive19@imperial.ac.uk        |   ~Jordan_Clive1            | Imperial College London   |
| gowitheflow-1998          | Chenghao     | Xiao    | chenghao.xiao@durham.ac.uk        |   ~Chenghao_Xiao1            | Durham University   |
| mariyahendriksen          | Mariya     | Hendriksen    | mariya.hendriksen@gmail.com   |   ~Mariya_Hendriksen1            | University of Amsterdam   |
| dokato          | Dominik     | Krzemiński    | dkk33@cantab.ac.uk   |   ~Dominik_Krzemiński1            | Cohere For AI Community   |
| Samoed            | Roman      | Solomatin  | risolomatin@gmail.com        | ~Roman_Solomatin1    | ITMO                                                  |
| Alenush           | Alena      | Fenogenova | alenush93@gmail.com          | ~Alena_Fenogenova1   | SaluteDevices, Russia                                 |
| ab1992ao          | Aleksandr  | Abramov    | andril772@gmail.com          | ~Aleksandr_Abramov1  | SaluteDevices, Russia                                 |
| artemsnegirev     | Artem      | Snegirev   | artem.s.snegirev@gmail.com   | ~Artem_Snegirev1     | SaluteDevices, Russia                                 |
| anpalmak2003      | Anna       | Maksimova  | anpalmak@gmail.com           | ~Anna_Maksimova1     | SaluteDevices, Russia                                 |
| MariyaTikhonova   | Maria      | Tikhonova  | m_tikhonova94@mail.ru        | ~Maria_Tikhonova1    | SaluteDevices, HSE University, Russia                 |
| vaibhavad | Vaibhav    | Adlakha | vaibhav.adlakha@mila.quebec | ~Vaibhav_Adlakha1 | McGill University && Mila - Quebec AI Institute && ServiceNow Research                            |
| sivareddyg | Siva    | Reddy | siva.reddy@mila.quebec | ~Siva_Reddy1 | McGill University && Mila - Quebec AI Institute && ServiceNow Research && Facebook CIFAR AI Chair                            |
| guenthermi        | Michael    | Günther    | michael.guenther@jina.ai     | ~Michael_Günther1    | Jina AI                                               |
| violenil          | Isabelle   | Mohr       | isabelle.mohr@jina.ai        | ~Isabelle_Mohr1      | Jina AI                                               |
| akshita-sukhlecha | Akshita    | Sukhlecha  | sukhlecha.akshita@gmail.com  |                      | N/A                                                   |
| Muennighoff          | Niklas   | Muennighoff       | n.muennighoff@gmail.com        |       | Contextual AI                                               |
| AlexeyVatolin          | Aleksei   | Vatolin       | vatolinalex@gmail.com        | ~Aleksei_Vatolin1      | FRC CSC RAS                                               |
| xhluca | Xing Han | Lù | xing.han.lu@mail.mcgill.ca | ~Xing_Han_Lù1 | McGill University && Mila - Quebec AI Institute |
| crystina-z | Xinyu | Zhang | xinyucrystina.zhang@uwaterloo.ca | ~Crystina_Zhang1 | University of Waterloo |
| tomaarsen          | Tom   | Aarsen       |                           | ~Tom_Aarsen1      | Hugging Face |
| crystina-z | Xinyu | Zhang | xinyucrystina.zhang@uwaterloo.ca | ~Crystina_Zhang1 | University of Waterloo |
| mrshu | Marek | Suppa | marek.suppa@fmph.uniba.sk | ~Marek_Suppa1 | Comenius University in Bratislava && Cisco Systems |
