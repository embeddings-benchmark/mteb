from __future__ import annotations

from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata
from mteb.tasks.RTEB.RTEBAILACasedocsTask import RTEBAILACasedocs
from mteb.tasks.RTEB.RTEBAILAStatutesTask import RTEBAILAStatutes
from mteb.tasks.RTEB.RTEBAPPSTask import RTEBAPPS
from mteb.tasks.RTEB.RTEBChatDoctor_HealthCareMagicTask import (
    RTEBChatDoctor_HealthCareMagic,
)
from mteb.tasks.RTEB.RTEBConvFinQATask import RTEBConvFinQA
from mteb.tasks.RTEB.RTEBCOVID_QATask import RTEBCOVID_QA
from mteb.tasks.RTEB.RTEBDialogsumGermanTask import RTEBDialogsumGerman
from mteb.tasks.RTEB.RTEBDS1000Task import RTEBDS1000
from mteb.tasks.RTEB.RTEBFinanceBenchTask import RTEBFinanceBench
from mteb.tasks.RTEB.RTEBFinQATask import RTEBFinQA
from mteb.tasks.RTEB.RTEBFiQAPersonalFinanceTask import RTEBFiQAPersonalFinance
from mteb.tasks.RTEB.RTEBFrenchBoolQTask import RTEBFrenchBoolQ
from mteb.tasks.RTEB.RTEBFrenchOpenFiscalTextsTask import RTEBFrenchOpenFiscalTexts
from mteb.tasks.RTEB.RTEBFrenchTriviaQAWikicontextTask import (
    RTEBFrenchTriviaQAWikicontext,
)
from mteb.tasks.RTEB.RTEBGermanLegalSentencesTask import RTEBGermanLegalSentences
from mteb.tasks.RTEB.RTEBGithubTask import RTEBGithub
from mteb.tasks.RTEB.RTEBHC3FinanceTask import RTEBHC3Finance
from mteb.tasks.RTEB.RTEBHealthCareGermanTask import RTEBHealthCareGerman
from mteb.tasks.RTEB.RTEBHumanEvalTask import RTEBHumanEval
from mteb.tasks.RTEB.RTEBJapaneseCoNaLaTask import RTEBJapaneseCoNaLa
from mteb.tasks.RTEB.RTEBJapanLawTask import RTEBJapanLaw
from mteb.tasks.RTEB.RTEBLegalQuADTask import RTEBLegalQuAD
from mteb.tasks.RTEB.RTEBLegalSummarizationTask import RTEBLegalSummarization
from mteb.tasks.RTEB.RTEBMBPPTask import RTEBMBPP
from mteb.tasks.RTEB.RTEBTAT_QATask import RTEBTAT_QA
from mteb.tasks.RTEB.RTEBWikiSQLTask import RTEBWikiSQL

task_list_rteb: list[AbsTask] = [
    RTEBAILACasedocs(),
    RTEBAILAStatutes(),
    RTEBAPPS(),
    RTEBLegalQuAD(),
    RTEBChatDoctor_HealthCareMagic(),
    RTEBConvFinQA(),
    RTEBCOVID_QA(),
    RTEBDialogsumGerman(),
    RTEBDS1000(),
    RTEBFinanceBench(),
    RTEBFinQA(),
    RTEBFiQAPersonalFinance(),
    RTEBFrenchBoolQ(),
    RTEBFrenchOpenFiscalTexts(),
    RTEBFrenchTriviaQAWikicontext(),
    RTEBGermanLegalSentences(),
    RTEBGithub(),
    RTEBHC3Finance(),
    RTEBHealthCareGerman(),
    RTEBHumanEval(),
    RTEBJapaneseCoNaLa(),
    RTEBJapanLaw(),
    RTEBLegalSummarization(),
    RTEBMBPP(),
    RTEBTAT_QA(),
    RTEBWikiSQL(),
]


class RTEBAggregatedTask(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="RTEBAggregatedTask",
        description="Aggregated task for all RTEB tasks",
        reference=None,
        tasks=task_list_rteb,
        main_score="average_score",
        type="RTEB",
        eval_splits=["test"],
        bibtex_citation=None,
    )
