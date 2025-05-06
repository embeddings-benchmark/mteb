from __future__ import annotations

from mteb.tasks.Retrieval import AILACasedocs, AILAStatutes, LegalQuAD, ChatDoctor_HealthCareMagic
from mteb.abstasks import AbsTask
from mteb.abstasks.aggregated_task import AbsTaskAggregate, AggregateTaskMetadata

task_list_rteb: list[AbsTask] = [
    AILACasedocs(),
    AILAStatutes(),
    LegalQuAD(),
    ChatDoctor_HealthCareMagic(),
    # APPS(),  # new
    # ConvFinQA(),  # new
    # COVID_QA(),  # new
    # DialogsumGerman(),  # new
    # DS1000(),  # new
    # FinanceBench(),  # new
    # FinQA(),  # new
    # FiQAPersonalFinance(),
    # FrenchBoolQ(),
    # FrenchOpenFiscalTexts(),
    # FrenchTriviaQAWikicontext(),
    # GermanLegalSentences(),
    # Github(),
    # HC3Finance(),
    # HealthCareGerman(),
    # HumanEval(),
    # JapaneseCoNaLa(),
    # JapanLaw(),
    # LegalSummarization(),
    # MBPP(),
    # TAT_QA(),
    # WikiSQL(),
]


class RTEBAggregatedTask(AbsTaskAggregate):
    metadata = AggregateTaskMetadata(
        name="RTEBAggregatedTask",
        description="Aggregated task for all RTEB tasks",
        reference=None,
        tasks=task_list_rteb,
        main_score="average_score",
        type="Retrieval",
        eval_splits=["test"],
        bibtex_citation=None,
    )
