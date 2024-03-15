from ....abstasks.AbsTaskSTS import AbsTaskSTS


class ATEC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "ATEC",
            "hf_hub_name": "C-MTEB/ATEC",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
            'revision': '0f319b1142f28d00e055a6770f3f726ae9b7d865',
        }



class BQ(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "BQ",
            "hf_hub_name": "C-MTEB/BQ",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
            'revision': 'e3dda5e115e487b39ec7e618c0c6a29137052a55',
        }


class LCQMC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "LCQMC",
            "hf_hub_name": "C-MTEB/LCQMC",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
            'revision': '17f9b096f80380fce5ed12a9be8be7784b337daf',
        }



class PAWSX(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "PAWSX",
            "hf_hub_name": "C-MTEB/PAWSX",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
            'revision': '9c6a90e430ac22b5779fb019a23e820b11a8b5e1',
        }


class STSB(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "STSB",
            "hf_hub_name": "C-MTEB/STSB",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 5,
            'revision': '0cde68302b3541bb8b3c340dc0644b0b745b3dc0',
        }


class AFQMC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "AFQMC",
            "hf_hub_name": "C-MTEB/AFQMC",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["validation"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
            'revision': 'b44c3b011063adb25877c13823db83bb193913c4',
        }



class QBQTC(AbsTaskSTS):
    @property
    def description(self):
        return {
            "name": "QBQTC",
            "hf_hub_name": "C-MTEB/QBQTC",
            "reference": "https://github.com/CLUEbenchmark/QBQTC/tree/main/dataset",
            "type": "STS",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["zh"],
            "main_score": "cosine_spearman",
            "min_score": 0,
            "max_score": 1,
            'revision': '790b0510dc52b1553e8c49f3d2afb48c0e5c48b7',
        }
