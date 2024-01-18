from datasets import load_dataset

from ...abstasks.AbsTaskRetrieval import AbsTaskRetrieval

_EVAL_SPLIT = 'test'


class MIRACLRetrieval(AbsTaskRetrieval):
    @property
    def description(self):
        return {
            'name': 'MIRACLRetrieval',
            'hf_hub_name': 'jinaai/miracl',
            'reference': 'https://project-miracl.github.io/',
            'description': (
                'MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual '
                'retrieval dataset that focuses on search across 18 different languages. This task focuses on '
                'the Spanish subset, using the test set containing 648 queries and 6443 passages.'
            ),
            'type': 'Retrieval',
            'category': 's2p',
            'eval_splits': ['test'],
            'eval_langs': ['es'],
            'main_score': 'ndcg_at_10',
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        data = load_dataset(
            self.description["hf_hub_name"],
            self.description['eval_langs'][0],
            trust_remote_code=True,
        )[_EVAL_SPLIT]

        queries = {}
        corpus = {}
        relevant_docs = {}

        # Generate unique IDs for queries and documents
        query_id_counter = 1
        document_id_counter = 1

        # Iterate through the dataset
        for row in data:
            query_text = row['query']
            positive_texts = row['positive']
            negative_texts = row['negative']

            # Assign unique ID to the query
            query_id = f"Q{query_id_counter}"
            queries[query_id] = query_text
            query_id_counter += 1

            # Add positive and negative texts to corpus with unique IDs
            for text in positive_texts + negative_texts:
                doc_id = f"D{document_id_counter}"
                corpus[doc_id] = text
                document_id_counter += 1

                # Add relevant document information to relevant_docs for positive texts only
                if text in positive_texts:
                    if query_id not in relevant_docs:
                        relevant_docs[query_id] = {}
                    relevant_docs[query_id][doc_id] = 1

        # relevant_docs will be positives for the query
        self.queries = {_EVAL_SPLIT: queries}
        self.corpus = {_EVAL_SPLIT: {k: {'text': v} for k, v in corpus.items()}}
        self.relevant_docs = {_EVAL_SPLIT: relevant_docs}

        self.data_loaded = True
