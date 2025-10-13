# Slovak MTEB - Tasks to Add

## High Priority - Missing Task Types

### Pair Classification
- [x] SlovakRTE - Recognizing Textual Entailment ✅
      Source: slovak-nlp/sklep (rte subset)
      Size: 4.4k examples (2,490 train, 277 val, 1,660 test)
      Quality: Professional translation + human verification (ACL 2025)
      Status: Implemented in mteb/tasks/PairClassification/slk/SlovakRTE.py

### Bitext Mining
- [ ] OpusSlovakEnglishBitextMining - Slovak-English parallel sentences
      Source: Helsinki-NLP/opus-100 (en-sk config)
      Size: 1M training pairs, 2k validation, 2k test
      Quality: Standard OPUS benchmark, industry standard

### Classification
- [ ] MultiEURLEXSlovak - Multi-label legal document classification
      Source: nlpaueb/multi_eurlex (sk config)
      Size: 11k train, 1k dev, 5k test
      Quality: Official EU translations, expert EUROVOC annotations
      Notes: Requires trust_remote_code=True

- [ ] SentiSkClassification - Facebook comment sentiment analysis
      Source: TUKE-KEMT/senti-sk
      Size: 19k train, 4.6k test
      Quality: Native speaker annotations (single annotator)
      Domain: Social media

## Medium Priority

### Bitext Mining (Domain-Specific)
- [ ] MultiEURLEXSlovakEnglishBitext - Legal parallel documents
      Source: nlpaueb/multi_eurlex (Slovak-English pairs)
      Size: ~11k document pairs
      Quality: Professional legal translations
      Notes: Optional, lower priority than OPUS-100

## Already in Code TODOs (for reference)
- SlovakNLI (Pair Classification)
- slovak-financial-exam (Pair Classification)
- SKQuadReranking (Reranking)
- SlovakFactCheckReranking (Reranking)
- SlovakSumClustering (Clustering)
- SMESumClustering (Clustering)
- PravdaTagsClustering (Clustering)
- SMESum (Retrieval)

## Implementation Priority Order
1. SlovakRTE (fills Pair Classification gap)
2. OpusSlovakEnglishBitextMining (fills Bitext Mining gap)
3. MultiEURLEXSlovak (adds legal domain + multi-label)
4. SentiSkClassification (adds social media domain)
5. MultiEURLEXSlovakEnglishBitext (optional legal bitext)
