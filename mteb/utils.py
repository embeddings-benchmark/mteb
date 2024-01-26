import inspect


def get_embed_with_lang_func(model, query_specific=False, corpus_specific=False):
    """
    If model.encode supports the `language` argument, return this function.
    Otherwise, return a wrapper of model.encode with this extra argument.
    This is needed, because some of the models migh embed in a language-specific way.
    """
    func = model.encode

    # for reranking tasks, a model (like E5) may have special methods for encoding queries and passages.
    if query_specific and hasattr(model, 'encode_queries'):
        func = model.encode_queries
    if corpus_specific and hasattr(model, 'encode_corpus'):
        func = model.encode_corpus

    signature = inspect.signature(func)
    if "language" in signature.parameters:
        return func

    # TODO: describe the signature of this function with more details
    # text, batch_size, maybe progress_bar
    def func_without_language(sentences, *args, language=None, **kwargs):
        return func(sentences, *args, **kwargs)

    return func_without_language


# language codes that contain a dash and should not be split 
LANGS_WITH_DASH = {
    "zh-CN", 
    "zh-TW",
    "da-bornholm",
    "en-ext",  # ???
}


def maybe_split_language_pair(lang: str):
    """
    If the language code is actually a pair (like "en-fr"), return the two individual language codes.
    Otherwise, return the input language code twice.
    """
    if isinstance(lang, str) and lang.count('-') == 1 and lang not in LANGS_WITH_DASH:
        lang1, lang2 = lang.split('-')
        return lang1, lang2
    return lang, lang
