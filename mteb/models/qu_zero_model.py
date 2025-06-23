from mteb.models.wrapper import Wrapper
from mteb.encoder_interface import PromptType
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os


class CustomWrapper(Wrapper):
    def __init__(self, model_name, model_revision):
        # load the tokenizer and the model
        self.qu_tokenizer =  = AutoTokenizer.from_pretrained("Qwen3-4B")
        self.qu_model = AutoModelForCausalLM.from_pretrained(
            "Qwen3-4B",
            torch_dtype="auto"
        )
        self.embed_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

        # prepare the model input
        self.prompt_qu = """
        你是搜索query分析专家，根据用户输入，分析搜索意图，并给出最适合用于检索的query, 要求这些query可以高效检索出用户所需结果，适合用于检索系统和排序系统；
        【任务说明】
        1. 搜索意图分析
        用1-2句话简洁概括用户核心需求，避免主观推测
        2. 核心词提取
        尝试提取query中已有或能回答query的1至3个最能代表用户意图的关键词/短语，确保核心词能完全覆盖原始query需求，体现用户核心需求，能保证检索结果正确性；
        3. 查询优化
        生成有助于检索出用户所需的信息的3个查询，需满足：
        -要素保留：100%保留原始核心需求
        -精准补充：添加必要限定信息（如同义词、专业术语变体）
        -高效检索：能准确表达原始query需求，同时能准确且高效检索出用户所需信息

        【输出格式要求】
        必须严格使用以下JSON结构：
        {{"analysis": "对用户搜索需求的分析","keywords": ["核心词1","核心词2"],"queries": ["查询1","查询2","查询3"]}}
        【示例】
        输入：杨幂演哪部剧火的
        输出：{{"analysis": "用户希望了解杨幂因出演哪部电视剧而走红，杨幂出演的电视剧有宫锁心玉、神雕侠侣、古剑奇谭等，杨幂凭借穿越剧《宫锁心玉》赢得广泛关注","keywords": ["杨幂","走红","电视剧"],"queries": ["杨幂因哪部剧成名","宫锁心玉","杨幂爆红作品"]}}
        输入：{query}
        输出：
        """
        super().__init__(model_name, model_revision)
        # your custom implementation here

    def parse_result(self, content_str):
        # 尝试解析 content 中的 JSON 列表
        dumped = None       
        try:
            dumped = json.loads(content_str.strip())
        except:
            try:
                dumped = eval(content_str)
            except:
                try:
                    dumped = eval(content_str.replace('```json', '').replace('```', ''))
                except Exception as e:
                    pass

        return dumped if dumped is not None else content_str
        
    def get_qu(self, q, tokenizer, model):
        messages = [ 
            {"role": "user", "content": self.prompt_qu.format(query=q)}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # conduct text completion
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        return self.parse_result(content)

    def cached_query_understanding(self, text):
        return self.get_qu(text, self.qu_tokenizer, self.qu_model)

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs
    ) -> np.ndarray:
        # your custom implementation here
        if prompt_type == PromptType.query:
            new_sentences = []
            for s in sentences:
                new_sentences.append(s + self.cached_query_understanding(s))
            query_embeddings = self.embed_model.encode(new_sentences, prompt_name="query")
            return query_embeddings
        else:
            document_embeddings = self.embed_model.encode(sentences)
            return document_embeddings

training_data = {
    "T2Retrieval": ["train"],
    "DuRetrieval": ["train"],
    "MMarcoReranking": ["train"],
    "CMedQAv2-reranking": ["train"],
    "NQ": ["train"],
    "MSMARCO": ["train"],
    "HotpotQA": ["train"],
    "FEVER": ["train"],
    "MrTidyRetrieval": ["train"],
    "MIRACLRetrieval": ["train"],
    "CodeSearchNet": ["train"],
}

qu_meta_emb_model = ModelMeta(
    loader=partial(
        CustomWrapper,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        model_revision="2025062301"
        model_prompts={
           "query": "",
           "passage": "",
        },
    ),
    name="qu-q4-0-q3-em-0.6",
    languages=["zho-Hans"],
    open_weights=True,
    revision="111",
    release_date="2025-06-23",
    n_parameters=595776512,
    memory_usage_mb=17613,
    embed_dim=1024,
    max_tokens=32768,
    license="apache-2.0",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
    public_training_code=None,
    public_training_data=None,
    training_datasets=training_data,
)
