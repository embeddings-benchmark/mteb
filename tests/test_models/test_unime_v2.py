import torch

import mteb
from mteb.mocks.mock_tasks import MockRetrievalTask
from mteb.models.model_implementations.unime_v2_models import UniMEV2Wrapper
from mteb.types import PromptType


def test_unime_v2_pooling_with_left_padding():
    hidden_states = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [3.0, 4.0]],
            [[0.0, 0.0], [0.0, 2.0], [5.0, 0.0]],
        ]
    )
    attention_mask = torch.tensor([[0, 1, 1], [0, 1, 1]])

    embeddings = UniMEV2Wrapper._pooling(hidden_states, attention_mask)

    expected = torch.tensor([[0.6, 0.8], [1.0, 0.0]])
    assert torch.allclose(embeddings, expected)


def test_unime_v2_pooling_without_left_padding():
    hidden_states = torch.tensor(
        [
            [[0.0, 0.0], [3.0, 4.0], [9.0, 9.0]],
            [[0.0, 5.0], [9.0, 9.0], [9.0, 9.0]],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    embeddings = UniMEV2Wrapper._pooling(hidden_states, attention_mask)

    expected = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
    assert torch.allclose(embeddings, expected)


def test_unime_v2_builds_multimodal_conversation():
    image = object()
    video = object()

    conversation = UniMEV2Wrapper._build_conversation(
        text="sample text",
        image=image,
        video=video,
        instruction="Retrieve the matching item.",
    )

    assert conversation == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "video", "video": video},
                {
                    "type": "text",
                    "text": "Retrieve the matching item.\nsample text",
                },
            ],
        }
    ]


def test_unime_v2_uses_default_media_prompts():
    image = object()
    video = object()

    image_conversation = UniMEV2Wrapper._build_conversation(
        text=None,
        image=image,
        video=None,
        instruction="",
    )
    video_conversation = UniMEV2Wrapper._build_conversation(
        text=None,
        image=None,
        video=video,
        instruction="",
    )

    assert image_conversation[0]["content"][-1] == {
        "type": "text",
        "text": "Find an image caption describing the given image.",
    }
    assert video_conversation[0]["content"][-1] == {
        "type": "text",
        "text": "Describe this video in detail.",
    }


def test_unime_v2_processes_variable_length_videos_individually():
    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": "one"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "two"}]}],
    ]

    batches = UniMEV2Wrapper._conversation_batches(conversations, has_video=True)

    assert batches == [[conversations[0]], [conversations[1]]]


def test_unime_v2_keeps_non_video_conversations_batched():
    conversations = [
        [{"role": "user", "content": [{"type": "text", "text": "one"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "two"}]}],
    ]

    batches = UniMEV2Wrapper._conversation_batches(conversations, has_video=False)

    assert batches == [conversations]


def test_unime_v2_model_meta_is_registered():
    meta = mteb.get_model_meta("TianchengGu/UniME-V2-LLaVA-OneVision-8B")

    assert meta.embed_dim == 3_584
    assert meta.modalities == ["text", "image", "video"]
    assert meta.revision == "36ef54da9f3dc3a2bfba115c4fb403b1a1f7cb0c"


def test_unime_v2_uses_generic_query_instruction_for_tasks_without_prompt():
    wrapper = object.__new__(UniMEV2Wrapper)
    wrapper.use_task_instructions = True
    task = MockRetrievalTask()

    instruction = wrapper._get_instruction(task.metadata, PromptType.query)

    assert instruction == "Represent the given input for retrieval."


def test_unime_v2_can_disable_task_instructions():
    wrapper = object.__new__(UniMEV2Wrapper)
    wrapper.use_task_instructions = False
    task = MockRetrievalTask()

    instruction = wrapper._get_instruction(task.metadata, PromptType.query)

    assert not instruction
