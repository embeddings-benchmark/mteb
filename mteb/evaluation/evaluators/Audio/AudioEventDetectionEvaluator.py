from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import medfilt


def onset_f_measure(
    predictions: dict[str, list[dict[str, Any]]],
    references: dict[str, list[dict[str, Any]]],
    t_collar: float = 0.2,
    labels: set[str] | None = None,
) -> tuple[tuple[str, float], ...]:
    """Calculate onset F-measure for audio event detection.

    This function evaluates how well predicted event onsets match reference onsets,
    considering only the onset (start time) and ignoring the offset (end time).

    Args:
        predictions: dictionary mapping file_id to list of predicted events
                    Each event must have keys: "label", "start"
        references: dictionary mapping file_id to list of reference events
                    Each event must have keys: "label", "start"
        t_collar: Time collar in seconds for matching onsets
                  (predicted onset must be within t_collar of reference onset)
        labels: set of labels to evaluate (if None, uses all labels in references)

    Returns:
        tuple of tuples containing the metric name and value:
        (
            ("f_measure", float),
            ("precision", float),
            ("recall", float)
        )
    """
    if labels is None:
        labels = set()
        for file_events in references.values():
            for event in file_events:
                labels.add(event["label"])

    n_ref = 0
    n_pred = 0
    n_correct = 0

    for file_id in set(predictions.keys()).union(references.keys()):
        ref_events = references.get(file_id, [])
        pred_events = predictions.get(file_id, [])

        ref_events = [e for e in ref_events if e["label"] in labels]
        pred_events = [e for e in pred_events if e["label"] in labels]

        ref_by_label = defaultdict(list)
        pred_by_label = defaultdict(list)

        for event in ref_events:
            ref_by_label[event["label"]].append(event)

        for event in pred_events:
            pred_by_label[event["label"]].append(event)

        for label in labels:
            label_ref_events = ref_by_label[label]
            label_pred_events = pred_by_label[label]

            n_ref += len(label_ref_events)
            n_pred += len(label_pred_events)

            correct_matches = match_events_by_onset(
                pred_events=label_pred_events,
                ref_events=label_ref_events,
                t_collar=t_collar,
            )

            n_correct += len(correct_matches)

    precision = n_correct / max(n_pred, 1)
    recall = n_correct / max(n_ref, 1)

    if precision + recall > 0:
        f_measure = 2 * precision * recall / (precision + recall)
    else:
        f_measure = 0.0

    return (("f_measure", f_measure), ("precision", precision), ("recall", recall))


def match_events_by_onset(
    pred_events: list[dict[str, Any]], ref_events: list[dict[str, Any]], t_collar: float
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match predicted events to reference events based on onset time.

    Args:
        pred_events: list of predicted events (must have "start" key)
        ref_events: list of reference events (must have "start" key)
        t_collar: Time collar in seconds for matching

    Returns:
        list of tuples containing matched (prediction, reference) pairs
    """
    pred_events = sorted(pred_events, key=lambda e: e["start"])
    ref_events = sorted(ref_events, key=lambda e: e["start"])

    cost_matrix = np.zeros((len(pred_events), len(ref_events)))
    valid_pairs = set()

    for i, pred in enumerate(pred_events):
        for j, ref in enumerate(ref_events):
            time_diff = abs(pred["start"] - ref["start"])

            if time_diff <= t_collar * 1000:
                cost_matrix[i][j] = time_diff
                valid_pairs.add((i, j))
            else:
                cost_matrix[i][j] = float("inf")

    matches = []
    used_pred = set()
    used_ref = set()
    sorted_pairs = sorted(valid_pairs, key=lambda pair: cost_matrix[pair[0]][pair[1]])

    for pred_idx, ref_idx in sorted_pairs:
        if pred_idx not in used_pred and ref_idx not in used_ref:
            matches.append((pred_events[pred_idx], ref_events[ref_idx]))
            used_pred.add(pred_idx)
            used_ref.add(ref_idx)

    return matches


class EventDetector:
    def __init__(self, seed=42, frame_rate=100):
        self.seed = seed
        self.frame_rate = frame_rate
        self.frame_duration = 1.0 / frame_rate
        self.model = None
        self.classes_ = []
        self.label_to_idx = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        torch.manual_seed(seed)
        np.random.seed(seed)

    def fit(self, X_train: list[np.ndarray], y_train: list[list[dict]]):
        """Train frame-level classifier on audio embeddings"""
        all_embeddings, all_labels = self._process_training_data(X_train, y_train)
        self._init_model(input_dim=all_embeddings.shape[1])
        X_tensor = torch.tensor(all_embeddings, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(all_labels, dtype=torch.float32).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        # Training loop
        self.model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def _init_model(self, input_dim):
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(self.classes_)),
            nn.Sigmoid(),
        ).to(self.device)

    def _process_training_data(self, X_train, y_train):
        self.classes_ = sorted({e["label"] for sample in y_train for e in sample})
        self.label_to_idx = {lbl: i for i, lbl in enumerate(self.classes_)}
        all_embs = []
        all_labels = []
        for embeddings, sample_events in zip(X_train, y_train):
            num_frames = embeddings.shape[0]
            frame_labels = np.zeros((num_frames, len(self.classes_)))
            for event in sample_events:
                start_frame = int(event["start"] * self.frame_rate)
                end_frame = int(event["end"] * self.frame_rate)
                lbl_idx = self.label_to_idx[event["label"]]
                frame_labels[start_frame:end_frame, lbl_idx] = 1

            all_embs.append(embeddings)
            all_labels.append(frame_labels)

        return np.vstack(all_embs), np.vstack(all_labels)

    def predict(self, X_test: list[np.ndarray]) -> list[list[dict]]:
        self.model.eval()
        pred_events = []

        with torch.no_grad():
            for embeddings in X_test:
                inputs = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
                outputs = self.model(inputs).cpu().numpy()
                binary_pred = (outputs >= 0.5).astype(int)
                filtered = medfilt(binary_pred, kernel_size=(25, 1))  # 250ms window
                events = self._predictions_to_events(filtered)
                pred_events.append(events)

        return pred_events

    def _predictions_to_events(self, predictions: np.ndarray) -> list[dict]:
        events = []
        for c, label in enumerate(self.classes_):
            changes = np.diff(predictions[:, c], prepend=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            if len(starts) > len(ends):
                ends = np.append(ends, len(predictions) - 1)

            for s, e in zip(starts, ends):
                events.append(
                    {
                        "label": label,
                        "start": s * self.frame_duration,
                        "end": e * self.frame_duration,
                    }
                )
        return events
