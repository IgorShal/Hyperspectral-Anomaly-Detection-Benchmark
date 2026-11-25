"""Select the highest scene AUC checkpoint for the scene benchmark."""

from copy import deepcopy

import torch
from sklearn.metrics import roc_auc_score

from scripts.comparison_helpers import validation_stage


class BestSceneEpoch:
    def __init__(self, model, predict, mask, scene_state=None, *, validation=None):
        self.model = model
        self.predict = predict
        self.labels = mask.ravel() > 0
        self.scene_state = scene_state
        self.best_epoch = None
        self.best_auc = float("-inf")
        self.last_auc = None
        self.auc_curve = []
        self.checkpoint = None
        self.checkpoint_scene = None
        self.validation = validation

    @torch.inference_mode()
    def __call__(self, epoch):
        with validation_stage(self.validation):
            training = self.model.training
            self.model.eval()
            try:
                auc = float(roc_auc_score(self.labels, self.predict().ravel()))
                self.last_auc = auc
                self.auc_curve.append(auc)
                if auc > self.best_auc:
                    self.best_epoch = epoch
                    self.best_auc = auc
                    self.checkpoint = deepcopy(self.model.state_dict())
                    if self.scene_state is not None:
                        self.checkpoint_scene = self.scene_state.clone()
            finally:
                self.model.train(training)

    def restore(self):
        if self.best_epoch is None:
            raise RuntimeError("No epoch was evaluated")
        with validation_stage(self.validation, check=False):
            self.model.load_state_dict(self.checkpoint)
            if self.scene_state is not None:
                self.scene_state.copy_(self.checkpoint_scene)
            self.model.eval()
            self.checkpoint = self.checkpoint_scene = None
        return self.best_epoch


class SceneEpochCurve:
    def __init__(self, model, predict, mask, *, validation=None):
        self.model = model
        self.predict = predict
        self.labels = mask.ravel() > 0
        self.auc_curve = []
        self.validation = validation

    @torch.inference_mode()
    def __call__(self, epoch):
        with validation_stage(self.validation):
            self.model.eval()
            try:
                self.auc_curve.append(float(roc_auc_score(self.labels, self.predict().ravel())))
            finally:
                self.model.train()

    def restore(self):
        self.model.eval()
        return len(self.auc_curve)


def epoch_evaluator(model, predict, mask, collect_curve, fixed_epoch, *, validation=None):
    if fixed_epoch is not None:
        return None
    evaluator = SceneEpochCurve if collect_curve else BestSceneEpoch
    return evaluator(model, predict, mask, validation=validation)


def finish_epoch_evaluation(model, selection, fixed_epoch):
    if selection is None:
        model.eval()
        return fixed_epoch
    return selection.restore()
