# ref. https://github.com/JunjH/Revisiting_Single_Depth_Estimation/blob/master/util.py

import dataclasses
import math
from typing import Tuple

import torch
import numpy as np
from torch import Tensor


@dataclasses.dataclass
class DepthMetrics(object):
    mse: float = 0.
    mae: float = 0.
    abs_rel: float = 0.
    lg10: float = 0.
    delta1: float = 0.
    delta2: float = 0.
    delta3: float = 0.


@dataclasses.dataclass
class EdgeMetrics(object):
    accuracy: float = 0.
    precision: float = 0.
    recall: float = 0.
    f1_score: float = 0.


def evaluate_edge_metrics(output_grad_xy: Tensor, depth_grad_xy: Tensor,
                          threshold: float = 0.25) -> EdgeMetrics:

    # calc edge valid
    depth_edge = torch.sqrt(
        torch.pow(depth_grad_xy[:, 0, :, :], 2) + torch.pow(depth_grad_xy[:, 1, :, :], 2))
    depth_edge_valid: Tensor = (depth_edge > threshold)

    output_edge = torch.sqrt(
        torch.pow(output_grad_xy[:, 0, :, :], 2) + torch.pow(output_grad_xy[:, 1, :, :], 2))
    output_edge_valid: Tensor = (output_edge > threshold)

    # count true pixels
    n_equal = np.sum(torch.eq(depth_edge_valid, output_edge_valid).float().data.cpu().numpy())
    n_equal_pos = np.sum((depth_edge_valid * output_edge_valid).float().data.cpu().numpy())

    # calc metrics
    n_total = depth_grad_xy.size(2) * depth_grad_xy.size(3)
    accuracy = n_equal / n_total
    n_out_pos = (np.sum(output_edge_valid.data.cpu().numpy()))
    precision = n_equal_pos / n_out_pos if n_out_pos else 0
    recall = n_equal_pos / (np.sum(depth_edge_valid.data.cpu().numpy()))
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall else 0

    metrics = EdgeMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score
    )
    return metrics


def evaluate_depth_metrics(output: Tensor, target: Tensor) -> DepthMetrics:

    _output, _target, nan_mask, n_valid_element = set_nan_to_zero(output, target)

    if n_valid_element.data.cpu().numpy():

        # calc diff
        diff_matrix = torch.abs(_output - _target)

        # mse, mae
        mse = torch.sum(torch.pow(diff_matrix, 2)) / n_valid_element
        mae = torch.sum(diff_matrix) / n_valid_element

        # abs rel
        real_matrix = torch.div(diff_matrix, _target)
        real_matrix[nan_mask] = 0
        abs_rel = torch.sum(real_matrix) / n_valid_element

        # lg10
        lg10_matrix = torch.abs(calc_lg10(_output) - calc_lg10(_target))
        lg10_matrix[nan_mask] = 0
        lg10 = torch.sum(lg10_matrix) / n_valid_element

        # delta
        y_over_z = torch.div(_output, _target)
        z_over_y = torch.div(_target, _output)
        max_ratio = max_of_two(y_over_z, z_over_y)
        delta1 = torch.sum(
            torch.le(max_ratio, 1.25).float()) / n_valid_element
        delta2 = torch.sum(
            torch.le(max_ratio, math.pow(1.25, 2)).float()) / n_valid_element
        delta3 = torch.sum(
            torch.le(max_ratio, math.pow(1.25, 3)).float()) / n_valid_element

        metrics = DepthMetrics(
            mse=float(mse.data.cpu().numpy()),
            mae=float(mae.data.cpu().numpy()),
            abs_rel=float(abs_rel.data.cpu().numpy()),
            lg10=float(lg10.data.cpu().numpy()),
            delta1=float(delta1.data.cpu().numpy()),
            delta2=float(delta2.data.cpu().numpy()),
            delta3=float(delta3.data.cpu().numpy())
        )

    else:
        metrics = DepthMetrics()

    return metrics


def calc_lg10(x: Tensor) -> Tensor:
    return torch.div(torch.log(x), math.log(10))


def max_of_two(x: Tensor, y: Tensor) -> Tensor:
    z = x.clone()
    mask_y_larger = torch.lt(x, y)
    z[mask_y_larger.detach()] = y[mask_y_larger.detach()]
    return z


def get_n_valid(x: Tensor) -> Tensor:
    return torch.sum(torch.eq(x, x).float())


def get_n_nan_element(x: Tensor) -> Tensor:
    return torch.sum(torch.ne(x, x).float())


def get_nan_mask(x: Tensor) -> Tensor:
    return torch.ne(x, x)


def set_nan_to_zero(input: Tensor, target: Tensor
                    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    nan_mask = get_nan_mask(target)
    n_valid_element = get_n_valid(target)

    _input = input.clone()
    _target = target.clone()

    _input[nan_mask] = 0
    _target[nan_mask] = 0

    return _input, _target, nan_mask, n_valid_element
