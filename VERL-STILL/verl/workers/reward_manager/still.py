# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing
from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from deepscaler.rewards.math_reward import deepscaler_reward_fn

from symeval import EvaluatorMathBatch


class MATHEvaluator:
    def __init__(self):
        # self.evaluator = EvaluatorMathBatch()
        self.evaluator = deepscaler_reward_fn

    def extract_answer_math(self, s):
        if ('<answer>' in s and "</answer>" in s):
            s = s.split("<answer>")[-1].strip().split("</answer>")[0].strip()
        ans = s.split("boxed")
        if len(ans) == 1:
            return s
        ans = ans[-1]
        if len(ans) == 0:
            return ""
        try:
            if ans[0] == "{":
                stack = 1
                a = ""
                for c in ans[1:]:
                    if c == "{":
                        stack += 1
                        a += c
                    elif c == "}":
                        stack -= 1
                        if stack == 0:
                            break
                        a += c
                    else:
                        a += c
            else:
                a = ans.split("$")[0].strip()
        except:
            return ""
        return a

    def score(self, solution_str, ground_truth):
        pool = multiprocessing.Pool(processes=1)
        async_result = pool.apply_async(self.evaluator, (solution_str, ground_truth))
        try:
            result = async_result.get(0.5)
        except multiprocessing.TimeoutError:
            print('TimeoutError')
            result = 0
        return result

        # return self.evaluator(
        #     solution_str=solution_str,
        #     ground_truth=ground_truth,
        # )

        if ('boxed' not in solution_str):
            return -1
        if (isinstance(ground_truth, list) == True):
            ground_truth = str(ground_truth[0])
        
        answers = [self.extract_answer_math(ground_truth)]
        preds = [self.extract_answer_math(solution_str)]
        scores = self.evaluator.batch_eq(ref_answers=answers, pred_answers=preds)

        if (scores[0] == True):
            return 1
        else:
            return 0


class STILLZero:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.compute_score = deepscaler_reward_fn
        self.compute_score = MATHEvaluator()
    
    def check_format(self, text):
        text = text.split("Assistant:")[-1].strip()
        text = text.replace("<|endoftext|>", "")
        text = text.replace("<|im_end|>", "")

        # Step 1: 检查正则表达式
        if (text.startswith('<think>') == False or text.endswith('</answer>') == False):
            return False

        # Step 2: 检查标签唯一性
        if text.count("<think>") != 1 or text.count("</think>") != 1:
            return False
        if text.count("<answer>") != 1 or text.count("</answer>") != 1:
            return False

        # Step 3: 检查标签顺序
        think_start = text.find("<think>")
        think_end = text.find("</think>")
        answer_start = text.find("<answer>")
        answer_end = text.find("</answer>")
        if think_end == -1 or answer_start == -1 or think_end > answer_start:
            return False
        if ((think_start < think_end and think_end < answer_start and answer_start < answer_end) == False):
            return False

        # Step 4: 检查内容非空
        think_text = text.split("<think>")[-1].strip().split("</think>")[0].strip()
        mid_text = text.split("</think>")[-1].strip().split("<answer>")[0].strip()
        answer_text = text.split('<answer>')[-1].strip().split("</answer>")[0].strip()
        if len(think_text) <= 0 or len(answer_text) <= 0 or len(mid_text) > 0:
            return False

        return True

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            assert(isinstance(sequences_str, str))
            assert(isinstance(ground_truth, str) or isinstance(ground_truth, list))
            score = self.compute_score.score(
                # data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            if (score < 0):
                score = 0

            if (self.check_format(sequences_str) == False):
                score = 0

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


class STILLVanilla:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = MATHEvaluator()

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            assert(isinstance(sequences_str, str))
            assert(isinstance(ground_truth, str) or isinstance(ground_truth, list))
            score = self.compute_score.score(
                # data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )
            if (score < 0):
                score = 0
            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor


    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = MATHEvaluator()

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            assert(isinstance(sequences_str, str))
            assert(isinstance(ground_truth, str) or isinstance(ground_truth, list))
            score = self.compute_score.score(
                # data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
            )

            is_slowthink = False
            if ('slow think' in sequences_str.lower() or '<think>' in sequences_str.lower()):
                is_slowthink = True
            
            if (is_slowthink == True): # Slow Thinking
                if (score > 0.5): # Correct
                    score = 0.8
                else: # Incorrect
                    score = 0
            else: # Fast Thinking
                if (score > 0.5): # Correct
                    score = 1.0
                else: # Incorrect
                    score = -0.2

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        return reward_tensor