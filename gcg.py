
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import random
import json
import os
from torch.optim import Adam
import heapq

from optimizationBasedPromptInjectionAttack import JudgeDeceiver

class GCGJudgeDeceiver:
    def __init__(self, model, tokenizer, device="cuda", alpha=1.0, beta=0.1, 
                 beam_width=5, learning_rate=0.001):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.judge_deceiver = JudgeDeceiver()
        
        self.init_tokens = ["correct" for _ in range(20)]  # 默认20个token
        self.num_adv_tkns = len(self.init_tokens)
        
        # 添加attack token
        self.tokenizer.add_tokens([f"<attack_tok>"])
        self.adv_special_tkn_id = len(self.tokenizer) - 1
        self.special_tkns_txt = ''.join(
            ["<attack_tok>" for _ in range(self.num_adv_tkns)]
        )
        
        self.beam_width = beam_width
        self.learning_rate = learning_rate

    def attack(
        self,
        question: str,
        target_response: str,
        shadow_responses: List[List[str]],
        max_iterations: int = 600,
        batch_size: int = 8,
        top_k: int = 10
    ):
        C_R = 1
        curr_iteration = 0
        injected_seq = self.init_tokens
        best_loss = float('inf')
        best_seq = injected_seq

        for _ in tqdm(range(max_iterations)):

            judge_losses = self.judge_deceiver.calculateSumOfLossesAtDifferentPositions(
                input_ids=None,
                adv_ids=" ".join(injected_seq),  
                target_response=target_response,
                position=0
            )

            injected_seq = self.optimize_batch(
                question=question,
                target_response=target_response,
                shadow_responses=shadow_responses[:C_R],
                injected_seq=injected_seq,
                judge_losses=judge_losses,
                batch_size=batch_size,
                top_k=top_k
            )
            
            current_loss = judge_losses.item() if isinstance(judge_losses, torch.Tensor) else judge_losses
            if current_loss < best_loss:
                best_loss = current_loss
                best_seq = injected_seq.copy()

            if self.check_attack_success(
                question=question,
                target_response=target_response,
                shadow_responses=shadow_responses[:C_R],
                injected_seq=injected_seq
            ):
                C_R = min(C_R + 1, len(shadow_responses))

            curr_iteration += 1

        return " ".join(best_seq)

    def optimize_batch(
        self,
        question: str,
        target_response: str,
        shadow_responses: List[List[str]],
        injected_seq: List[str],
        judge_losses: float,
        batch_size: int,
        top_k: int
    ):
        # 转换注入序列为token ids
        adv_ids = self.tokenizer(
            " ".join(injected_seq),
            add_special_tokens=False,
            return_tensors='pt'
        )['input_ids'].squeeze().to(self.device)

        param_tensor = adv_ids.clone().float().requires_grad_(True)
        optimizer = Adam([param_tensor], lr=self.learning_rate)

        candidates = self.beam_search_optimize(
            question=question,
            target_response=target_response,
            shadow_responses=shadow_responses,
            param_tensor=param_tensor,
            injected_seq=injected_seq,
            optimizer=optimizer,
            batch_size=batch_size,
            top_k=top_k
        )

        # 返回最佳候选的token列表
        return self.tokenizer.decode(candidates[0][1]).split()

    def beam_search_optimize(
        self,
        question: str,
        target_response: str,
        shadow_responses: List[List[str]],
        param_tensor: torch.Tensor,
        injected_seq: List[str],
        optimizer: Adam,
        batch_size: int,
        top_k: int
    ) -> List[Tuple[float, torch.Tensor]]:
        beam = [(0.0, param_tensor.clone())]
        
        for _ in range(batch_size):
            new_candidates = []
            
            for beam_score, beam_tensor in beam:
                optimizer.zero_grad()
                total_loss = 0
                total_grad = None

                for responses in shadow_responses:
                    # 使用judge_deceiver优化目标
                    judge_loss = self.judge_deceiver.judgeDeceiver(
                        question=question,
                        Rs_list=responses,
                        r_target_i=target_response,
                        𝛿_injectedSeq=self.tokenizer.decode(beam_tensor).split(),
                        B_BatchSize=batch_size,
                        T_MaxNumOfIterations=10,
                        K=top_k,
                        alpha=self.alpha,
                        beta=self.beta,
                        model_name=self.model.__class__.__name__,
                        device=self.device
                    )
                    
                    aligned_loss, aligned_grad = self.compute_aligned_loss(
                        question, target_response, responses, beam_tensor
                    )
                    
                    enhancement_loss, enhancement_grad = self.compute_enhancement_loss(
                        question, target_response, responses, beam_tensor
                    )
                    
                    perplexity_loss = self.compute_perplexity_loss(beam_tensor)
                    
                    loss = (judge_loss +
                           aligned_loss + 
                           self.alpha * enhancement_loss +
                           self.beta * perplexity_loss)
                    
                    grad = (aligned_grad + 
                           self.alpha * enhancement_grad)
                    
                    total_loss += loss
                    if total_grad is None:
                        total_grad = grad
                    else:
                        total_grad += grad

                loss = total_loss
                loss.backward()
                optimizer.step()

                logits = self.model(input_ids=beam_tensor.unsqueeze(0)).logits
                top_values, top_indices = torch.topk(
                    F.softmax(logits[:, -1], dim=-1),
                    k=top_k
                )

                for value, index in zip(top_values[0], top_indices[0]):
                    new_tensor = beam_tensor.clone()
                    new_tensor[-1] = index
                    new_score = beam_score - torch.log(value).item()
                    new_candidates.append((new_score, new_tensor))

            beam = heapq.nsmallest(
                self.beam_width, 
                new_candidates,
                key=lambda x: x[0]
            )

        return beam

    def compute_aligned_loss(self, question, target_response, responses, adv_ids):
        """计算对齐生成损失"""
        attacked_response = target_response + f' {self.special_tkns_txt}'
        inputs = self.prepare_input(question, attacked_response, responses[0])
        
        grad, output = self.token_gradients(inputs, adv_ids, torch.LongTensor([0]).to(self.device))
        
        probs = F.softmax(output.logits, dim=-1)
        loss = -torch.log(probs[0][0])
        
        return loss, grad

    def compute_enhancement_loss(self, question, target_response, responses, adv_ids):
        """计算增强损失"""
        losses = []
        grads = []
        
        for pos in range(len(responses)):
            attacked_responses = responses.copy()
            attacked_responses[pos] = target_response + f' {self.special_tkns_txt}'
            
            inputs = self.prepare_input(question, attacked_responses[pos], responses[0])
            grad, output = self.token_gradients(
                inputs, adv_ids, torch.LongTensor([pos]).to(self.device)
            )
            
            logits = output.logits[:, pos, :]
            prob = F.softmax(logits, dim=-1)[0, pos]
            loss = -torch.log(prob)
            
            losses.append(loss)
            grads.append(grad)
            
        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(grads), dim=0)

    def compute_perplexity_loss(self, adv_ids):
        """计算困惑度损失"""
        log_probs = []
        for i in range(1, len(adv_ids)):
            prev_ids = adv_ids[:i]
            target_id = adv_ids[i]
            
            output = self.model(input_ids=prev_ids.unsqueeze(0))
            log_prob = F.log_softmax(output.logits[:, -1], dim=-1)
            log_probs.append(log_prob[0, target_id])
            
        return -torch.mean(torch.stack(log_probs))

    def token_gradients(self, input_ids, adv_ids, target):
        """计算token梯度"""
        assert len(input_ids.shape) == 1
        
        input_ids = input_ids.clone()
        attack_toks = (input_ids == self.adv_special_tkn_id)  
        input_ids[attack_toks] = adv_ids
        
        start, stop = np.where(attack_toks.cpu())[0][[0, -1]]
        input_slice = slice(start, stop+1)
        
        embed_weights = self.model.get_embedding_matrix()
        one_hot = torch.zeros(
            input_ids[input_slice].shape[0],
            embed_weights.shape[0],
            device=self.device,
            dtype=embed_weights.dtype
        )
        
        input_ids = input_ids.to(self.device)
        one_hot.scatter_(
            1,
            input_ids[input_slice].unsqueeze(1),
            torch.ones(one_hot.shape[0], 1, device=self.device)
        )
        
        one_hot.requires_grad_()
        input_embeds = (one_hot @ embed_weights).unsqueeze(0)
        
        embeds = self.model.get_embeddings(input_ids.unsqueeze(0)).detach()
        full_embeds = torch.cat([
            embeds[:,:input_slice.start,:],
            input_embeds,
            embeds[:,input_slice.stop:,:]
        ], dim=1)
        
        output = self.model.forward(inputs_embeds=full_embeds)
        loss = F.cross_entropy(output.logits, target)
        loss.backward()
        
        return one_hot.grad.clone(), output
'''
    def check_attack_success(self, question, target_response, shadow_responses, injected_seq):
        """评估攻击是否成功"""
        success = True
        for responses in shadow_responses:
            result = self.judge_deceiver.sucessfullyAttacksAllPositionsInRange(
                start=0,
                end=len(responses),
                seq=injected_seq  # 直接使用token列表
            )
            if not result:
                success = False
                break
        return success
'''