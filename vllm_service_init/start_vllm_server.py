#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'timeout-decorator' library to apply fine-grained timeout control 
directly to the `grade_answer` function. This approach is more robust than setting a global timeout for the 
entire post-processing step, as it prevents the loss of an entire question's result due to a single, 
faulty comparison.

Setup Instructions:
    # 1. Install the required library
    pip install timeout-decorator

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
from transformers import AutoTokenizer
from mathruler.grader import extract_boxed_content, grade_answer
import timeout_decorator  # 1. Import the timeout-decorator library

# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.8,
                    help='The maximum GPU memory utilization fraction for vLLM.')
args = parser.parse_args()

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
print('[init] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
)

sample_params = vllm.SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10, # Generate 10 candidate answers for each question
)

# ---------------------- GPU Idle Utilization Thread ---------------------- #
# (This section remains unchanged)
stop_event = threading.Event()    # Event to stop the thread globally
pause_event = threading.Event()   # Event to pause the thread during requests

def gpu_idle_worker():
    '''
    This worker occupies the GPU with a continuous matrix multiplication loop when idle,
    preventing potential performance drops from GPU power state changes.
    '''
    print('[idle_worker] GPU idle worker started.')
    running = True
    while not stop_event.is_set():
        if pause_event.is_set():
            if running:
                print('[idle_worker] Paused.')
                running = False
            time.sleep(0.1) # Sleep briefly while paused
            continue
        else:
            if not running:
                print('[idle_worker] Resumed.')
                running = True
        try:
            # A simple but effective way to keep the GPU busy
            a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
            torch.matmul(a, b)
            torch.cuda.synchronize()
        except RuntimeError as e:
            print(f'[idle_worker] Caught a RuntimeError: {e}. Sleeping for 1s...')
            time.sleep(1)
    print('[idle_worker] GPU idle worker stopped.')

idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
idle_thread.start()

# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Instead of a custom timeout function, we now wrap `grade_answer` with a decorator.
@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a 2-second timeout to each individual `grade_answer` call.
    If the function's execution exceeds 2 seconds, it will raise a `TimeoutError`.
    """
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    pause_event.set()
    torch.cuda.synchronize()

    name = request.args.get('name', 'None')
    print(f'[server] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)
    os.remove(name)

    questions = [item.get('question', '') for item in data]
    answers   = [item.get('answer',   '') for item in data]

    # (Data preparation logic remains unchanged)
    valid_indices, valid_questions, valid_answers, valid_chats = [], [], [], []
    for i, (q, a) in enumerate(zip(questions, answers)):
        if q and a:
            valid_indices.append(i)
            valid_questions.append(q)
            valid_answers.append(a)
            valid_chats.append([
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user',   'content': q}
            ])
    print('[server] Valid chat prompts have been prepared.')

    # ---------- vLLM Generation ----------
    # (vLLM generation logic remains unchanged)
    if valid_chats:
        if tokenizer.chat_template:
            prompts = [
                tokenizer.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True, add_special_tokens=True)
                for chat in valid_chats
            ]
        else:
            prompts = [
                'system: ' + chat[0]['content'] + '\n' + 'user: ' + chat[1]['content']
                for chat in valid_chats
            ]
        responses = model.generate(prompts, sampling_params=sample_params, use_tqdm=True)
    else:
        responses = []
    print('[server] Generation completed.')

    # ---------- Results Post-Processing (Core Refactoring Here) ----------
    def process_single(question, golden_answer, response):
        '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
        results = [extract_boxed_content(out.text) for out in response.outputs]
        print(f"[process_single] Processing question: '{question[:70]}...'")

        answer_counts = {}
        for res in results:
            if not res: continue # Skip empty results
            matched = False
            
            # 3. This inner loop now handles timeouts for each individual comparison.
            for exist_ans in list(answer_counts.keys()):
                try:
                    # Use the new timeout-wrapped function for the comparison logic.
                    if (res == exist_ans or 
                        ('no ' in res.lower() and 'no ' in exist_ans.lower()) or
                        grade_answer_with_timeout(res, exist_ans) or 
                        grade_answer_with_timeout(exist_ans, res)):
                        answer_counts[exist_ans] += 1
                        matched = True
                        break # Match found, break from the inner loop
                        
                except TimeoutError:
                    # If a single comparison times out, log it and gracefully continue.
                    print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'. Skipping this pair.")
                    continue # Continue to the next comparison
                except Exception as e:
                    # Catch any other potential errors from the grader function.
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score if majority_ans == golden_answer and score > 0.1 else 0,
            'results':  results
        }

    results_all = []
    response_idx = 0
    for q, a in zip(questions, answers):
        try:
            if q and a:
                response = responses[response_idx]
                response_idx += 1
                # 4. Directly call `process_single`. The complex timeout logic is now handled internally.
                item = process_single(q, a, response)
                results_all.append(item)
            else:
                results_all.append({'question': q, 'answer': a, 'score': -1, 'results': []})
        except Exception as e:
            # 5. Catch any other unexpected exceptions from within process_single.
            print(f'[server] CRITICAL: An unhandled error occurred while processing question: {q}')
            print(f'[server] Error details: {e}')
            results_all.append({
                'question': q,
                'answer':   a,
                'score':    -1,
                'results':  [],
                'error':    f'unhandled exception in process_single: {str(e)}'
            })
    print('[server] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    # --- Resume the GPU idle worker ---
    pause_event.clear()
    print(f'[server] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='127.0.0.1', port=int(args.port), threaded=True)
    finally:
        # Gracefully shut down the background thread on exit
        stop_event.set()
        idle_thread.join()
        print('[main] Application shutdown complete.')
