import os
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from .config import BASE_DATA_DIR
from .dataset import MVBenchDataset
from .model import LLaVAModel

def main():
    # Initialize dataset and dataloader
    dataset = MVBenchDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = LLaVAModel()
    
    # Create results directory
    results_dir = Path("../results")
    results_dir.mkdir(exist_ok=True)
    
    # Evaluation loop
    results = []
    for batch in tqdm(dataloader, desc="Evaluating"):
        video = batch['video'].to(model.device)
        question = batch['question']
        candidates = batch['candidates']
        answer = batch['answer']
        task_type = batch['task_type']
        
        response = model.evaluate(video, question, candidates, answer)
        
        results.append({
            'task_type': task_type,
            'question': question,
            'candidates': candidates,
            'answer': answer,
            'response': response
        })
    
    # Save results
    import json
    with open(results_dir / "evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate accuracy per task type
    task_accuracies = {}
    for result in results:
        task_type = result['task_type']
        if task_type not in task_accuracies:
            task_accuracies[task_type] = {'correct': 0, 'total': 0}
        
        task_accuracies[task_type]['total'] += 1
        if result['response'].strip().upper() == result['answer'].strip().upper():
            task_accuracies[task_type]['correct'] += 1
    
    # Print results
    print("\nResults by task type:")
    for task_type, acc in task_accuracies.items():
        accuracy = (acc['correct'] / acc['total']) * 100
        print(f"{task_type}: {accuracy:.2f}% ({acc['correct']}/{acc['total']})")

if __name__ == "__main__":
    main() 