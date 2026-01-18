#!/usr/bin/env python3
"""
analyze_correlation.pyを並列実行し、全体の進捗をtqdmで表示
"""
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm
import argparse

def run_single_task(layer1, layer2):
    """単一タスクを実行"""
    cmd = ["uv", "run", "analyze_correlation.py", "--layer1", str(layer1), "--layer2", str(layer2)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return (layer1, layer2, False, result.stderr)
        return (layer1, layer2, True, None)
    except Exception as e:
        return (layer1, layer2, False, str(e))

def main():
    parser = argparse.ArgumentParser(description="Run correlation analysis in parallel")
    parser.add_argument("--workers", type=int, default=32, help="Number of parallel workers")
    parser.add_argument("--start", type=int, default=0, help="Start layer (inclusive)")
    parser.add_argument("--end", type=int, default=32, help="End layer (exclusive)")
    args = parser.parse_args()
    
    # タスクリストを生成
    tasks = list(product(range(args.start, args.end), range(args.start, args.end)))
    
    print(f"Total tasks: {len(tasks)}")
    print(f"Workers: {args.workers}")
    print(f"Layer range: {args.start}-{args.end-1}")
    print()
    
    # 並列実行
    failed_tasks = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 全タスクをサブミット
        future_to_task = {
            executor.submit(run_single_task, layer1, layer2): (layer1, layer2)
            for layer1, layer2 in tasks
        }
        
        # 進捗バーで表示
        with tqdm(total=len(tasks), desc="Processing", unit="task") as pbar:
            for future in as_completed(future_to_task):
                layer1, layer2, success, error = future.result()
                pbar.update(1)
                
                if not success:
                    failed_tasks.append((layer1, layer2, error))
                    tqdm.write(f"Failed: layer1={layer1}, layer2={layer2}")
                    if error:
                        tqdm.write(f"   Error: {error[:100]}")
    
    # 結果のサマリー
    print("\n" + "="*50)
    print(f"Completed: {len(tasks) - len(failed_tasks)}/{len(tasks)} tasks")
    
    if failed_tasks:
        print(f"Failed: {len(failed_tasks)} tasks")
        print("\nFailed tasks:")
        for layer1, layer2, error in failed_tasks:
            print(f"  - layer1={layer1}, layer2={layer2}")
            if error and len(error) < 200:
                print(f"    {error}")

if __name__ == "__main__":
    main()
