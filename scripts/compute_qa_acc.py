import json
import sys
import argparse
from collections import defaultdict

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Searcher: Video Frame Search and QA Tool")

    parser.add_argument('--answer_type', type=str, default="adaptive_pred_answer", help='answer type(uniform or adaptive).')
    parser.add_argument('--qa_path', type=str, default="./runs/qa/qa_LongVideoBench.json", help='qa result json path')
    return parser.parse_args() 

def load_data(file_path,answer_type="uniform_pred_answer"):
    """加载并验证JSON数据结构"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("JSON文件根元素应为列表结构")
            
        required_fields = {'duration_group', 'answer',answer_type }
        for i, item in enumerate(data):
            if not required_fields.issubset(item.keys()):
                missing = required_fields - item.keys()
                raise ValueError(f"条目{i}缺少必要字段: {missing}")
                
        return data
    except FileNotFoundError:
        sys.exit(f"错误：文件 {file_path} 不存在")
    except json.JSONDecodeError:
        sys.exit(f"错误：文件 {file_path} 不是有效的JSON格式")
    except ValueError as ve:
        sys.exit(f"数据结构错误: {ve}")

def calculate_accuracy(data,answer_type="uniform_pred_answer"):
    """带错误统计的准确率计算"""
    stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'errors': 0})
    global_errors = 0
    
    for item in data:
        duration = item["duration_group"]
        
        if 'error' in item or item[answer_type] == "":
            stats[duration]['errors'] += 1
            global_errors += 1
            continue
            
        stats[duration]['total'] += 1
        if item[answer_type] == item["answer"]:
            stats[duration]['correct'] += 1
            
    # 计算最终结果
    results = {
        'duration_stats': {},
        'global_errors': global_errors
    }
    
    total_num = 0
    correct_num = 0

    for duration, counts in stats.items():
        valid_samples = counts['total']
        total_num += counts['total']
        acc = counts['correct'] / valid_samples if valid_samples > 0 else 0.0
        correct_num += counts['correct']        
        results['duration_stats'][duration] = {
            'accuracy': acc,
            'valid_samples': valid_samples,
            'error_samples': counts['errors']
        }
    
    print("{:<20} | {}".format("Total average accuracy", correct_num / total_num))

    return results

def print_results(results):
    """增强版结果输出"""
    print("\n{:<10} | {:<8} | {:<12} | {}".format(
        "Duration", "Accuracy", "Valid Samples", "Error Samples"))
    print("-" * 50)
    
    # 按duration分组输出
    for duration in sorted(results['duration_stats'].keys()):
        info = results['duration_stats'][duration]
        print("{:<10} | {:>7.2%} | {:>12} | {:>12}".format(
            duration, 
            info['accuracy'],
            info['valid_samples'],
            info['error_samples']
        ))
    
    # 全局统计
    print("\n{:<20} | {}".format("Total Error Samples", results['global_errors']))
    total_valid = sum(v['valid_samples'] for v in results['duration_stats'].values())
    print("{:<20} | {}".format("Total Valid Samples", total_valid))



if __name__ == "__main__":
    args = parse_arguments()
    data = load_data(args.qa_path, answer_type=args.answer_type)
    results = calculate_accuracy(data,answer_type=args.answer_type)
    print_results(results)