import json
from .vqa_score import extract_answer

def imagenet_eval(results_file, use_extract_answer=True):
    answers = json.load(open(results_file))
    for item in answers:
        answer = item['caption']
        
        if use_extract_answer:
            answer = extract_answer(answer)
    
        
        answer = answer.replace(':', ' ')
        answer = answer.replace('[', ' ')
        answer = answer.replace(']', ' ')
        item['caption'] = answer
        
    if use_extract_answer:
        with open(results_file.replace('.json', '_processed.json'), 'w') as file:
            json.dump(answers, file)
    
    res={}
    count=0
    suss=0
    for item in answers:

        pred_answer=item['caption'].replace('\n', ' ')
        pred_answer = pred_answer.replace('\t', ' ')
        pred_answer = pred_answer.strip()

        gt_answer=item['gt_caption'].replace('\n', ' ')
        gt_answer = gt_answer.replace('\t', ' ')
        gt_answer = gt_answer.strip()
        count+=1
        if gt_answer.lower()==pred_answer.lower():
            suss+=1

    res['overall_accuracy']=suss/count*100

    return res