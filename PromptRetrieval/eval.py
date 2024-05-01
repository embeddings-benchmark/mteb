import os
import nltk
import json
import datasets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True,type=str,choices=['mnli','rte','mrpc','sst5','dbpedia_14','nq','xsum','hellaswag'])
parser.add_argument('--output_dir', required=True, type=str)
args = parser.parse_args()

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

label_map = {
    'mnli': {
        0: 'entailment',
        1: 'neutral',
        2: 'contradiction'
    },
    'rte': {
        0: 'entailment',
        1: 'contradiction'
    },
    'sst5': {
        0: 'very negative',
        1: 'negative',
        2: 'neutral',
        3: 'positive',
        4: 'very positive'
    },
    'dbpedia_14': {
        0: 'company',
        1: 'education',
        2: 'artist',
        3: 'athlete',
        4: 'officeHolder',
        5: 'transportation',
        6: 'building',
        7: 'nature',
        8: 'village',
        9: 'animal',
        10: 'plant',
        11: 'album',
        12: 'film',
        13: 'book',
    },
    'hellaswag': {
        '0': '(A)',
        '1': '(B)',
        '2': '(C)',
        '3': '(D)',
    },
}
args.output_dir = os.path.join(args.output_dir,args.task)
correct = 0
total = 0
golds = []
preds = []
for file in os.listdir(args.output_dir):
    if not file.endswith('.json'):
        continue
    with open(os.path.join(args.output_dir,file)) as f:
        e = json.load(f)
    e['pred'] = e['pred'].lower()
    pred = e['pred'].split('\n')[0].strip().lower()
    if args.task in ['mnli','rte'] and label_map[args.task][e['label']]==pred:
        correct += 1
    elif args.task in ['xsum']:
        preds.append(e['pred'])
        golds.append(e['summary'].lower())
    elif args.task in ['sst5']:
        if 'neutral' in e['pred'] and e['label']==2:
            correct += 1
        elif 'very negative' in e['pred'] and e['label']==0:
            correct += 1
        elif 'negative' in e['pred'] and e['label']==1:
            correct += 1
        elif 'very positive' in e['pred'] and e['label']==4:
            correct += 1
        elif 'positive' in e['pred'] and e['label']==3:
            correct += 1
    elif args.task in ['dbpedia_14','hellaswag']:
        # if file=='0.json':
        #     print(label_map[args.task][e['label']].lower())
        #     print(e['pred'])
        if label_map[args.task][e['label']].lower() in e['pred']:
            # print(file)
            correct += 1
    elif args.task in ['nq']:
        for a in e['answers']:
            if a.lower() in e['pred']:
                correct += 1
                break
    elif args.task in ['mrpc'] and (('not' in pred and e['label']==0) or ('not' not in pred and e['label']==1)):
        correct += 1
    total += 1
if args.task in ['xsum']:
    import evaluate
    rouge = evaluate.load('rouge')
    preds, golds = postprocess_text(preds,golds)
    results = rouge.compute(predictions=preds,
                            references=golds)
    print(results)
    with open(os.path.join(args.output_dir,'result.txt'),'w') as f:
        f.write(f"{results}")
else:
    print("Correct:",correct)
    print("Total:",total)
    print("Accuracy:",correct/total)
    with open(os.path.join(args.output_dir,'result.txt'),'w') as f:
        f.write(f"Correct: {correct}\n")
        f.write(f"Total: {total}\n")
        f.write(f"Accuracy: {correct/total}\n")
