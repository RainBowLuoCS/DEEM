from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from pycocotools.coco import COCO
import json

def ref_caption_eval(
    results_file,
    use_1st_sentence_only=False,
):

    # create coco object and coco_result object


    with open(results_file) as f:
        anns = json.load(f)
    if use_1st_sentence_only:
        for ann in anns:
            ann["caption"] = ann["caption"].split(".")[0]

    coco_result = {str(idx):[{'caption':ann["caption"]}] for idx, ann in enumerate(anns)}
    coco_gt = {str(idx):[{'caption':ann["gt_caption"]}] for idx, ann in enumerate(anns)}
    # create coco_eval object by taking coco and coco_result
    coco_eval = RefEvalCap(coco_gt, coco_result)

    # coco_eval.evaluate()
    try:
        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.evaluate()
    except Exception as exp:
        print(exp)
        return {}

    # print output evaluation scores
    return coco_eval.eval

class RefEvalCap:
    def __init__(self, gts, res):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        # ann[id]:caption
        self.gts=gts
        self.res=res

    def evaluate(self):

        gts=self.gts
        res=self.res

        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]


# ss=ref_caption_eval("OUTPUT/debug/eval_refcoco_val_generate_referring/ckpt-0-None/val_caption_pred.json")
# print(ss)