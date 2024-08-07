from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import os
import json


def coco_caption_eval(
    annotation_file,
    results_file,
    use_1st_sentence_only=False,
):

    # we use the test dataset as the evaluation
    annotation_file = annotation_file.replace(
        ".json", "_gt.json"
    )
    # create coco object and coco_result object
    coco = COCO(annotation_file)

    with open(results_file) as f:
        anns = json.load(f)
    if use_1st_sentence_only:
        for ann in anns:
            ann["caption"] = ann["caption"].split(".")[0]
    coco_result = coco.loadRes(anns)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params["image_id"] = coco_result.getImgIds()

    try:
        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.evaluate()
    except Exception as exp:
        print(exp)
        return {}

    # print output evaluation scores
    return coco_eval.eval


# if __name__=="__main__":
#     re=coco_caption_eval("/home/luorun/datasets/coco/annotations/coco_karpathy_test.json",
#                          "/home/luorun/workspace/MM-Interleaved/OUTPUT/test/eval_coco_karpathy_generate_texts/ckpt-1500-23.80952380952381/val_caption_pred.json")
#     print(re)