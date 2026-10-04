"""
subject_split.py — 
========================================================================
أداة تقسيم واحدة مشتركة تُستخدم في موضعين مختلفين من التدقيق:

  ITEM 5 (within-subject baseline): يجب تقسيم تكرارات (repetitions) كل
    شخص كاملة إلى train/test *قبل* التنويف (windowing). التقسيم على
    مستوى النافذة (كما كان سابقاً) يُسرّب عينات خام مشتركة بين train
    وtest لأن النوافذ المتجاورة تتداخل 50%.

  ITEM 4 (transductive UDA): يجب تقسيم تكرارات الشخص المستهدَف (target)
    في كل LOSO fold إلى:
      - "calibration" (بدون تسميات تُستخدم في التقييم): تُغذّي فقط
        عملية مواءمة CORAL/TCA/SA (حساب Ct، الفراغ الفرعي، إلخ).
      - "final-test": لا تُستخدم إطلاقاً في مواءمة أي طريقة — تُستخدم
        فقط لقياس الدقة النهائية.

كلا الحالتين هما نفس المسألة رياضياً: "قسّم مجموعة التكرارات المتاحة
لكل صنف بنسبة معيّنة، بشكل طبقي (stratified) بحيث كل صنف يُمثَّل في
الطرفين إن أمكن". لذلك دالة واحدة `split_by_repetition` تخدم الاثنين.

يتطلب هذا وجود عمود `rep_id` لكل نافذة (أضيف إلى data_loader.py في
FIX-DATALOADER-01 — انظر التعليق هناك).
"""
import numpy as np


def split_by_repetition(y, rep_id, held_out_fraction, seed=42,
                        min_reps_per_class_each_side=1):
    """
    يقسم النوافذ إلى (mask_kept, mask_held_out) بحيث:
      - القرار يُتخذ على مستوى (class, repetition) كاملاً، وليس نافذة
        بنافذة -> لا نافذتان متجاورتان (متداخلتان في العينات الخام) يمكن
        أن تنتهيا في طرفين مختلفين.
      - طبقي: لكل صنف على حدة، تُقسَّم تكراراته المتاحة بنفس النسبة
        التقريبية قدر الإمكان.
      - يحاول ضمان `min_reps_per_class_each_side` تكرار واحد على الأقل
        لكل صنف في كل طرف، إن كان عدد التكرارات المتاحة يسمح؛ إن لم
        يسمح (صنف له تكرار واحد فقط)، يُترك ذلك الصنف بالكامل في الطرف
        "kept" ويُصدر تحذير عبر القيمة المُعادة الثالثة (warnings_list).

    المعاملات:
        y (np.ndarray[int]):       تسمية الصنف لكل نافذة.
        rep_id (np.ndarray[int]):  رقم التكرار (repetition) لكل نافذة،
                                    من نفس الشخص (لا معنى لمقارنته بين
                                    أشخاص مختلفين).
        held_out_fraction (float): نسبة التكرارات (0-1) التي تذهب إلى
                                    الطرف "held_out" لكل صنف.
        seed (int):                لضمان التكرار.
        min_reps_per_class_each_side (int): حد أدنى من التكرارات لكل
                                    طرف قبل اعتبار التقسيم صالحاً لذلك
                                    الصنف.

    المخرجات:
        mask_kept (np.ndarray[bool]), mask_held_out (np.ndarray[bool]),
        warnings_list (list[str])
    """
    y      = np.asarray(y)
    rep_id = np.asarray(rep_id)
    if y.shape[0] != rep_id.shape[0]:
        raise ValueError(
            f"y and rep_id must have the same length "
            f"(got {y.shape[0]} vs {rep_id.shape[0]})"
        )
    if not (0.0 < held_out_fraction < 1.0):
        raise ValueError("held_out_fraction must be in (0, 1)")

    rng = np.random.RandomState(seed)
    mask_kept      = np.zeros(y.shape[0], dtype=bool)
    mask_held_out  = np.zeros(y.shape[0], dtype=bool)
    warnings_list  = []

    for cls in np.unique(y):
        cls_mask   = (y == cls)
        cls_reps   = np.unique(rep_id[cls_mask])
        n_reps     = len(cls_reps)

        if n_reps < 2 * min_reps_per_class_each_side:
            # لا يوجد ما يكفي من التكرارات لتقسيم هذا الصنف بأمان ->
            # يبقى بالكامل في "kept" (السلوك الأكثر تحفظاً؛ لا يُسرَّب
            # شيء إلى held_out من هذا الصنف).
            mask_kept |= cls_mask
            warnings_list.append(
                f"class {cls}: only {n_reps} repetition(s) available; "
                f"kept entirely on the 'kept' side (no split possible "
                f"with min_reps_per_class_each_side="
                f"{min_reps_per_class_each_side})."
            )
            continue

        shuffled = cls_reps.copy()
        rng.shuffle(shuffled)
        n_held = max(min_reps_per_class_each_side,
                    int(round(n_reps * held_out_fraction)))
        n_held = min(n_held, n_reps - min_reps_per_class_each_side)

        held_reps = set(shuffled[:n_held].tolist())
        kept_reps = set(shuffled[n_held:].tolist())

        for r in held_reps:
            mask_held_out |= (cls_mask & (rep_id == r))
        for r in kept_reps:
            mask_kept |= (cls_mask & (rep_id == r))

    return mask_kept, mask_held_out, warnings_list


def summarize_split(y, rep_id, mask_kept, mask_held_out):
    """ملخص نصي مختصر للتحقق اليدوي السريع (عدد النوافذ/التكرارات لكل طرف)."""
    n_total = len(y)
    n_kept, n_held = int(mask_kept.sum()), int(mask_held_out.sum())
    n_dropped = n_total - n_kept - n_held
    reps_kept = len(np.unique(rep_id[mask_kept])) if n_kept else 0
    reps_held = len(np.unique(rep_id[mask_held_out])) if n_held else 0
    overlap = np.any(mask_kept & mask_held_out)
    return {
        "n_total_windows": n_total,
        "n_kept_windows": n_kept,
        "n_held_out_windows": n_held,
        "n_dropped_windows": n_dropped,
        "n_kept_repetitions": reps_kept,
        "n_held_out_repetitions": reps_held,
        "classes_kept": sorted(set(np.unique(y[mask_kept]).tolist())),
        "classes_held_out": sorted(set(np.unique(y[mask_held_out]).tolist())),
        "mask_overlap_detected": bool(overlap),   # must always be False
    }
