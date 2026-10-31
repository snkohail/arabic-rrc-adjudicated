"""Generate the MOCK annotation folder used by the examples.

Everything here is invented: four short synthetic "judgments" written for this demo, with fictitious
case numbers and no connection to any real court file or to the confidential corpus.  The output has
the same file layout the code expects (annotator_A/, annotator_B/, Annotator_C/, one <id>.json each)
and contains, in every judgment, one instance of each of the four structural forms:

  form 1  single-role region            (one boundary, one role)
  form 2  co-extensive multi-role region (one boundary, two roles)
  form 3  nested structure               (an outer region containing an inner region with another role)
  form 4  ambiguous projection sentence  (a sentence on which two roles tie in character coverage)

Four judgments x four forms = 16 showcase items, listed in mock_items.json.

Usage: python examples/mock/make_mock_data.py [output_dir]      (default: examples/mock/annotation)
"""
import hashlib
import json
import sys
from pathlib import Path

# Sentence slots (same layout in every mock judgment):
#  0 PREAMBLE | 1 FACTS | 2 FACTS+ARGUMENT_PLAINTIFF (co-extensive) | 3 ISSUE
#  4-5 ARGUMENT_DEFENDANT region, 5-6 ANALYSIS region  -> sentence 5 is covered by both (tie, crossing)
#  7-9 outer ANALYSIS region with inner LAW_REFERENCE on sentence 8 (nested)
#  10 DECISION
PARAGRAPHS = [[0], [1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

JUDGMENTS = {
    "M0001": [
        "محكمة المدينة الابتدائية، الدائرة الجزائية الأولى، الجلسة المنعقدة علناً بتاريخ 2024/03/11 برئاسة القاضي فلان وعضوية القاضيين فلان وفلان، في الدعوى رقم 1001 لسنة 2024 جزاء.",
        "تتحصل الوقائع في أن المتهم قاد مركبته في الطريق العام ليلاً وتجاوز الإشارة الضوئية الحمراء فاصطدم بمركبة أخرى كانت تعبر التقاطع.",
        "وأضافت النيابة العامة أن المتهم غادر موقع الحادث دون إبلاغ الشرطة، وطلبت توقيع أقصى العقوبة المقررة قانوناً.",
        "وأحالته النيابة إلى هذه المحكمة بتهمة القيادة بتهور وتعريض حياة الآخرين للخطر.",
        "وبجلسة المحاكمة حضر المتهم وأنكر ما نسب إليه قائلاً إن الإشارة كانت صفراء عند عبوره وإنه توقف لاحقاً عند أقرب نقطة آمنة.",
        "وطلب دفاعه البراءة تأسيساً على أن تقرير الفحص الفني لم يثبت سرعة زائدة، وهو ما ترى المحكمة أنه لا ينفي بذاته تجاوز الإشارة.",
        "وحيث إن المحكمة اطمأنت إلى أقوال شاهد الواقعة وإلى تسجيل الكاميرا المرورية فإنها تطرح إنكار المتهم جانباً.",
        "وحيث إن الثابت من الأوراق أن المتهم عبر التقاطع والإشارة حمراء فإن ركن الخطأ متوافر في حقه.",
        "وتنص المادة 56 من قانون السير والمرور الافتراضي على أن تجاوز الإشارة الضوئية الحمراء يعاقب عليه بالغرامة التي لا تقل عن ألف درهم.",
        "ومن ثم تنتهي المحكمة إلى ثبوت التهمة في حق المتهم على النحو الوارد بأمر الإحالة.",
        "فلهذه الأسباب حكمت المحكمة حضورياً بتغريم المتهم ثلاثة آلاف درهم وإلزامه بالرسوم.",
    ],
    "M0002": [
        "محكمة المدينة الابتدائية، الدائرة المدنية الثانية، الجلسة المنعقدة علناً بتاريخ 2024/05/02، في الدعوى رقم 2002 لسنة 2024 مدني.",
        "تخلص الوقائع في أن المدعي أجر للمدعى عليه شقة سكنية بموجب عقد إيجار سنوي وأن المدعى عليه توقف عن سداد الأجرة اعتباراً من الشهر الرابع.",
        "ويقول المدعي إنه أنذر المدعى عليه كتابياً مرتين دون جدوى، ويطلب إلزامه بالأجرة المتأخرة وإخلاء العين المؤجرة.",
        "وتتحدد المسألة المطروحة فيما إذا كان المدعى عليه قد أخل بالتزامه بسداد الأجرة بما يبرر الإخلاء.",
        "ودفع المدعى عليه بأن المدعي لم يقم بإصلاح أعطال التكييف رغم مطالبته المتكررة وأنه احتبس الأجرة لهذا السبب.",
        "وأضاف أن احتباس الأجرة كان مؤقتاً إلى حين الإصلاح، وهو دفع ترى المحكمة أنه لا يستند إلى شرط في العقد يجيز ذلك.",
        "وحيث إن العقد لم يتضمن ما يخول المستأجر الامتناع عن السداد فإن الدفع يكون في غير محله.",
        "وحيث إن الثابت من المستندات أن المدعى عليه توقف عن السداد لمدة تجاوزت الثلاثة أشهر.",
        "وتنص المادة 12 من قانون الإيجارات الافتراضي على أن تأخر المستأجر في سداد الأجرة مدة تزيد على ثلاثة أشهر يجيز للمؤجر طلب الإخلاء.",
        "ومن ثم تقضي المحكمة بإجابة المدعي إلى طلبيه معاً.",
        "فلهذه الأسباب حكمت المحكمة بإلزام المدعى عليه بأداء الأجرة المتأخرة وبإخلاء العين المؤجرة وتسليمها خالية.",
    ],
    "M0003": [
        "محكمة المدينة الابتدائية، الدائرة التجارية، الجلسة المنعقدة علناً بتاريخ 2024/06/20، في الدعوى رقم 3003 لسنة 2024 تجاري.",
        "تتلخص الوقائع في أن الشركة المدعية وردت بضائع إلى الشركة المدعى عليها بموجب فواتير معتمدة وأن قيمة الفواتير لم تسدد في موعدها.",
        "وتضيف المدعية أنها سلمت البضائع كاملة وفق المواصفات المتفق عليها، وتطلب الحكم لها بالقيمة والفائدة القانونية.",
        "والمسألة محل النزاع هي ما إذا كان الدين ثابتاً ومستحق الأداء في ذمة المدعى عليها.",
        "ودفعت المدعى عليها بأن جزءاً من البضائع وصل تالفاً وأنها أخطرت المدعية بذلك عبر البريد الإلكتروني.",
        "وطلبت رفض الدعوى أو خصم قيمة التالف، وهو ما ترى المحكمة أنه لم يقدم عليه دليل كاف سوى مراسلة غير مؤرخة.",
        "وحيث إن المدعى عليها لم تقدم تقرير معاينة ولم تطلب ندب خبير فإن دفعها يبقى مرسلاً.",
        "وحيث إن الفواتير موقعة بالاستلام ولم ينازع في صحة التوقيع عليها.",
        "وتنص المادة 88 من قانون المعاملات التجارية الافتراضي على أن الفاتورة الموقعة بالاستلام تعد قرينة على ثبوت الدين ما لم يثبت العكس.",
        "ومن ثم يكون الدين ثابتاً في ذمة المدعى عليها وتقضي المحكمة به.",
        "فلهذه الأسباب حكمت المحكمة بإلزام المدعى عليها بأن تؤدي للمدعية قيمة الفواتير مع الفائدة القانونية من تاريخ المطالبة.",
    ],
    "M0004": [
        "محكمة المدينة الابتدائية، الدائرة العمالية، الجلسة المنعقدة علناً بتاريخ 2024/09/15، في الدعوى رقم 4004 لسنة 2024 عمالي.",
        "تتحصل الوقائع في أن العامل التحق بالعمل لدى المنشأة المدعى عليها بوظيفة فني وأن علاقة العمل انتهت بإخطار من صاحب العمل.",
        "ويقول العامل إنه لم يتسلم مكافأة نهاية الخدمة ولا بدل الإجازات المستحقة، ويطلب إلزام المنشأة بأدائها.",
        "وتنحصر المسألة فيما إذا كان العامل مستحقاً للمكافأة والبدلات المطالب بها.",
        "ودفعت المنشأة بأن العامل استقال من تلقاء نفسه قبل إكمال سنة واحدة وأنه لا يستحق المكافأة.",
        "وقدمت صورة من خطاب استقالة غير موقع، وهو مستند ترى المحكمة أنه لا يصلح دليلاً على الاستقالة.",
        "وحيث إن عبء إثبات الاستقالة يقع على صاحب العمل ولم يقم به فإن إنهاء العلاقة يعد صادراً عنه.",
        "وحيث إن مدة الخدمة الثابتة بالأوراق تجاوزت السنة.",
        "وتنص المادة 30 من قانون العمل الافتراضي على أن العامل الذي أكمل سنة في الخدمة يستحق مكافأة نهاية الخدمة عن مدة خدمته.",
        "ومن ثم يستحق العامل المكافأة وبدل الإجازات المطالب بهما.",
        "فلهذه الأسباب حكمت المحكمة بإلزام المنشأة بأن تؤدي للعامل مكافأة نهاية الخدمة وبدل الإجازات المستحقة.",
    ],
}

# which single-role sentence is the form-1 showcase in each judgment (role varies)
FORM1_SHOWCASE = {"M0001": 0, "M0002": 10, "M0003": 3, "M0004": 1}
SINGLE_LABEL = {0: "PREAMBLE", 1: "FACTS", 3: "ISSUE", 10: "DECISION"}


def build_text(sentences):
    """Join sentences into paragraphs; return text and [begin, end) offsets of every sentence."""
    text, offsets = "", []
    for pi, para in enumerate(PARAGRAPHS):
        if pi:
            text += "\n\n"
        for k, si in enumerate(para):
            if k:
                text += " "
            b = len(text); text += sentences[si]; offsets.append((b, len(text)))
    return text, offsets


def span(b, e, label):
    return {"begin": b, "end": e, "label": label, "kind": "rhetorical"}


def c_spans(o):
    return [
        span(*o[0], "PREAMBLE"),
        span(*o[1], "FACTS"),
        span(*o[2], "FACTS"), span(*o[2], "ARGUMENT_PLAINTIFF"),          # form 2: same boundary, two roles
        span(*o[3], "ISSUE"),
        span(o[4][0], o[5][1], "ARGUMENT_DEFENDANT"),                      # 4-5
        span(o[5][0], o[6][1], "ANALYSIS"),                                # 5-6  -> crossing; sentence 5 ties
        span(o[7][0], o[9][1], "ANALYSIS"),                                # outer 7-9
        span(*o[8], "LAW_REFERENCE"),                                      # inner 8   -> form 3
        span(*o[10], "DECISION"),
    ]


def a_spans(o):
    """Annotator A (synthetic variant): no inner LAW_REFERENCE, no crossing."""
    return [
        span(*o[0], "PREAMBLE"), span(*o[1], "FACTS"), span(*o[2], "FACTS"), span(*o[2], "ARGUMENT_PLAINTIFF"),
        span(*o[3], "ISSUE"), span(o[4][0], o[5][1], "ARGUMENT_DEFENDANT"), span(*o[6], "ANALYSIS"),
        span(o[7][0], o[9][1], "ANALYSIS"), span(*o[10], "DECISION"),
    ]


def b_spans(o):
    """Annotator B (synthetic variant): one role only on the multi-role sentence."""
    return [s for s in c_spans(o) if not (s["begin"] == o[2][0] and s["label"] == "FACTS")]


def write(out: Path):
    items = []
    for ann in ("annotator_A", "annotator_B", "Annotator_C"):
        (out / ann).mkdir(parents=True, exist_ok=True)
    for cid, sents in JUDGMENTS.items():
        text, o = build_text(sents)
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        base = {"canonical_case_id": cid, "text_len": len(text), "text_sha256": sha,
                "source": "SYNTHETIC MOCK DATA generated by examples/mock/make_mock_data.py; not a real judgment"}
        (out / "Annotator_C" / f"{cid}.json").write_text(json.dumps({**base, "text": text, "spans": c_spans(o)}, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "annotator_A" / f"{cid}.json").write_text(json.dumps({**base, "text": text, "spans": a_spans(o)}, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / "annotator_B" / f"{cid}.json").write_text(json.dumps({**base, "spans": b_spans(o)}, ensure_ascii=False, indent=1), encoding="utf-8")   # blind: no text
        f1 = FORM1_SHOWCASE[cid]
        items += [
            {"form": 1, "name": "single_role_region", "case_id": cid, "begin": o[f1][0], "end": o[f1][1], "labels": [SINGLE_LABEL[f1]]},
            {"form": 2, "name": "multi_role_region", "case_id": cid, "begin": o[2][0], "end": o[2][1], "labels": ["ARGUMENT_PLAINTIFF", "FACTS"]},
            {"form": 3, "name": "nested_structure", "case_id": cid, "outer": [o[7][0], o[9][1]], "inner": list(o[8]), "outer_labels": ["ANALYSIS"], "inner_labels": ["LAW_REFERENCE"]},
            {"form": 4, "name": "ambiguous_projection_sentence", "case_id": cid, "begin": o[5][0], "end": o[5][1], "sent_idx": 5, "tied_roles": ["ANALYSIS", "ARGUMENT_DEFENDANT"]},
        ]
    (out.parent / "mock_items.json").write_text(json.dumps({"note": "16 synthetic showcase items: 4 judgments x 4 structural forms", "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    return items


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "annotation"
    items = write(out)
    print(f"wrote {len(JUDGMENTS)} synthetic judgments x 3 annotator folders to {out}; {len(items)} showcase items in {out.parent / 'mock_items.json'}")
