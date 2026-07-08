"""E6: causal test of the covert language register.

E4b found that under the mouth-exclusion criterion the only robustly covert
lens content is *metadata* (which language the context is in, typo targets),
not content plans. This experiment asks whether that language register is
causally load-bearing: swapping its direction mid-band should flip the
output language while preserving the answer's semantics.

Design, and how it differs from SAE language steering (Chou et al. 2025,
arXiv:2507.13410):

- The register direction is estimated from *content-matched parallel
  sentences* (English/Chinese), as the mass-mean contrast of band-layer
  residuals averaged over all positions. It is not an SAE feature and is
  not defined by output-language differences, so finding it does not
  presuppose the register is visible at the output.
- Intervention is a translation along the contrast axis by the measured
  population gap: h' = h + alpha * (c_tgt - c_src) * d, where c_src/c_tgt
  are the mean coordinates of the two languages' estimation sentences on
  the axis. This is the population-level form of the paper's "clamping a
  lens coordinate": v1 used the per-position reflection h - 2c*d and the
  ambient coordinate on this axis turned out to be huge (mean |c| ~2453 at
  1.7B vs ~66 on a random direction — the axis overlaps high-norm shared
  components), so reflection destroys the state (0% preserved). The gap
  translation moves the register by exactly the between-language distance
  and nothing else.
- Every item carries a *covert flag*: whether the target-language answer
  form is absent from the clean next-token top-100 (E4b's strict
  criterion). The causal claim is evaluated separately on covert items.

Arms per item and direction (zh->en on Chinese prompts, en->zh on English):
  baseline    no intervention
  register    gap translation along the language-contrast axis
  random      translation of the same per-layer norm along a random unit
              direction (seed-fixed) — the amplitude-matched control

Metrics: output language (CJK vs Latin character count), answer correctness
in either language, full flip (language flipped AND answer correct in the
target language), semantic preservation (correct in either language).
Register vs random compared by exact McNemar.

Dosing note: the gap measured at layer l already contains everything that
propagated from earlier layers, so adding the full gap at every band layer
compounds ~len(band)-fold. alpha=1 flips the language 100% but erases the
question (generic target-language text); the sweep looks for an operating
point where the flip survives with content intact.

Run:  HF_HUB_DISABLE_XET=1 .venv/bin/python experiments/e6-covert-register/run_e6.py [1.7b|4b] [alpha ...]
"""

from __future__ import annotations

import json
import os
import pathlib
import random
import re
import sys

import torch
import transformers

import jlens

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.interventions import DirectionShiftHooks  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e1-flexible-generalization"))
from run_e1 import BAND_START_FRAC, MODELS, family, greedy_gen, text_match  # noqa: E402

sys.path.insert(0, str(ROOT / "experiments" / "e2-probe-swap"))
from run_e2 import mcnemar_exact_p  # noqa: E402

COVERT_RANK = 100  # E4b strict mouth-exclusion threshold

# Content-matched parallel sentences for register estimation. Deliberately
# disjoint in topic from the test items so the contrast axis cannot encode
# their answers.
REGISTER_PAIRS = [
    ("The weather is very nice today", "今天的天气非常好"),
    ("I went to the library yesterday afternoon", "我昨天下午去了图书馆"),
    ("She likes to drink tea in the morning", "她喜欢在早上喝茶"),
    ("The train arrives at seven o'clock", "火车七点钟到站"),
    ("My younger brother is learning to swim", "我弟弟正在学游泳"),
    ("The food at this restaurant is delicious", "这家餐厅的菜很好吃"),
    ("He reads the newspaper every day", "他每天都看报纸"),
    ("The meeting has been moved to next week", "会议改到了下周"),
    ("We watched a movie last night", "我们昨晚看了一部电影"),
    ("The store closes at nine in the evening", "商店晚上九点关门"),
    ("Autumn is my favorite season", "秋天是我最喜欢的季节"),
    ("The children are playing in the park", "孩子们正在公园里玩"),
    ("Please close the window before you leave", "离开前请把窗户关上"),
    ("I forgot to bring my umbrella this morning", "我今天早上忘了带伞"),
    ("The new phone is much faster than the old one", "新手机比旧的快多了"),
    ("Learning a language takes a lot of practice", "学一门语言需要大量练习"),
]

# (zh_prompt, en_prompt, en_answer, zh_answer)
ITEMS = [
    ("事实：法国的首都是", "Fact: The capital of France is", "Paris", "巴黎"),
    ("事实：日本的首都是", "Fact: The capital of Japan is", "Tokyo", "东京"),
    ("事实：俄罗斯的首都是", "Fact: The capital of Russia is", "Moscow", "莫斯科"),
    ("事实：德国的首都是", "Fact: The capital of Germany is", "Berlin", "柏林"),
    ("事实：意大利的首都是", "Fact: The capital of Italy is", "Rome", "罗马"),
    ("事实：西班牙的首都是", "Fact: The capital of Spain is", "Madrid", "马德里"),
    ("事实：英国的首都是", "Fact: The capital of England is", "London", "伦敦"),
    ("事实：中国的首都是", "Fact: The capital of China is", "Beijing", "北京"),
    ("事实：埃及的首都是", "Fact: The capital of Egypt is", "Cairo", "开罗"),
    ("事实：希腊的首都是", "Fact: The capital of Greece is", "Athens", "雅典"),
    ("事实：韩国的首都是", "Fact: The capital of South Korea is", "Seoul", "首尔"),
    ("事实：泰国的首都是", "Fact: The capital of Thailand is", "Bangkok", "曼谷"),
    ("在沙漠里靠驼峰储存脂肪的动物是", "The desert animal that stores fat in its hump is the", "camel", "骆驼"),
    ("中国特有的黑白相间的熊是", "The black and white bear native to China is the", "panda", "熊猫"),
    ("陆地上最大的动物是", "The largest land animal is the", "elephant", "大象"),
    ("被称为丛林之王的动物是", "The animal known as the king of the jungle is the", "lion", "狮子"),
    ("脖子最长的动物是", "The animal with the longest neck is the", "giraffe", "长颈鹿"),
    ("猴子最爱吃的黄色水果是", "The long yellow fruit monkeys love is the", "banana", "香蕉"),
    ("被称为红色星球的行星是", "The planet known as the red planet is", "Mars", "火星"),
    ("水结冰后叫做", "Frozen water is called", "ice", "冰"),
    ("用烘焙过的豆子冲泡的热饮是", "The hot drink brewed from roasted beans is", "coffee", "咖啡"),
    ("事实：法国人说的语言是", "Fact: The language spoken in France is", "French", "法语"),
    ("事实：地球上最大的海洋是", "Fact: The largest ocean on Earth is the", "Pacific", "太平洋"),
    ("事实：地球上最高的山峰是", "Fact: The tallest mountain on Earth is Mount", "Everest", "珠穆朗玛"),
    # --- Expansion (2026-07-07): 36 additional pairs biased toward common
    # nouns and second-tier entities. Rationale: in the first 24-pair run the
    # covert-verified subset was n=7, and every covert item was a common noun
    # or a non-flagship entity, while flagship capitals (Paris, Tokyo) always
    # co-activated their translations inside the mouth top-100. Topics avoid
    # every content word used in REGISTER_PAIRS.
    ("只在澳大利亚生活、靠跳跃前进的有袋动物是", "The hopping marsupial that lives only in Australia is the", "kangaroo", "袋鼠"),
    ("在北极冰面上捕猎海豹的白色大熊是", "The big white bear that hunts seals on Arctic ice is the", "polar bear", "北极熊"),
    ("会模仿人类说话的彩色鸟是", "The colorful bird that can mimic human speech is the", "parrot", "鹦鹉"),
    ("有八条腿、会织网捕虫的动物是", "The eight-legged animal that spins webs to catch insects is the", "spider", "蜘蛛"),
    ("倒挂在山洞里睡觉、会飞的哺乳动物是", "The flying mammal that sleeps hanging upside down in caves is the", "bat", "蝙蝠"),
    ("全身长满尖刺的小型哺乳动物是", "The small mammal covered in sharp spines is the", "hedgehog", "刺猬"),
    ("生活在南极、不会飞的黑白色鸟是", "The flightless black and white bird that lives in Antarctica is the", "penguin", "企鹅"),
    ("世界上最大的不会飞的鸟是", "The largest flightless bird in the world is the", "ostrich", "鸵鸟"),
    ("海洋里最聪明、用回声定位的哺乳动物是", "The intelligent sea mammal that uses echolocation is the", "dolphin", "海豚"),
    ("秋天埋藏坚果准备过冬的小动物是", "The small animal that buries nuts for the winter is the", "squirrel", "松鼠"),
    ("能改变皮肤颜色来伪装的爬行动物是", "The reptile that changes its skin color to camouflage is the", "chameleon", "变色龙"),
    ("背着壳慢慢爬行的软体动物是", "The soft slow animal that crawls with its shell on its back is the", "snail", "蜗牛"),
    ("由牛奶发酵制成的酸味乳制品是", "The sour dairy product made from fermented milk is", "yogurt", "酸奶"),
    ("蜜蜂采花蜜酿成的甜食是", "The sweet food that bees make from nectar is", "honey", "蜂蜜"),
    ("用葡萄发酵酿成的酒是", "The alcoholic drink fermented from grapes is", "wine", "葡萄酒"),
    ("上面铺奶酪和番茄的意大利圆形烤饼是", "The round Italian flatbread topped with cheese and tomato is", "pizza", "披萨"),
    ("用小麦粉做成的细长条状主食是", "The long thin staple food made from wheat flour is", "noodles", "面条"),
    ("由牛奶制成的黄色固体发酵乳制品是", "The solid yellow dairy product made from fermented milk is", "cheese", "奶酪"),
    ("冬天从天上落下的白色冰晶是", "The white ice crystals that fall from the sky in winter are called", "snow", "雪"),
    ("雨后天空中出现的七色圆弧是", "The seven-colored arc that appears in the sky after rain is a", "rainbow", "彩虹"),
    ("夜晚绕着地球转、照亮夜空的天体是", "The object that orbits the Earth and lights up the night sky is the", "moon", "月亮"),
    ("人类用来思考的器官是", "The organ humans use for thinking is the", "brain", "大脑"),
    ("在胸腔里把血液泵向全身的器官是", "The organ that pumps blood around the body is the", "heart", "心脏"),
    ("人体内用来呼吸空气的器官是", "The organs in the chest used for breathing are the", "lungs", "肺"),
    ("蚕吐丝织成的名贵天然织物是", "The precious natural fabric woven from silkworm threads is", "silk", "丝绸"),
    ("从绵羊身上剪下来纺线织衣的纤维是", "The fiber sheared from sheep to make clothing is", "wool", "羊毛"),
    ("在法庭上主持审判的人是", "The person who presides over trials in court is a", "judge", "法官"),
    ("驾驶飞机的人是", "The person who flies an airplane is a", "pilot", "飞行员"),
    ("在学校给学生上课的人是", "The person who teaches students at school is a", "teacher", "老师"),
    ("在医院里协助医生照顾病人的人是", "The person who assists doctors and cares for patients in a hospital is a", "nurse", "护士"),
    ("陈列古代文物和艺术品供人参观的建筑是", "The building that displays ancient artifacts and art for visitors is a", "museum", "博物馆"),
    ("飞机起飞和降落的场所是", "The place where airplanes take off and land is an", "airport", "机场"),
    ("医生给病人看病治疗的场所是", "The place where doctors treat patients is a", "hospital", "医院"),
    ("流经埃及的著名大河是", "The famous great river that flows through Egypt is the", "Nile", "尼罗河"),
    ("非洲北部最大的沙漠是", "The largest desert in northern Africa is the", "Sahara", "撒哈拉"),
    ("事实：印度的首都是", "Fact: The capital of India is", "New Delhi", "新德里"),
]

CJK = re.compile(r"[一-鿿]")
LATIN = re.compile(r"[A-Za-z]")


def lang_of(text: str) -> str:
    cjk, latin = len(CJK.findall(text)), len(LATIN.findall(text))
    if cjk == 0 and latin == 0:
        return "none"
    return "zh" if cjk > latin / 2 else "en"


@torch.no_grad()
def register_axis(hf, tok, band: list[int], device, pairs=None):
    """Per band layer: unit contrast axis d = normalize(mean_zh - mean_en)
    and the population gap g = mean_zh·d - mean_en·d along it. Residuals
    averaged over token positions 1: (position 0 is the attention-sink
    outlier and is excluded).

    ``pairs`` defaults to the full REGISTER_PAIRS (the canonical axis); the
    multi-seed hardening passes a bootstrap resample of the pairs so we can
    report the flip rates' sensitivity to *which* parallel sentences estimate
    the axis, not just to item sampling."""
    pairs = pairs if pairs is not None else REGISTER_PAIRS
    means = {"en": None, "zh": None}
    for en, zh in pairs:
        for key, sent in (("en", en), ("zh", zh)):
            ids = tok(sent, return_tensors="pt").to(device)
            hs = hf(**ids, output_hidden_states=True).hidden_states
            act = torch.stack([hs[l + 1][0, 1:].float().mean(0) for l in band])
            means[key] = act if means[key] is None else means[key] + act
    mu_en = means["en"] / len(pairs)
    mu_zh = means["zh"] / len(pairs)
    axis, gap = {}, {}
    for j, l in enumerate(band):
        diff = mu_zh[j] - mu_en[j]
        axis[l] = diff / diff.norm()
        gap[l] = float(diff.norm())  # (mu_zh - mu_en)·d == |diff|
    return axis, gap


@torch.no_grad()
def mouth_rank(hf, tok, prompt: str, word_ids: list[int], device) -> int:
    """Rank of a word's first token in the clean next-token distribution."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    v = hf(ids).logits[0, -1].float()
    return int((v > v[word_ids[0]]).sum())


def main(model_key: str = "1.7b", alphas: list[float] | None = None) -> None:
    alphas = alphas or [1.0]
    model_id, _, _ = MODELS[model_key]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    model = jlens.from_hf(hf, tok)
    band = list(range(round(BAND_START_FRAC * model.n_layers), model.n_layers - 1))
    print(f"model={model_id} band={band[0]}..{band[-1]} alphas={alphas}")

    # multi-seed hardening (env-driven so the canonical positional interface
    # is untouched): E6_SEED reseeds the amplitude-matched random control and,
    # when E6_RESAMPLE=1, a bootstrap resample of the estimation pairs.
    seed = int(os.environ.get("E6_SEED", "0"))
    resample = os.environ.get("E6_RESAMPLE", "0") == "1"
    if resample:
        rnd_pairs = random.Random(seed)
        pairs = rnd_pairs.choices(REGISTER_PAIRS, k=len(REGISTER_PAIRS))
        print(f"[multi-seed] E6_SEED={seed} resampled estimation pairs (bootstrap)")
    else:
        pairs = REGISTER_PAIRS
        if seed != 0:
            print(f"[multi-seed] E6_SEED={seed} (random control only; axis = full set)")

    axis, gap = register_axis(hf, tok, band, device, pairs=pairs)
    gen = torch.Generator().manual_seed(seed)
    d_model = axis[band[0]].numel()
    rand = {}
    for l in band:
        r = torch.randn(d_model, generator=gen)
        rand[l] = (r / r.norm()).to(device)
    gaps = " ".join(f"L{l}:{gap[l]:.1f}" for l in band[::4])
    print(f"register axis from {len(REGISTER_PAIRS)} parallel pairs; gaps {gaps}")

    records = []
    for zh_prompt, en_prompt, en_ans, zh_ans in ITEMS:
        en_ids = tok.encode(" " + en_ans, add_special_tokens=False)
        zh_ids = tok.encode(zh_ans, add_special_tokens=False)
        for src, prompt, tgt_ids in (("zh", zh_prompt, en_ids), ("en", en_prompt, zh_ids)):
            tgt = "en" if src == "zh" else "zh"
            covert = mouth_rank(hf, tok, prompt, tgt_ids, device) >= COVERT_RANK

            # zh->en moves against the (zh - en) axis; en->zh moves along it
            sgn = -1.0 if src == "zh" else 1.0

            def grade(text: str) -> dict:
                hit_en = text_match(text, en_ans)
                hit_zh = zh_ans in text
                lang = lang_of(text)
                return {"text": text, "lang": lang,
                        "hit_en": hit_en, "hit_zh": hit_zh,
                        "flip_lang": lang == tgt,
                        "flip_full": lang == tgt and (hit_en if tgt == "en" else hit_zh),
                        "preserved": hit_en or hit_zh}

            rec = {"src": src, "prompt": prompt, "en": en_ans, "zh": zh_ans,
                   "covert": covert, "arms": {}}
            _, base_text = greedy_gen(hf, tok, prompt, device, n_new=8)
            rec["arms"]["baseline"] = grade(base_text)
            for alpha in alphas:
                reg_shift = {l: sgn * alpha * gap[l] * axis[l] for l in band}
                rnd_shift = {l: alpha * gap[l] * rand[l] for l in band}
                for name, shifts in ((f"register@{alpha:g}", reg_shift),
                                     (f"random@{alpha:g}", rnd_shift)):
                    ctx = DirectionShiftHooks(model.layers, shifts)
                    _, text = greedy_gen(hf, tok, prompt, device, n_new=8, swap_ctx=ctx)
                    rec["arms"][name] = grade(text)
            rec["baseline_ok"] = (rec["arms"]["baseline"]["preserved"]
                                  and rec["arms"]["baseline"]["lang"] == src)
            records.append(rec)
            reg_mid = rec["arms"][f"register@{alphas[len(alphas) // 2]:g}"]["text"]
            print(f"[{src}->{tgt}] {en_ans:9s} covert={int(covert)} "
                  f"base={base_text[:12]!r} reg[mid-alpha]={reg_mid[:12]!r}")

    ok = [r for r in records if r["baseline_ok"]]

    def rate(rs, arm, key):
        return sum(r["arms"][arm][key] for r in rs) / len(rs) if rs else float("nan")

    print(f"\nbaseline ok (answers in prompt language): {len(ok)}/{len(records)}")
    summary = {"n_ok": len(ok), "per_alpha": {}}
    covert_ok = [r for r in ok if r["covert"]]
    for alpha in alphas:
        reg, rnd = f"register@{alpha:g}", f"random@{alpha:g}"
        only_reg = sum(r["arms"][reg]["flip_full"] and not r["arms"][rnd]["flip_full"] for r in ok)
        only_rnd = sum(r["arms"][rnd]["flip_full"] and not r["arms"][reg]["flip_full"] for r in ok)
        p = mcnemar_exact_p(only_reg, only_rnd)
        print(f"\nalpha={alpha:g} (n={len(ok)}, covert n={len(covert_ok)}):")
        for arm in (reg, rnd):
            print(f"  {arm:16s} flip_lang={rate(ok, arm, 'flip_lang'):6.1%} "
                  f"flip_full={rate(ok, arm, 'flip_full'):6.1%} "
                  f"preserved={rate(ok, arm, 'preserved'):6.1%}   "
                  f"[covert: flip_full={rate(covert_ok, arm, 'flip_full'):6.1%} "
                  f"preserved={rate(covert_ok, arm, 'preserved'):6.1%}]")
        print(f"  register vs random (flip_full): only-reg {only_reg}, only-rnd {only_rnd}, "
              f"exact McNemar p = {p:.4f}")
        summary["per_alpha"][f"{alpha:g}"] = {
            "register_flip_lang": rate(ok, reg, "flip_lang"),
            "register_flip_full": rate(ok, reg, "flip_full"),
            "register_preserved": rate(ok, reg, "preserved"),
            "random_flip_full": rate(ok, rnd, "flip_full"),
            "random_preserved": rate(ok, rnd, "preserved"),
            "covert_register_flip_full": rate(covert_ok, reg, "flip_full"),
            "p_register_vs_random": p,
        }

    tag = "" if (seed == 0 and not resample) else f"_seed{seed}{'r' if resample else ''}"
    out = ROOT / "results" / f"e6_covert_register_{family(model_id)}{model_key.replace('.', '')}{tag}.json"
    out.write_text(json.dumps({
        "model": model_id, "band": [band[0], band[-1]], "alphas": alphas,
        "seed": seed, "resample_pairs": resample,
        "n_register_pairs": len(REGISTER_PAIRS), "covert_rank": COVERT_RANK,
        "gap": {str(l): gap[l] for l in band},
        "records": records, "summary": summary,
    }, ensure_ascii=False, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "1.7b"
    alphas = [float(a) for a in sys.argv[2:]] or None
    main(key, alphas)
