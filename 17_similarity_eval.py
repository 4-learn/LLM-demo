"""第 17 節：相似度、失敗案例與門檻評測。

需要第 16 節的套件與模型；模型以外的函式（cosine、門檻挑選、指標）只用標準函式庫。
"""
import argparse
import math

MODEL_ID = "BAAI/bge-small-zh-v1.5"
REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："  # 模型卡原字串，見第 16 節

# (查詢, 段落, 人工標註 1=相關 0=不相關, 類型)。全部為合成句。
# 開發集：用來看分數、挑門檻。
DEV = [
    ("如何把路由器恢復出廠設定？", "長按重設鍵十秒，路由器會回到出廠狀態。", 1, "同義"),
    ("如何把路由器恢復出廠設定？", "路由器出廠時的預設密碼印在底部貼紙。", 0, "字面重疊"),
    ("印表機卡紙怎麼處理？", "打開後蓋，輕輕拉出夾住的紙張。", 1, "同義"),
    ("印表機卡紙怎麼處理？", "印表機紙匣可放兩百五十張紙。", 0, "字面重疊"),
    ("冷氣沒有漏水", "冷氣排水管漏水，地上有水漬。", 0, "否定"),
    ("工人有戴安全帽", "工人未戴安全帽進入工地。", 0, "否定"),
    ("工人有戴安全帽", "施工人員頭盔配戴完整。", 1, "同義"),
    ("AX-300 的保固多久？", "AX-300 保固兩年。", 1, "型號"),
    ("AX-300 的保固多久？", "AX-500 保固三年。", 0, "型號"),
    ("電池可以充電五百次", "電池可以充電五十次。", 0, "數字"),
    ("電池可以充電五百次", "電池循環壽命約五百次。", 1, "數字"),
    ("會議改到星期三下午", "會議延到週三午後舉行。", 1, "同義"),
    ("會議改到星期三下午", "會議改到星期四下午。", 0, "數字"),
    ("如何申請退貨？", "退貨請於七天內填寫線上表單。", 1, "同義"),
    ("如何申請退貨？", "本店不提供換貨服務。", 0, "近義干擾"),
    ("咖啡機一直漏水", "咖啡機底部持續滲水。", 1, "同義"),
    ("咖啡機一直漏水", "咖啡機水箱容量一點二公升。", 0, "字面重疊"),
    ("手機螢幕破了能修嗎", "螢幕碎裂可送修更換面板。", 1, "同義"),
    ("手機螢幕破了能修嗎", "手機螢幕亮度可在設定中調整。", 0, "字面重疊"),
    ("密碼忘記了", "點選「忘記密碼」並收取驗證信即可重設。", 1, "同義"),
]
# 保留集：在看任何分數之前就寫好；門檻挑完後只看一次，不再回頭改門檻。
HOLDOUT = [
    ("洗衣機脫水時很吵", "洗衣機脫水階段發出巨大噪音。", 1, "同義"),
    ("洗衣機脫水時很吵", "洗衣機脫水轉速最高一千二百轉。", 0, "字面重疊"),
    ("門鎖電池沒電怎麼開門", "電子鎖沒電時，可用九伏特電池接觸底部端子臨時供電。", 1, "同義"),
    ("門鎖電池沒電怎麼開門", "門鎖電池建議每年更換一次。", 0, "字面重疊"),
    ("熱水器沒有異味", "熱水器運作時有瓦斯味，請立即關閉。", 0, "否定"),
    ("訂單尚未出貨", "您的訂單已於今日出貨。", 0, "否定"),
    ("訂單尚未出貨", "訂單目前仍在備貨中，還沒寄出。", 1, "同義"),
    ("BX-12 支援快充嗎？", "BX-12 支援十八瓦快充。", 1, "型號"),
    ("BX-12 支援快充嗎？", "BX-21 支援十八瓦快充。", 0, "型號"),
    ("耳機續航八小時", "耳機續航十八小時。", 0, "數字"),
    ("耳機續航八小時", "充飽電可連續播放約八個鐘頭。", 1, "數字"),
    ("班機改到早上七點起飛", "班機改為上午七時出發。", 1, "同義"),
    ("班機改到早上七點起飛", "班機改到晚上七點起飛。", 0, "數字"),
    ("怎麼取消訂閱？", "在帳戶設定中點選「停止續訂」即可。", 1, "同義"),
    ("怎麼取消訂閱？", "訂閱方案可升級為年繳享九折。", 0, "近義干擾"),
    ("冰箱不冷", "冰箱冷藏室溫度異常升高，食物不夠冰。", 1, "同義"),
    ("冰箱不冷", "冰箱冷凍室容量為一百公升。", 0, "字面重疊"),
    ("筆電開不了機", "按電源鍵沒有任何反應時，先接上變壓器再試。", 1, "同義"),
    ("筆電開不了機", "筆電開機後會自動更新系統。", 0, "字面重疊"),
    ("帳號被盜了", "發現帳號遭他人登入，請立即變更密碼並登出所有裝置。", 1, "同義"),
]
# 正規化實驗用的小語料：同一題用 dot 與 cosine 排名。
CORPUS = [
    "打開後蓋，輕輕拉出夾住的紙張。",
    "印表機紙匣可放兩百五十張紙。",
    "路由器出廠時的預設密碼印在底部貼紙。",
    "請參閱使用手冊。",
    "電池可以充電五十次。",
]


def load_models():
    """回傳 (正規化模型, 未正規化模型)；後者共用同一權重，只少了最後的 Normalize 層。"""
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(2)
    model = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu",
                                trust_remote_code=False, model_kwargs={"use_safetensors": True})
    raw = SentenceTransformer(modules=[model[0], model[1]], device="cpu")
    return model, raw


def encode(model, texts, query=False):
    prefix = QUERY_PREFIX if query else ""
    vectors = model.encode([prefix + t for t in texts], batch_size=16, show_progress_bar=False)
    return [[float(x) for x in row] for row in vectors]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cosine(a, b):
    na, nb = math.sqrt(dot(a, a)), math.sqrt(dot(b, b))
    if na == 0 or nb == 0:
        raise ValueError("零向量沒有方向，cosine 無定義")
    return dot(a, b) / (na * nb)


def score_pairs(model, pairs):
    queries = encode(model, [p[0] for p in pairs], query=True)
    passages = encode(model, [p[1] for p in pairs])
    return [round(cosine(q, d), 4) for q, d in zip(queries, passages)]


def confusion(scores, labels, threshold):
    tp = sum(s >= threshold and y == 1 for s, y in zip(scores, labels))
    fp = sum(s >= threshold and y == 0 for s, y in zip(scores, labels))
    fn = sum(s < threshold and y == 1 for s, y in zip(scores, labels))
    tn = sum(s < threshold and y == 0 for s, y in zip(scores, labels))
    return tp, fp, fn, tn


def metrics(scores, labels, threshold):
    tp, fp, fn, tn = confusion(scores, labels, threshold)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round((tp + tn) / len(labels), 3),
            "precision": round(precision, 3), "recall": round(recall, 3)}


def pick_threshold(scores, labels):
    """只看傳進來的資料挑門檻：候選為各分數本身，取 accuracy 最高者；同分取較高門檻（較保守）。"""
    if not scores or len(scores) != len(labels):
        raise ValueError("分數與標註數量必須相同且不為空")
    best = None
    for t in sorted(set(scores)):
        m = metrics(scores, labels, t)
        if best is None or m["accuracy"] >= best["accuracy"]:
            best = m
    return best["threshold"]


def report(name, pairs, scores):
    print(f"[{name}]")
    for (q, d, y, kind), s in sorted(zip(pairs, scores), key=lambda x: -x[1]):
        print(f"{s:.4f} {'相關' if y else '不相關'} {kind:4} {q} ｜ {d}")


def worst_cases(pairs, scores):
    """最高分的不相關（高分錯配）與最低分的相關（低分漏配）。"""
    wrong = max((s, p) for p, s in zip(pairs, scores) if p[2] == 0)
    miss = min((s, p) for p, s in zip(pairs, scores) if p[2] == 1)
    return wrong, miss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", nargs=2, action="append", metavar=("QUERY", "PASSAGE"),
                        help="另外算一組句對的 cosine；可重複")
    args = parser.parse_args()
    model, raw = load_models()
    print("model:", MODEL_ID, "revision:", REVISION)

    print("\n== 一、正規化：同一題，dot 與 cosine 排名不同 ==")
    query = "手機螢幕破了能修嗎"
    q_raw, docs_raw = encode(raw, [query], query=True)[0], encode(raw, CORPUS)
    print("未正規化長度:", [round(math.sqrt(dot(v, v)), 2) for v in docs_raw])
    by_dot = sorted(range(len(CORPUS)), key=lambda i: -dot(q_raw, docs_raw[i]))
    by_cos = sorted(range(len(CORPUS)), key=lambda i: -cosine(q_raw, docs_raw[i]))
    print("dot   排名:", [CORPUS[i] for i in by_dot[:3]])
    print("cosine排名:", [CORPUS[i] for i in by_cos[:3]])
    q_norm, docs_norm = encode(model, [query], query=True)[0], encode(model, CORPUS)
    same = all(abs(dot(q_norm, d) - cosine(q_norm, d)) < 1e-5 for d in docs_norm)
    print("正規化後 dot == cosine:", same)

    print("\n== 二、開發集句對 ==")
    dev_scores = score_pairs(model, DEV)
    report("dev", DEV, dev_scores)
    wrong, miss = worst_cases(DEV, dev_scores)
    print(f"高分錯配: {wrong[0]:.4f} {wrong[1][0]} ｜ {wrong[1][1]}（{wrong[1][3]}）")
    print(f"低分漏配: {miss[0]:.4f} {miss[1][0]} ｜ {miss[1][1]}（{miss[1][3]}）")

    print("\n== 三、在開發集挑門檻，保留集只看一次 ==")
    labels_dev = [p[2] for p in DEV]
    threshold = pick_threshold(dev_scores, labels_dev)
    print("dev 挑出的門檻:", threshold, metrics(dev_scores, labels_dev, threshold))
    hold_scores = score_pairs(model, HOLDOUT)
    labels_hold = [p[2] for p in HOLDOUT]
    print("holdout 同一門檻:", metrics(hold_scores, labels_hold, threshold))
    print("對照：全部判「不相關」的 holdout accuracy:", round(labels_hold.count(0) / len(labels_hold), 3))
    cheat = pick_threshold(hold_scores, labels_hold)
    print("（錯誤示範）在 holdout 上挑門檻:", cheat, metrics(hold_scores, labels_hold, cheat))
    for kind in ("否定", "數字", "型號"):
        rows = [(s, p) for p, s in zip(HOLDOUT, hold_scores) if p[3] == kind]
        print(f"holdout {kind}:", [(f"{s:.4f}", "相關" if p[2] else "不相關") for s, p in rows])

    if args.pair:
        print("\n== 自訂句對 ==")
        for (q, d), s in zip(args.pair, score_pairs(model, [(q, d, None, "自訂") for q, d in args.pair])):
            print(f"{s:.4f} {q} ｜ {d}")
