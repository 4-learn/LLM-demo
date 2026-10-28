#!/usr/bin/env python3
"""Generate EPIC #12's auditable course inventory from the mapped repo tree."""
from __future__ import annotations
import argparse, csv, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT/'docs/course-mapping.csv'
API_COMPLETE = {'07','15','16','17','27','28','31','33','34','35','41','42','50'}
API_PARTIAL = {'05','10','12','14','25','29','38','43','45','51','E01','E02','E03','E04','E05'}
API_NO = {'03','09','48','49','E07','E09'}
LONG = {'14','16','23','27','28','33','35','38','39','40','42','43','44','45','46','48','E01','E02','E03','E07','E09'}


def classify(text, files):
    s=text+'\n'+'\n'.join(p.read_text(encoding='utf-8',errors='ignore') for p in files)
    checks=[('replay/fixture',r'fixture|重播|Replay|錄製'),('HTTP/API live',r'NVIDIA_API_KEY|chat/completions|chat\.completions\.create|urllib\.request|requests\.post'),('HF/local inference',r'AutoModel|from_pretrained|\.generate\(|sentence_transformers|SentenceTransformer'),('tool/MCP',r'\bMCP\b|tools=|tool_call|tool_calls')]
    found=[name for name,pat in checks if re.search(pat,s,re.I)]
    return '; '.join(found) or '未見模型呼叫（純流程/演算法或缺 demo）'


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--mapping',type=pathlib.Path,default=DEFAULT_MAPPING); ap.add_argument('--output',type=pathlib.Path,default=ROOT/'docs/course-inventory.md'); ap.add_argument('--check',action='store_true'); a=ap.parse_args(argv)
    if not a.mapping.is_file():
        raise SystemExit(f'mapping file not found: {a.mapping}; pass --mapping PATH to the authoritative 60-course CSV')
    rows=list(csv.DictReader(a.mapping.open(encoding='utf-8-sig')))
    if len(rows)!=60: raise SystemExit(f'mapping rows={len(rows)}, expected 60')
    out=['# 60 課呼叫、圖示與長程式盤點','',f'> 盤點來源：課程 mapping（{len(rows)} 節）與目前目標 main 基線；此 inventory 由 `tools/build_epic12_inventory.py` 產生。Demo 檔依講義 `<!-- demo: ... -->` 錨點核對；行數為錨點所指 demo 最大行數。','', '| 課號 | 課名 | 講義 | Demo 檔案（依錨點） | 行數(最大) | Mermaid flowchart 數 | 執行型態/依賴線索 | Build SDK 結論 |','|---|---|---|---|---:|---:|---|---|']
    missing=[]
    for r in rows:
        id=r['new_id']; mds=sorted(ROOT.glob(id+'-*.md')); md=mds[0] if mds else (ROOT/'51-acceptance.md' if id=='51' and (ROOT/'51-acceptance.md').exists() else None)
        anchors=re.findall(r'<!--\s*demo:\s*([^\s]+\.py)\s*-->',md.read_text(encoding='utf8') if md else '')
        pys=[ROOT/name for name in dict.fromkeys(anchors) if (ROOT/name).is_file()]
        missing_demos=[name for name in dict.fromkeys(anchors) if not (ROOT/name).is_file()]
        allfiles=list(dict.fromkeys(([md] if md else [])+pys))
        lines=max((len(p.read_text(encoding='utf8',errors='ignore').splitlines()) for p in allfiles if p.suffix=='.py'),default=0)
        charts=sum(1 for x in md.read_text(encoding='utf8',errors='ignore').splitlines() if x.strip()=='```mermaid' and md) if md else 0
        if not md and id!='51': missing.append(id)
        if not anchors and id!='51': missing.append(f'{id}:demo-anchor')
        if missing_demos: missing.extend(f'{id}:missing-demo:{name}' for name in missing_demos)
        filename=', '.join(f'`{p.name}`' for p in pys) or '—'
        if id in API_COMPLETE: decision='可完整替換文字 LLM live 呼叫'
        elif id in API_PARTIAL: decision='部分可替換；保留本地/工具/驗證流程'
        elif id in API_NO: decision='不改呼叫方式；可補本機 vs API 對照'
        else: decision='依課程目的不需/不宜改，或非模型呼叫'
        note=classify(md.read_text(encoding='utf8',errors='ignore') if md else '',allfiles)
        out.append(f"| {id} | {r['title']} | `{md.name if md else 'MISSING'}` | {filename} | {lines} | {charts} | {note} | {decision} |")
    out+=['','## 覆蓋核對','',f'- Mapping 課數：{len(rows)}（必修 {sum(r["track"]=="core" for r in rows)}、選修 {sum(r["track"]=="elective" for r in rows)}）。','- 指定長 demo 集合：'+', '.join(sorted(LONG))+'；程式行數與講義需逐項人工核對。', '- 講義缺檔：'+(', '.join(missing) if missing else '無（課號 51 無對應講義是既有 mapping/README 特例，請補課或明確標記為彙總驗收）'), '- 呼叫型態以源碼掃描關鍵詞為提示，逐課決策需人工複核；fixture/replay 不等同執行 live 模型。','']
    content='\n'.join(out)
    if a.check:
        old=a.output.read_text(encoding='utf8') if a.output.exists() else ''
        if content!=old: raise SystemExit('inventory stale or missing; run without --check')
        if missing: raise SystemExit('missing lecture(s): '+', '.join(missing))
        print('inventory current; 60 mapped courses')
    else:
        a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(content,encoding='utf8'); print(f'wrote {a.output}')
if __name__=='__main__': main()
