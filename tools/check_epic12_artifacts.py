#!/usr/bin/env python3
"""Check EPIC #12 lecture diagrams, demo anchors, and long-session pairs."""
from __future__ import annotations
import csv, pathlib, re, sys
ROOT=pathlib.Path(__file__).resolve().parents[1]
MAPPING=pathlib.Path('/home/dsh/workspace/swarm/hives/4-learn/course-plans/llm-restructure-mapping.csv')
if not MAPPING.is_file(): MAPPING=ROOT.parent/'course-plans'/'llm-restructure-mapping.csv'
LONG={'14','16','23','27','28','33','35','38','39','40','42','43','44','45','46','48','E01','E02','E03','E07','E09'}

def main():
 rows=list(csv.DictReader(MAPPING.open(encoding='utf-8'))); errors=[]
 lessons={}
 for row in rows:
  lid=row['new_id']; matches=sorted(ROOT.glob(f'{lid}-*.md'))
  md=matches[0] if matches else (ROOT/'51-acceptance.md' if lid=='51' else None)
  if not md or not md.is_file(): errors.append(f'{lid}: missing lesson markdown'); continue
  lessons[lid]=md; text=md.read_text(encoding='utf-8')
  blocks=re.findall(r'```mermaid\n(.*?)\n```',text,re.S)
  if not blocks: errors.append(f'{lid}: no Mermaid diagrams')
  for i,block in enumerate(blocks,1):
   if not re.search(r'^\s*(?:flowchart|graph)\s+',block,re.M): errors.append(f'{lid}: Mermaid block {i} lacks flowchart/graph declaration')
  if text.count('```mermaid')!=len(blocks): errors.append(f'{lid}: unclosed Mermaid fence')
  intro=re.search(r'^> \*\*(?:正文改版|新版正文)[^\n]*對應新版 LLM 第 ([^ 節]+) 節：([^。*]+)',text,re.M)
  if intro and (intro.group(1)!=lid or intro.group(2).strip()!=row['title']): errors.append(f'{lid}: intro mapping differs from canonical title/id')
  anchors=re.findall(r'<!--\s*demo:\s*([^\s]+\.py)\s*-->',text)
  if not anchors: errors.append(f'{lid}: missing demo anchor')
  for name in anchors:
   if not (ROOT/name).is_file(): errors.append(f'{lid}: missing demo file {name}')
 for lid in LONG:
  md=lessons.get(lid)
  if not md: continue
  text=md.read_text(encoding='utf-8')
  sessions=list(re.finditer(r'^### (?:Call )?[Ss]ession\s+\d+[^\n]*',text,re.M))
  if not sessions: errors.append(f'{lid}: no call-session headings'); continue
  for i,match in enumerate(sessions):
   end=sessions[i+1].start() if i+1<len(sessions) else len(text)
   section=text[match.start():end]
   charts=re.findall(r'```mermaid\n(.*?)\n```',section,re.S)
   if len(charts)!=2: errors.append(f'{lid}: {match.group()} has {len(charts)} diagrams, expected 2')
   if not any(label in section for label in ('系統層','System layer')): errors.append(f'{lid}: {match.group()} missing system-layer chart label')
   if not any(label in section for label in ('函式內部','Internal function')): errors.append(f'{lid}: {match.group()} missing internal-function chart label')
   for name in re.findall(r'<!--\s*demo:\s*([^\s]+\.py)\s*-->',section):
    if not (ROOT/name).is_file(): errors.append(f'{lid}: session points to missing demo {name}')
 print(f'checked {len(lessons)}/{len(rows)} mapped lessons; {len(LONG)} long-demo lessons')
 if errors:
  print('\n'.join(errors)); return 1
 print('all lesson flowcharts, demo anchors, and long-session pairs passed')
 return 0
if __name__=='__main__': sys.exit(main())
