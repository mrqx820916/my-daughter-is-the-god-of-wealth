#!/bin/bash
# 《我的女儿是财神》构建脚本
# 功能：统计信息 + 静态导出（生成可离线阅读的 HTML 文件）
# 用法: bash build.sh [export]

NOVEL_DIR="/mnt/zspace/worksp/xiaoshuo"
CHAPTERS_DIR="$NOVEL_DIR/chapters"
OUTPUT_DIR="$NOVEL_DIR/dist"

# 章节文件命名规范: V01C001_章节名.md
# V=卷号(01-04) C=章号(001-999)

VOL_NAMES_ZH=("第一卷：财神降世" "第二卷：石破天惊" "第三卷：股海翻云" "第四卷：封神之路")

echo "📖 构建小说..."
echo ""

# ─── 统计信息 ───
chapter_count=0
total_chars=0

for vol_idx in 0 1 2 3; do
  vol=$(printf "%02d" $((vol_idx + 1)))
  vol_chapters=$(ls "$CHAPTERS_DIR"/V${vol}C*.md 2>/dev/null | sort)
  
  if [ -z "$vol_chapters" ]; then
    echo "  ${VOL_NAMES_ZH[$vol_idx]}: 暂无章节"
    continue
  fi
  
  vol_count=$(echo "$vol_chapters" | wc -l)
  vol_chars=0
  for f in $vol_chapters; do
    chars=$(wc -m < "$f")
    vol_chars=$((vol_chars + chars))
  done
  
  chapter_count=$((chapter_count + vol_count))
  total_chars=$((total_chars + vol_chars))
  
  # 格式化字数（万）
  vol_wan=$(echo "scale=1; $vol_chars / 10000" | bc)
  echo "  ${VOL_NAMES_ZH[$vol_idx]}: ${vol_count}章 / ${vol_wan}万字"
done

total_wan=$(echo "scale=1; $total_chars / 10000" | bc)
echo ""
echo "✅ 共 ${chapter_count} 章 / ${total_wan} 万字"
echo ""

# ─── 如果参数是 export，执行静态导出 ───
if [ "$1" = "export" ]; then
  echo "📦 正在导出静态 HTML..."
  
  mkdir -p "$OUTPUT_DIR"
  
  python3 -c "
import os, re, json, sys

NOVEL_DIR = '$NOVEL_DIR'
CHAPTERS_DIR = os.path.join(NOVEL_DIR, 'chapters')
OUTPUT_DIR = '$OUTPUT_DIR'

VOL_NAMES = {
    '01': '第一卷：财神降世',
    '02': '第二卷：石破天惊',
    '03': '第三卷：股海翻云',
    '04': '第四卷：封神之路'
}

def md_to_html(text):
    lines = text.strip().split('\n')
    html = []
    in_thought = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_thought:
                html.append('</div>')
                in_thought = False
            continue
        if stripped == '---':
            if in_thought:
                html.append('</div>')
                in_thought = False
            html.append('<hr>')
            continue
        if stripped.startswith('#') and not stripped.startswith('【'):
            if in_thought:
                html.append('</div>')
                in_thought = False
            level = min(len(stripped) - len(stripped.lstrip('#')), 6)
            heading = stripped.lstrip('#').strip()
            html.append(f'<h{level}>{heading}</h{level}>')
            continue
        if stripped.startswith('【') and stripped.endswith('】') and len(stripped) > 2:
            html.append(f'<div class=\"thought\">{stripped[1:-1]}</div>')
            continue
        if stripped.startswith('【') and not stripped.endswith('】'):
            if in_thought:
                html.append('</div>')
            html.append(f'<div class=\"thought\">{stripped[1:]}')
            in_thought = True
            continue
        if in_thought:
            if stripped.endswith('】'):
                html.append(f'{stripped[:-1]}</div>')
                in_thought = False
            else:
                html.append(stripped)
            continue
        safe = stripped.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        html.append(f'<p>{safe}</p>')
    if in_thought:
        html.append('</div>')
    return '\n'.join(html)

# 读取 index.html 作为模板
with open(os.path.join(NOVEL_DIR, 'index.html'), 'r', encoding='utf-8') as f:
    template = f.read()

# 解析所有章节
volumes = {}
for fn in sorted(os.listdir(CHAPTERS_DIR)):
    if not fn.endswith('.md'):
        continue
    m = re.match(r'V(\d+)C(\d+)_(.+)\.md', fn)
    if not m:
        continue
    vol, ch, title = m.group(1), int(m.group(2)), m.group(3)
    with open(os.path.join(CHAPTERS_DIR, fn), 'r', encoding='utf-8') as f:
        content = md_to_html(f.read())
    volumes.setdefault(vol, []).append({'num': ch, 'title': title, 'content': content})

# 生成内联数据替换 API 调用
api_data = []
for vol in sorted(volumes.keys()):
    api_data.append({'vol': VOL_NAMES.get(vol, vol), 'ch': volumes[vol]})

data_json = json.dumps(api_data, ensure_ascii=False)

# 将 fetch API 调用替换为内联数据
output = template.replace(
    \"fetch('/api/chapters')\",
    \"Promise.resolve({json: () => \" + data_json + \"})\"
)
# 修复 response.json() 调用
output = output.replace(
    'response => response.json()',
    'response => response.json()'
)

# 直接嵌入数据：替换整个 fetch 块
old_fetch_pattern = r\"fetch\('/api/chapters'\).*?\.then\(response => response\.json\(\)\)\.then\(\s*data =>\"
import re
replacement = \"/* 内联数据 */ const data = \" + data_json + \"; data =>\"
output = re.sub(old_fetch_pattern, replacement, output, flags=re.DOTALL)

out_path = os.path.join(OUTPUT_DIR, 'index.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(output)

total = sum(len(v) for v in volumes.values())
print(f'  ✅ 已导出: {out_path}')
print(f'  📊 包含 {len(volumes)} 卷 {total} 章')
"
  
  echo ""
  echo "💡 导出文件位于: $OUTPUT_DIR/index.html"
  echo "   可直接在浏览器打开，无需服务器"
fi

echo ""
echo "💡 命令:"
echo "   bash build.sh          → 查看统计"
echo "   bash build.sh export   → 导出离线 HTML"
echo "   python3 serve.py       → 启动阅读服务 → http://0.0.0.0:8083"
