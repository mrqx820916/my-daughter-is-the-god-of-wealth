#!/usr/bin/env python3
"""
小说阅读服务器 - 自动扫描chapters目录，生成API数据
支持语音听书（配合tts_server.py使用）
用法: python3 serve.py
访问: http://你的IP:8083
"""
import os, re, json, time, http.server, socketserver, urllib.request, urllib.parse, urllib.error

NOVEL_DIR = os.path.dirname(os.path.abspath(__file__))
CHAPTERS_DIR = os.path.join(NOVEL_DIR, "chapters")

VOL_NAMES = {
    "01": "第一卷：财神降世",
    "02": "第二卷：石破天惊", 
    "03": "第三卷：股海翻云",
    "04": "第四卷：封神之路"
}

def parse_chapters():
    volumes = {}
    if not os.path.exists(CHAPTERS_DIR):
        return volumes
    for f in sorted(os.listdir(CHAPTERS_DIR)):
        if not f.endswith('.md'):
            continue
        m = re.match(r'V(\d+)C(\d+)_(.+)\.md', f)
        if not m:
            continue
        vol, ch, title = m.group(1), m.group(2), m.group(3)
        ch_num = int(ch)
        if vol not in volumes:
            volumes[vol] = []
        try:
            with open(os.path.join(CHAPTERS_DIR, f), 'r', encoding='utf-8') as fp:
                raw = fp.read()
                content = md_to_html(raw)
                # 同时保存纯文本版本供TTS使用
                plain_text = md_to_plain(raw)
        except Exception as e:
            content = f'<p class="error">章节读取失败: {e}</p>'
            plain_text = ''
        volumes[vol].append({'num': ch_num, 'title': title, 'content': content, 'plain': plain_text, 'prefix': f'V{vol}C{ch}'})
    for v in volumes:
        volumes[v].sort(key=lambda x: x['num'])
    return volumes

def md_to_html(text):
    """Markdown → HTML 转换器，处理【】内心OS、标题、分隔线等"""
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
            level = len(stripped) - len(stripped.lstrip('#'))
            level = min(level, 6)
            heading_text = stripped.lstrip('#').strip()
            html.append(f'<h{level}>{heading_text}</h{level}>')
            continue
        
        if stripped.startswith('【') and stripped.endswith('】') and len(stripped) > 2:
            html.append(f'<div class="thought">{stripped[1:-1]}</div>')
            continue
        
        if stripped.startswith('【') and not stripped.endswith('】'):
            if in_thought:
                html.append('</div>')
            html.append(f'<div class="thought">{stripped[1:]}')
            in_thought = True
            continue
        
        if in_thought:
            if stripped.endswith('】'):
                html.append(f'{stripped[:-1]}</div>')
                in_thought = False
            else:
                html.append(stripped)
            continue
        
        safe_line = stripped.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html.append(f'<p>{safe_line}</p>')
    
    if in_thought:
        html.append('</div>')
    
    return '\n'.join(html)

def md_to_plain(text):
    """Markdown → 纯文本，供TTS使用"""
    lines = text.strip().split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == '---' or stripped.startswith('#'):
            continue
        # 去掉【】但保留内心OS文字
        stripped = stripped.replace('【', '').replace('】', '')
        result.append(stripped)
    return '\n'.join(result)

class NovelHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/chapters':
            try:
                volumes = parse_chapters()
                result = []
                for vol in sorted(volumes.keys()):
                    result.append({'vol': VOL_NAMES.get(vol, f'第{vol}卷'), 'ch': volumes[vol]})
                data = json.dumps(result, ensure_ascii=False)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(data.encode('utf-8'))
            except Exception as e:
                error = json.dumps({'error': str(e)}, ensure_ascii=False)
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(error.encode('utf-8'))
        elif self.path == '/voice' or self.path == '/voice.html':
            self.path = '/voice.html'
            super().do_GET()
        elif self.path.startswith('/api/tts/'):
            # TTS API 代理 → 转发到 8084
            self._proxy_tts('GET')
        else:
            # 静态文件（含 /audio/ 目录）由 SimpleHTTPServer 直接提供
            super().do_GET()
    
    def do_POST(self):
        if self.path.startswith('/api/tts/'):
            self._proxy_tts('POST')
        elif self.path.startswith('/api/synthesize'):
            self._proxy_tts('POST')
        else:
            self.send_error(404)
    
    def _proxy_tts(self, method):
        """将请求代理到 TTS 服务器 (8084)"""
        TTS_PORT = 8084
        try:
            # 构造目标URL
            target_path = self.path
            if target_path.startswith('/api/tts/'):
                target_path = target_path.replace('/api/tts/', '/api/', 1)
            
            url = f'http://127.0.0.1:{TTS_PORT}{target_path}'
            
            # 根据路径决定超时：音频合成需要长超时，状态查询短超时
            is_synth = 'synthesize' in target_path
            timeout = 600 if (method == 'POST' or is_synth) else 8
            
            if method == 'GET':
                req = urllib.request.Request(url)
            else:
                # POST: 读取请求体并转发
                content_len = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_len) if content_len > 0 else b''
                req = urllib.request.Request(url, data=body, method='POST')
                if 'Content-Type' in self.headers:
                    req.add_header('Content-Type', self.headers['Content-Type'])
            
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_data = resp.read()
                self.send_response(resp.status)
                # 透传所有响应头
                for key, val in resp.getheaders():
                    if key.lower() not in ('transfer-encoding', 'connection'):
                        self.send_header(key, val)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(resp_data)
        except urllib.error.URLError as e:
            err_msg = json.dumps({'error': f'TTS服务未启动: {e.reason}'}, ensure_ascii=False)
            self.send_response(502)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(err_msg.encode('utf-8'))
        except Exception as e:
            err_msg = json.dumps({'error': str(e)}, ensure_ascii=False)
            self.send_response(500)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(err_msg.encode('utf-8'))
    
    def log_message(self, format, *args):
        pass

if __name__ == '__main__':
    PORT = 8083
    os.chdir(NOVEL_DIR)
    for attempt in range(10):
        try:
            with socketserver.TCPServer(("", PORT), NovelHandler, bind_and_activate=False) as httpd:
                httpd.allow_reuse_address = True
                httpd.server_bind()
                httpd.server_activate()
                chapter_count = sum(len(v) for v in parse_chapters().values())
                vol_count = len(parse_chapters())
                print(f"📖 小说阅读服务已启动")
                print(f"   地址: http://0.0.0.0:{PORT}")
                print(f"   共 {vol_count} 卷 {chapter_count} 章")
                print(f"   语音试听: http://0.0.0.0:{PORT}/voice.html")
                httpd.serve_forever()
        except OSError as e:
            if 'Address already in use' in str(e):
                if attempt < 9:
                    time.sleep(2)
                    continue
                print(f"❌ 端口 {PORT} 被占用，请手动释放或更换端口")
                break
            raise
