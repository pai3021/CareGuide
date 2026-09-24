"""Developer source inspection; no patient data or environment secrets."""
import re
from html.parser import HTMLParser
from pathlib import Path

import httpx


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ignore = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.ignore += 1

    def handle_endtag(self, tag):
        if tag in {'script', 'style'}:
            self.ignore = max(0, self.ignore - 1)

    def handle_data(self, data):
        if not self.ignore:
            self.parts.append(data)


def extract(raw):
    # Respect declared charset; these hospital pages use UTF-8 without HTTP charset.
    head = raw[:4096].decode('ascii', errors='ignore')
    match = re.search(r'charset\s*=\s*["\']?([\w-]+)', head, re.I)
    parser = TextParser()
    parser.feed(raw.decode(match[1] if match else 'utf-8'))
    return re.sub(r'\s+', ' ', ' '.join(parser.parts)).strip()


if __name__ == '__main__':
    urls = {
        'pkuih-neurology': 'https://www.pkuih.edu.cn/department/details_931_3308.html',
        'pkuih-gastroenterology': 'https://www.pkuih.edu.cn/department/details_931_3301.html',
        'pkuih-appointment': 'https://www.pkuih.edu.cn/notice/details_940_1487.html',
        'pkufh-daxing-appointment': 'https://www.pkufh.com/Html/News/Articles/61679.html',
        'nhs-headaches': 'https://www.nhs.uk/symptoms/headaches/',
    }
    target = Path('data/source_cache')
    target.mkdir(parents=True, exist_ok=True)
    for name, url in urls.items():
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        text = extract(response.content)
        (target / f'{name}.html').write_bytes(response.content)
        (target / f'{name}.txt').write_text(text, encoding='utf-8')
        keys = {'pkuih-neurology': '神经内科介绍',
                'pkuih-gastroenterology': '消化内科介绍',
                'pkuih-appointment': '放号时间',
                'pkufh-daxing-appointment': '发布时间',
                'nhs-headaches': 'When and where to get medical help'}
        index = text.find(keys[name])
        if index < 0:
            index = text.find('科室介绍')
        print(name, len(text), text[max(0, index-100):index+650])
