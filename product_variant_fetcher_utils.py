import os
import re
import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List, Tuple

USER_AGENT = "Mozilla/5.0 (compatible; ProductVariantFetcher/utility-3.0)"
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT = 20
DEBUG_DIR = os.getenv("VARIANT_FETCH_DEBUG_DIR", "variant_fetch_debug")

STOP_SECTIONS = {
    'Capacity', 'Capacity 1', 'Color', 'Finish', 'Size and Weight', 'Weight and Dimensions',
    'Display', 'Chip', 'Chips', 'Camera', 'Camera, Photos, and Video', 'Storage', 'Memory',
    'Internal Storage', 'Dimensions', 'Network', 'Battery', 'Overview', 'Specifications',
    'Features', 'Support', 'Manuals', 'Location', 'Cellular and Wireless', 'Available Colors', 'Colours'
}


def _safe_name(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9._-]+', '_', name)[:120]


def _write_debug(product_name: str, suffix: str, content: str):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    path = os.path.join(DEBUG_DIR, f"{_safe_name(product_name)}_{suffix}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def _get_requests(url: str) -> str:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def _get_playwright(url: str) -> str:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(url, wait_until='networkidle', timeout=30000)
        html = page.content()
        browser.close()
        return html


def _search_web_fallback(query: str) -> List[str]:
    urls = []
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=10):
                href = item.get('href') or item.get('url')
                if href:
                    urls.append(href)
    except Exception:
        pass
    return urls


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    return soup.get_text('\n', strip=True)


def _extract_title(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    return soup.title.string.strip() if soup.title and soup.title.string else ''


def _normalize_storage_token(text: str) -> Optional[str]:
    m = re.search(r'\b(\d+\s?(?:GB|TB))\b', text, flags=re.I)
    return re.sub(r'\s+', '', m.group(1).upper()) if m else None


def _extract_capacities(text: str) -> List[str]:
    found = re.findall(r'\b(\d+\s?(?:GB|TB))\b', text, flags=re.I)
    return sorted({re.sub(r'\s+', '', x.upper()) for x in found})


def _extract_sizes(text: str) -> List[str]:
    patterns = [r'\b\d{2}mm\b', r'\b\d(?:\.\d)?-inch\b', r'\b\d(?:\.\d)?\s*inch\b']
    out = []
    for p in patterns:
        out.extend(re.findall(p, text, flags=re.I))
    return sorted(set(x.replace(' inch', '-inch') for x in out))


def _extract_colors_generic(text: str) -> List[str]:
    known_colors = [
        'Black','White','Blue','Pink','Silver','Gold','Gray','Grey','Green','Purple','Yellow','Red','Midnight',
        'Starlight','Space Gray','Space Black','Rose Gold','Natural Titanium','Blue Titanium','Black Titanium',
        'Silver Titanium','Titanium Black','Titanium Gray','Titanium Blue','Phantom Black','Cream','Navy',
        'Lavender','Mint','Graphite','Pink Gold','Denim','Clay','Stone Gray','Lake Green','Sand',
        'Black & Slate','White & Silver','Pebble Blue','Titanium Silver','Silver Shadow'
    ]
    found = []
    for c in known_colors:
        if re.search(r'\b' + re.escape(c) + r'\b', text, flags=re.I):
            found.append(c)
    return sorted(set(found))


def _extract_section_values(text: str, section_names: List[str], max_lines: int = 10) -> List[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    values = []
    section_set = set(section_names)
    for i, line in enumerate(lines):
        if line in section_set:
            collected = 0
            for nxt in lines[i+1:]:
                if nxt in STOP_SECTIONS and nxt not in section_set:
                    break
                if collected >= max_lines:
                    break
                values.append(nxt)
                collected += 1
    return values


def _normalize_color_list(values: List[str]) -> List[str]:
    cleaned = []
    for x in values:
        x = re.sub(r'^[•\-]+\s*', '', x).strip()
        if not x or len(x) > 60:
            continue
        if re.search(r'\b(GB|TB|mm|inch|Height|Width|Depth|Weight)\b', x, flags=re.I):
            continue
        cleaned.append(x)
    return sorted(set(cleaned))


def _normalize_storage_list(values: List[str]) -> List[str]:
    out = []
    for x in values:
        tok = _normalize_storage_token(x)
        if tok:
            out.append(tok)
    return sorted(set(out))


def _parse_apple_support(text: str) -> Dict[str, Any]:
    capacities = _normalize_storage_list(_extract_section_values(text, ['Capacity', 'Capacity 1'], max_lines=12))
    colors = _normalize_color_list(_extract_section_values(text, ['Color', 'Finish'], max_lines=8))
    sizes = _extract_sizes(text)
    if not capacities:
        capacities = _extract_capacities(text)
    if not colors:
        colors = _extract_colors_generic(text)
    return {'storages': capacities, 'sizes': sizes, 'colors': colors}


def _parse_samsung_support(text: str) -> Dict[str, Any]:
    capacities = _normalize_storage_list(_extract_section_values(text, ['Storage', 'Memory', 'Internal Storage'], max_lines=12))
    colors = _normalize_color_list(_extract_section_values(text, ['Color', 'Colours', 'Available Colors', 'Finish'], max_lines=10))
    size_values = _extract_section_values(text, ['Size', 'Dimensions', 'Display Size'], max_lines=10)
    sizes = _extract_sizes('\n'.join(size_values) + '\n' + text)
    if not capacities:
        capacities = _extract_capacities(text)
    if not colors:
        colors = _extract_colors_generic(text)
    return {'storages': capacities, 'sizes': sizes, 'colors': colors}


def _parse_generic(text: str) -> Dict[str, Any]:
    return {
        'storages': _extract_capacities(text),
        'sizes': _extract_sizes(text),
        'colors': _extract_colors_generic(text),
    }


def _rank_url(url: str, product_name: str, brand: Optional[str]) -> int:
    score = 0
    u = url.lower()
    p = product_name.lower()
    if brand and brand.lower() == 'apple':
        if 'support.apple.com' in u:
            score += 50
        if '/docs/' in u:
            score -= 120
        if 'technical-specifications' in u:
            score += 80
        if re.search(r'/\d{6}$', u):
            score += 20
        if 'manuals' in u or 'downloads' in u:
            score -= 100
    if brand and brand.lower() == 'samsung':
        if 'samsung.com/us/support' in u:
            score += 50
        if 'owners/product' in u:
            score += 25
        elif 'support' in u:
            score += 15
    tokens = [t for t in re.split(r'\W+', p) if t]
    score += sum(3 for t in tokens[:4] if t in u)
    return score


def _candidate_urls(product_name: str, brand: Optional[str]) -> List[str]:
    if brand and brand.lower() == 'apple':
        queries = [
            f'site:support.apple.com "{product_name}" "Technical Specifications"',
            f'site:support.apple.com "{product_name}" technical specifications',
            f'site:apple.com "{product_name}" specs'
        ]
    elif brand and brand.lower() == 'samsung':
        queries = [
            f'site:samsung.com/us/support "{product_name}" specs',
            f'site:samsung.com/us/support/owners/product "{product_name}"',
            f'site:samsung.com "{product_name}" support'
        ]
    else:
        queries = [
            f'site:support.apple.com "{product_name}" "Technical Specifications"',
            f'site:samsung.com/us/support "{product_name}" specs',
            f'"{product_name}" technical specifications'
        ]

    seen = set()
    ranked = []
    for q in queries:
        for url in _search_web_fallback(q):
            if url in seen:
                continue
            if any(domain in url for domain in ['support.apple.com', 'apple.com', 'samsung.com']):
                seen.add(url)
                ranked.append((url, _rank_url(url, product_name, brand)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return [u for u, _ in ranked]


def _normalize_apple_model(product_name: str) -> Dict[str, Any]:
    p = product_name.lower().strip()
    aliases = [p]
    must_have = []
    must_not_have = []

    if 'iphone se (1st generation)' in p or 'iphone se 1st generation' in p:
        aliases += ['iphone se', '1st generation', 'first generation']
        must_have += ['iphone se']
        must_not_have += ['ipad pro', 'ipad', 'iphone 11', 'iphone 12', 'iphone 13', 'iphone 14', 'iphone 15', 'iphone 16']
    elif 'iphone 6' in p:
        aliases += ['iphone 6']
        must_have += ['iphone 6']
        must_not_have += ['iphone 6s', 'iphone 11', 'iphone se']
    elif 'iphone 5' in p:
        aliases += ['iphone 5']
        must_have += ['iphone 5']
        must_not_have += ['iphone 5s', 'iphone se', 'iphone 11']
    elif 'ipad mini 1st generation' in p or 'ipad mini (1st generation)' in p:
        aliases += ['ipad mini', '1st generation', 'first generation']
        must_have += ['ipad mini']
        must_not_have += ['6th generation', '5th generation', '4th generation', 'ipad air']
    elif re.search(r'\bipad 2\b', p):
        aliases += ['ipad 2']
        must_have += ['ipad 2']
        must_not_have += ['ipad air 2', 'ipad pro', 'ipad mini']
    elif 'apple watch series 0' in p or 'apple watch (1st generation)' in p:
        aliases += ['apple watch', '1st generation', 'first generation', 'series 0']
        must_have += ['apple watch']
        must_not_have += ['series 2', 'series 3', 'ultra', 'se']

    return {'aliases': list(dict.fromkeys(aliases)), 'must_have': must_have, 'must_not_have': must_not_have}


def _apple_model_matches(product_name: str, title: str, text: str) -> bool:
    title_l = title.lower()
    body = text.lower()
    norm = _normalize_apple_model(product_name)

    if not any(alias in title_l or alias in body for alias in norm['aliases']):
        return False
    if any(term not in title_l and term not in body for term in norm['must_have']):
        return False
    if any(term in title_l or term in body for term in norm['must_not_have']):
        return False
    if 'technical specifications' not in title_l and 'capacity' not in body and 'finish' not in body and 'color' not in body:
        return False
    return True


def _fetch_candidate(url: str, product_name: str) -> Tuple[str, str, str]:
    errors = []
    try:
        html = _get_requests(url)
        text = _clean_text(html)
        title = _extract_title(html)
        return html, text, title
    except Exception as e:
        errors.append(f'requests: {e}')
    try:
        html = _get_playwright(url)
        text = _clean_text(html)
        title = _extract_title(html)
        return html, text, title
    except Exception as e:
        errors.append(f'playwright: {e}')
    raise RuntimeError(' | '.join(errors))


def _find_best_source(product_name: str, brand: Optional[str]) -> Dict[str, Any]:
    urls = _candidate_urls(product_name, brand)
    if not urls:
        return {'url': None, 'title': None, 'text': None}

    if brand and brand.lower() == 'apple':
        urls = [u for u in urls if 'support.apple.com' in u.lower() and '/docs/' not in u.lower()]
        for idx, url in enumerate(urls[:6], start=1):
            try:
                html, text, title = _fetch_candidate(url, product_name)
                _write_debug(product_name, f'candidate_{idx}.url.txt', url)
                _write_debug(product_name, f'candidate_{idx}.title.txt', title)
                _write_debug(product_name, f'candidate_{idx}.txt', text[:8000])
                if _apple_model_matches(product_name, title, text):
                    return {'url': url, 'title': title, 'text': text}
            except Exception as e:
                _write_debug(product_name, f'candidate_{idx}.error.txt', str(e))
        return {'url': urls[0] if urls else None, 'title': None, 'text': None}

    return {'url': urls[0], 'title': None, 'text': None}


def _fetch_with_fallback(url: str, product_name: str) -> Dict[str, str]:
    errors = []
    try:
        html = _get_requests(url)
        _write_debug(product_name, 'requests.html', html)
        text = _clean_text(html)
        _write_debug(product_name, 'requests.txt', text)
        return {'html': html, 'text': text, 'method': 'requests'}
    except Exception as e:
        errors.append(f'requests: {e}')

    try:
        html = _get_playwright(url)
        _write_debug(product_name, 'playwright.html', html)
        text = _clean_text(html)
        _write_debug(product_name, 'playwright.txt', text)
        return {'html': html, 'text': text, 'method': 'playwright'}
    except Exception as e:
        errors.append(f'playwright: {e}')

    raise RuntimeError(' | '.join(errors))


def fetch_product_variants(product_name: str, brand_hint: Optional[str] = None) -> Dict[str, Any]:
    source = _find_best_source(product_name, brand_hint)
    if not source.get('url'):
        return {
            'product_name': product_name,
            'brand_hint': brand_hint,
            'source_url': None,
            'variants': {'storages': [], 'sizes': [], 'colors': []},
            'evidence': None,
            'status': 'no_source_found',
            'fetch_method': None,
            'debug_dir': DEBUG_DIR,
        }

    try:
        if source.get('text'):
            text = source['text']
            fetched = {'method': 'validated_candidate'}
            _write_debug(product_name, 'validated_source_url.txt', source['url'])
            _write_debug(product_name, 'validated_source_title.txt', source.get('title') or '')
        else:
            fetched = _fetch_with_fallback(source['url'], product_name)
            text = fetched['text']

        if 'support.apple.com' in source['url']:
            variants = _parse_apple_support(text)
        elif 'samsung.com' in source['url']:
            variants = _parse_samsung_support(text)
        else:
            variants = _parse_generic(text)

        if not any(variants.values()) and fetched['method'] in ('requests', 'validated_candidate'):
            try:
                html = _get_playwright(source['url'])
                _write_debug(product_name, 'playwright_retry.html', html)
                text = _clean_text(html)
                _write_debug(product_name, 'playwright_retry.txt', text)
                if 'support.apple.com' in source['url']:
                    variants = _parse_apple_support(text)
                elif 'samsung.com' in source['url']:
                    variants = _parse_samsung_support(text)
                else:
                    variants = _parse_generic(text)
                fetched['method'] = 'playwright_retry'
            except Exception as e:
                _write_debug(product_name, 'playwright_retry.error.txt', str(e))

        _write_debug(product_name, 'variants.json', str(variants))
        return {
            'product_name': product_name,
            'brand_hint': brand_hint,
            'source_url': source['url'],
            'variants': variants,
            'evidence': text[:5000],
            'status': 'ok',
            'fetch_method': fetched['method'],
            'debug_dir': DEBUG_DIR,
        }
    except Exception as e:
        _write_debug(product_name, 'fatal.error.txt', str(e))
        return {
            'product_name': product_name,
            'brand_hint': brand_hint,
            'source_url': source.get('url'),
            'variants': {'storages': [], 'sizes': [], 'colors': []},
            'evidence': str(e),
            'status': 'error',
            'fetch_method': None,
            'debug_dir': DEBUG_DIR,
        }
