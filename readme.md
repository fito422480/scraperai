# ScraperAI

[![PyPI version](https://img.shields.io/pypi/v/scraperai?logo=pypi)](https://pypi.org/project/scraperai/)
[![Python versions](https://img.shields.io/pypi/pyversions/scraperai?logo=python)](https://pypi.org/project/scraperai/)
[![Tests](https://github.com/scraperai/scraperai/actions/workflows/tests.yml/badge.svg)](https://github.com/scraperai/scraperai/actions/workflows/tests.yml)
[![License](https://img.shields.io/github/license/scraperai/scraperai)](LICENSE)

ES | [EN](#english)

ScraperAI es un framework de scraping asistido por IA para detectar estructura web y extraer datos en pipelines reutilizables.

## Para Usuario Final (rápido)

### 1) Instalar

```bash
pip install scraperai
```

### 2) Configurar API Key

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

### 3) Ejecutar CLI

```bash
scraperai --url https://www.ycombinator.com/companies
```

O:

```bash
scraperai
```

### 4) Qué hace la CLI

1. Detecta tipo de página (`catalog`, `detailed_page`, `captcha`, `other`)
2. Detecta paginación (`xpath`, `scroll`, `urls`, `none`)
3. Detecta ítems repetitivos (cards/rows)
4. Detecta campos a extraer
5. Ejecuta scraping con límites (`max_pages`, `max_rows`)
6. Exporta resultados (`json`, `csv`, `xlsx`)

## Para Developer API (técnico)

### Arquitectura

- `ParserAI`: detección de tipo/paginación/card/campos
- `Scraper`: ejecución de scraping según `ScraperConfig`
- `SeleniumCrawler`: crawler por defecto para páginas dinámicas
- `RequestsCrawler`: crawler simple para páginas estáticas

### Ejemplo end-to-end

```python
from scraperai import ParserAI, Scraper, SeleniumCrawler
from scraperai.models import ScraperConfig

start_url = "https://www.ycombinator.com/companies"

crawler = SeleniumCrawler()
crawler.get(start_url)

parser = ParserAI(openai_api_key="sk-...")

page_type = parser.detect_page_type(
    page_source=crawler.page_source,
    screenshot=crawler.get_screenshot_as_base64(),
)
pagination = parser.detect_pagination(crawler.page_source)
catalog_item = parser.detect_catalog_item(crawler.page_source, start_url)
fields = parser.extract_fields(catalog_item.html_snippet)

config = ScraperConfig(
    start_url=start_url,
    page_type=page_type,
    pagination=pagination,
    catalog_item=catalog_item,
    open_nested_pages=False,
    fields=fields,
    max_pages=3,
    max_rows=100,
)

rows = list(Scraper(config=config, crawler=crawler).scrape())
print(f"Scraped {len(rows)} rows")
```

### ScraperConfig (resumen)

- `start_url: str`
- `page_type: catalog | detailed_page | captcha | other`
- `pagination: type/xpath/urls`
- `catalog_item: card_xpath/url_xpath`
- `open_nested_pages: bool`
- `fields: static_fields + dynamic_fields`
- `max_pages: int`
- `max_rows: int`

### Extensión

Crawler custom:

```python
from scraperai import BaseCrawler
from scraperai.models import Pagination

class MyCrawler(BaseCrawler):
    def get(self, url: str): ...

    @property
    def page_source(self) -> str: ...

    def switch_page(self, pagination: Pagination) -> bool:
        return False
```

Modelos custom:

- JSON: hereda `BaseJsonLM`
- Visión: hereda `BaseVision`

## Requisitos

- Python 3.10+
- Chrome/Chromium (para `SeleniumCrawler`)
- `OPENAI_API_KEY` (flujo por defecto)

## Desarrollo

```bash
pip install -r requirements.txt
pytest -q
```

## Ejemplos

- `examples/ycombinator_full.ipynb`
- `examples/techcrunch.ipynb`
- `examples/github_user.ipynb`
- `examples/ikea.ipynb`
- `examples/acronis_jobs.ipynb`

## Problemas comunes

- Falta API key: define `OPENAI_API_KEY` o `.env`
- Sitio JS-heavy con resultados pobres: usa `SeleniumCrawler`
- CAPTCHA detectado: no hay bypass automático incluido

## Limitaciones

- No resuelve CAPTCHA
- Sin scraping async actualmente
- La calidad depende del HTML objetivo y del modelo LLM

## Licencia

GPL-3.0. Ver `LICENSE`.

---

## English

ScraperAI is an AI-assisted scraping framework that detects webpage structure and extracts data into reusable pipelines.

## For End Users (quick path)

### 1) Install

```bash
pip install scraperai
```

### 2) Set API key

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
```

### 3) Run CLI

```bash
scraperai --url https://www.ycombinator.com/companies
```

Or:

```bash
scraperai
```

### 4) What CLI does

1. Detects page type (`catalog`, `detailed_page`, `captcha`, `other`)
2. Detects pagination (`xpath`, `scroll`, `urls`, `none`)
3. Detects repeated items (cards/rows)
4. Detects extractable fields
5. Runs scraping with limits (`max_pages`, `max_rows`)
6. Exports results (`json`, `csv`, `xlsx`)

## For API Developers (technical path)

### Architecture

- `ParserAI`: detects page type/pagination/card/fields
- `Scraper`: executes scraping from `ScraperConfig`
- `SeleniumCrawler`: default crawler for dynamic pages
- `RequestsCrawler`: lightweight crawler for static pages

### End-to-end example

```python
from scraperai import ParserAI, Scraper, SeleniumCrawler
from scraperai.models import ScraperConfig

start_url = "https://www.ycombinator.com/companies"

crawler = SeleniumCrawler()
crawler.get(start_url)

parser = ParserAI(openai_api_key="sk-...")

page_type = parser.detect_page_type(
    page_source=crawler.page_source,
    screenshot=crawler.get_screenshot_as_base64(),
)
pagination = parser.detect_pagination(crawler.page_source)
catalog_item = parser.detect_catalog_item(crawler.page_source, start_url)
fields = parser.extract_fields(catalog_item.html_snippet)

config = ScraperConfig(
    start_url=start_url,
    page_type=page_type,
    pagination=pagination,
    catalog_item=catalog_item,
    open_nested_pages=False,
    fields=fields,
    max_pages=3,
    max_rows=100,
)

rows = list(Scraper(config=config, crawler=crawler).scrape())
print(f"Scraped {len(rows)} rows")
```

### Extend with custom components

- Custom crawler: inherit `BaseCrawler`
- Custom JSON model: inherit `BaseJsonLM`
- Custom vision model: inherit `BaseVision`

## Requirements

- Python 3.10+
- Chrome/Chromium (for `SeleniumCrawler`)
- `OPENAI_API_KEY` (default LLM flow)

## Development

```bash
pip install -r requirements.txt
pytest -q
```

## License

GPL-3.0. See `LICENSE`.

