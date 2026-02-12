from __future__ import annotations

import datetime
import json
import os
import uuid
from urllib.parse import urljoin

import boto3
import pandas as pd
import requests
import streamlit as st
from lxml import html  # type: ignore[import-not-found]

from scraperai import ParserAI, RequestsCrawler, Scraper
from scraperai.llm.openai import JsonOpenAI, PythonCodeOpenAI, VisionOpenAI
from scraperai.models import Pagination, ScraperConfig, WebpageType
from scraperai.utils.html import minify_html

MODEL_PRESETS = {
    "Qwen 72B (OpenRouter)": {
        "provider": "OpenRouter",
        "model": "qwen/qwen-2.5-72b-instruct",
        "tokens": 256,
    },
    "Qwen 7B (OpenRouter)": {
        "provider": "OpenRouter",
        "model": "qwen/qwen-2.5-7b-instruct",
        "tokens": 256,
    },
    "GPT-5.2 (OpenRouter)": {
        "provider": "OpenRouter",
        "model": "openai/gpt-5.2",
        "tokens": 256,
    },
    "GPT-4o (OpenAI)": {
        "provider": "OpenAI",
        "model": "gpt-4o",
        "tokens": 1024,
    },
    "DeepSeek Chat": {
        "provider": "DeepSeek",
        "model": "deepseek-chat",
        "tokens": 512,
    },
    "Personalizado": None,
}

MANUAL_DEFAULT_CARD_XPATH = (
    "//div[contains(@class,'promocao-item-info')]"
)

MANUAL_DEFAULT_FIELDS = (
    "titulo|normalize-space(.//div[contains(@class,'promocao-item-nome')]/a)|text\n"
    "url_detalle|.//div[contains(@class,'promocao-item-nome')]/a/@href|text\n"
    "codigo||text\n"
    "url_whatsapp||text\n"
    "url_loja||text\n"
    "precio_usd||text\n"
    "precio_brl||text\n"
    "advertiser||text\n"
    "category||text\n"
    "product||text"
)


st.set_page_config(
    page_title="ScraperAI Studio",
    page_icon="S",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 1.2rem; max-width: 1060px;}
    .hero {
        background: linear-gradient(120deg, #0f172a 0%, #1f2937 55%, #0ea5e9 100%);
        border-radius: 16px;
        padding: 22px 24px;
        color: #f8fafc;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 30px rgba(15, 23, 42, .18);
    }
    .hero h1 {margin: 0; font-size: 1.85rem;}
    .hero p {margin: .5rem 0 0 0; color: #e2e8f0;}
    .hint {font-size: .9rem; color: #64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _config_to_pretty_json(config: ScraperConfig) -> str:
    dump_json = getattr(config, "model_dump_json", None)
    if callable(dump_json):
        return dump_json(indent=2)

    dump_dict = getattr(config, "model_dump", None)
    if callable(dump_dict):
        return json.dumps(dump_dict(), indent=2, ensure_ascii=False)

    to_json = getattr(config, "json", None)
    if callable(to_json):
        return to_json(indent=2, ensure_ascii=False)

    to_dict = getattr(config, "dict", None)
    if callable(to_dict):
        return json.dumps(to_dict(), indent=2, ensure_ascii=False)

    return json.dumps(config.__dict__, indent=2, ensure_ascii=False)


def _normalize_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.json_normalize(rows)


def _upload_csv_to_s3(df: pd.DataFrame) -> tuple[str, str] | tuple[None, None]:
    bucket = os.getenv("CSV_BUCKET", "").strip()
    if not bucket or df.empty:
        return None, None

    key = f"exports/{datetime.date.today().isoformat()}/{uuid.uuid4()}.csv"
    region = os.getenv("AWS_REGION")
    s3 = boto3.client("s3", region_name=region)
    payload = df.to_csv(index=False).encode("utf-8")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=payload,
        ContentType="text/csv; charset=utf-8",
    )
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=3600,
    )
    return key, url


def _prepare_html_for_llm(html_text: str, max_chars: int = 6000) -> str:
    minified, _ = minify_html(
        html_text,
        good_attrs={"class", "href", "id", "src"},
        use_substituions=True,
    )
    if len(minified) <= max_chars:
        return minified
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return minified[:head] + "\n<!-- trimmed -->\n" + minified[-tail:]


def _build_config_with_ai(
    start_url: str,
    api_key: str,
    provider: str,
    model_name: str,
    site_url: str | None,
    site_name: str | None,
    max_output_tokens: int,
    max_pages: int,
    max_rows: int,
    open_nested_pages: bool,
    low_budget_mode: bool = False,
) -> ScraperConfig:
    crawler = RequestsCrawler()
    base_url = None
    default_headers = None
    if provider == "DeepSeek":
        base_url = "https://api.deepseek.com/v1"
    elif provider == "OpenRouter":
        base_url = "https://openrouter.ai/api/v1"
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_name:
            default_headers["X-Title"] = site_name

    common_kwargs = {
        "openai_api_key": api_key,
        "openai_organization": None,
        "model_name": model_name,
        "temperature": 0,
        "max_tokens": max_output_tokens,
    }
    if base_url:
        common_kwargs["openai_api_base"] = base_url
    if default_headers:
        common_kwargs["default_headers"] = default_headers

    json_model = JsonOpenAI(**common_kwargs)
    vision_model = VisionOpenAI(**common_kwargs)
    code_model = PythonCodeOpenAI(**common_kwargs)
    parser = ParserAI(
        json_lm_model=json_model,
        vision_model=vision_model,
        code_model=code_model,
    )
    crawler.get(start_url)
    low_budget_mode = low_budget_mode or provider == "OpenRouter"
    if low_budget_mode:
        llm_html = _prepare_html_for_llm(crawler.page_source, max_chars=5500)
    else:
        llm_html = crawler.page_source

    page_type = parser.detect_page_type(page_source=llm_html)
    if page_type in (WebpageType.OTHER, WebpageType.CAPTCHA):
        # Fallback to details so scraping can continue instead of returning zero rows.
        page_type = WebpageType.DETAILS
    pagination = Pagination(type="none", urls=[])
    catalog_item = None

    if page_type == WebpageType.CATALOG:
        if not low_budget_mode:
            detected_pagination = parser.detect_pagination(llm_html)
            if detected_pagination.type == "urls":
                pagination = detected_pagination

        catalog_item = parser.detect_catalog_item(
            page_source=llm_html,
            website_url=start_url,
        )

        if open_nested_pages and catalog_item and catalog_item.urls_on_page:
            crawler.get(catalog_item.urls_on_page[0])
            if low_budget_mode:
                nested_html = _prepare_html_for_llm(
                    crawler.page_source, max_chars=3500
                )
            else:
                nested_html = crawler.page_source
            if low_budget_mode:
                html_snippet = nested_html
            else:
                html_snippet = parser.summarize_details_page_as_valid_html(nested_html)
        elif catalog_item:
            html_snippet = catalog_item.html_snippet
        else:
            html_snippet = llm_html
    else:
        if low_budget_mode:
            html_snippet = llm_html
        else:
            html_snippet = parser.summarize_details_page_as_valid_html(llm_html)

    fields = parser.extract_fields(html_snippet)
    return ScraperConfig(
        start_url=start_url,
        page_type=page_type,
        pagination=pagination,
        catalog_item=catalog_item,
        open_nested_pages=open_nested_pages,
        fields=fields,
        max_pages=max_pages,
        max_rows=max_rows,
    )


def _run_scraper(config: ScraperConfig) -> list[dict]:
    crawler = RequestsCrawler()
    scraper = Scraper(config=config, crawler=crawler)
    return list(scraper.scrape())


def _parse_manual_fields(fields_text: str) -> list[tuple[str, str, str]]:
    fields = []
    for raw_line in fields_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            raise ValueError(f"Linea invalida: {line}")
        name = parts[0]
        xpath_expr = parts[1]
        mode = parts[2].lower() if len(parts) > 2 and parts[2] else "text"
        if mode not in ("text", "href", "src"):
            raise ValueError(f"Modo invalido '{mode}' en linea: {line}")
        fields.append((name, xpath_expr, mode))
    if not fields:
        raise ValueError("Debes definir al menos un campo.")
    return fields


def _value_from_xpath(node, xpath_expr: str, mode: str) -> str:
    if not xpath_expr or not xpath_expr.strip():
        return ""
    values = node.xpath(xpath_expr)
    if not isinstance(values, list):
        values = [values]
    if not values:
        return ""

    out = []
    for value in values:
        if isinstance(value, str):
            out.append(value.strip())
            continue
        if mode in ("href", "src"):
            out.append((value.get(mode) or "").strip())
        else:
            out.append(" ".join([t.strip() for t in value.itertext() if t.strip()]))
    out = [v for v in out if v]
    return " | ".join(out)


def _run_manual_scrape(
    start_url: str,
    card_xpath: str,
    fields_text: str,
    max_rows: int,
) -> list[dict]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://www.google.com/",
    }
    response = requests.get(start_url, headers=headers, timeout=30)
    if response.status_code == 403:
        # Retry once with a clean session; some sites block plain first-hit requests.
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(start_url, timeout=30)
    response.raise_for_status()
    tree = html.fromstring(response.text)
    fields = _parse_manual_fields(fields_text)

    nodes = tree.xpath(card_xpath) if card_xpath else [tree]
    rows = []
    for node in nodes:
        row = {}
        for name, xpath_expr, mode in fields:
            row[name] = _value_from_xpath(node, xpath_expr, mode)
        row = _fill_missing_from_detail(row, start_url)
        if any(v for v in row.values()):
            rows.append(row)
        if len(rows) >= max_rows:
            break
    return rows


def _first_non_empty_xpath(tree, xpaths: list[str]) -> str:
    for xpath_expr in xpaths:
        value = _value_from_xpath(tree, xpath_expr, "text")
        if value:
            return value
    return ""


def _fill_missing_from_detail(row: dict, start_url: str) -> dict:
    detail_url = row.get("url_detalle", "")
    if not detail_url:
        return row

    fields_to_fill = {
        "codigo",
        "url_whatsapp",
        "url_loja",
        "precio_usd",
        "precio_brl",
        "advertiser",
        "category",
        "product",
    }
    if all(row.get(field) for field in fields_to_fill):
        return row

    full_url = urljoin(start_url, detail_url.strip())
    row["url_detalle"] = full_url
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    try:
        response = requests.get(full_url, headers=headers, timeout=20)
        response.raise_for_status()
        tree = html.fromstring(response.text)
    except Exception:
        return row

    if not row.get("codigo"):
        row["codigo"] = _first_non_empty_xpath(
            tree,
            [
                "normalize-space(substring-after("
                "(.//*[contains(text(),'Código:')])[1], 'Código:'))",
                "normalize-space((.//*[contains(@class,'caracteristicas')])[1])",
            ],
        )
    if not row.get("url_whatsapp"):
        row["url_whatsapp"] = _first_non_empty_xpath(
            tree, [".//a[contains(@href,'api.whatsapp.com')]/@href"]
        )
    if not row.get("url_loja"):
        row["url_loja"] = _first_non_empty_xpath(
            tree,
            [
                ".//a[contains(@class,'btn-store-redirect')]/@href",
                ".//a[contains(@onclick,'external_website_advertiser')]/@href",
            ],
        )
    if not row.get("precio_usd"):
        row["precio_usd"] = _first_non_empty_xpath(
            tree, ["normalize-space((.//*[contains(text(),'US$')])[1])"]
        )
    if not row.get("precio_brl"):
        row["precio_brl"] = _first_non_empty_xpath(
            tree, ["normalize-space((.//*[contains(text(),'R$')])[1])"]
        )
    if not row.get("advertiser"):
        row["advertiser"] = _first_non_empty_xpath(
            tree,
            [
                "substring-before(substring-after("
                "(.//a[contains(@onclick,\"'advertiser'\")]/@onclick)[1], "
                "\"'advertiser': '\"), \"'\")",
            ],
        )
    if not row.get("category"):
        row["category"] = _first_non_empty_xpath(
            tree,
            [
                "substring-before(substring-after("
                "(.//a[contains(@onclick,\"'category'\")]/@onclick)[1], "
                "\"'category': '\"), \"'\")",
            ],
        )
    if not row.get("product"):
        row["product"] = _first_non_empty_xpath(
            tree,
            [
                "substring-before(substring-after("
                "(.//a[contains(@onclick,\"'product'\")]/@onclick)[1], "
                "\"'product': '\"), \"'\")",
            ],
        )

    # Retry once if page loaded but key fields are still empty.
    if not any([row.get("codigo"), row.get("advertiser"), row.get("product")]):
        try:
            response = requests.get(full_url, headers=headers, timeout=30)
            response.raise_for_status()
            tree = html.fromstring(response.text)
            if not row.get("codigo"):
                row["codigo"] = _first_non_empty_xpath(
                    tree,
                    [
                        "normalize-space(substring-after("
                        "(.//*[contains(text(),'Código:')])[1], 'Código:'))"
                    ],
                )
            if not row.get("advertiser"):
                row["advertiser"] = _first_non_empty_xpath(
                    tree,
                    [
                        "substring-before(substring-after("
                        "(.//a[contains(@onclick,\"'advertiser'\")]/@onclick)[1], "
                        "\"'advertiser': '\"), \"'\")",
                    ],
                )
            if not row.get("category"):
                row["category"] = _first_non_empty_xpath(
                    tree,
                    [
                        "substring-before(substring-after("
                        "(.//a[contains(@onclick,\"'category'\")]/@onclick)[1], "
                        "\"'category': '\"), \"'\")",
                    ],
                )
            if not row.get("product"):
                row["product"] = _first_non_empty_xpath(
                    tree,
                    [
                        "substring-before(substring-after("
                        "(.//a[contains(@onclick,\"'product'\")]/@onclick)[1], "
                        "\"'product': '\"), \"'\")",
                    ],
                )
        except Exception:
            pass
    return row


st.markdown(
    """
    <div class="hero">
        <h1>ScraperAI GVA</h1>
        <p>Pega una URL, pulsa un boton y listo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Inicio rapido")

mode = st.radio(
    "Modo",
    options=["Automatico (IA)", "Manual (sin IA)"],
    horizontal=True,
)

col1, col2 = st.columns([2.1, 1], gap="large")
with col1:
    start_url = st.text_input("URL", placeholder="https://www.comprasparaguai.com.br")
with col2:
    default_tokens = 512
    if mode == "Automatico (IA)":
        preset_name = st.selectbox("Preset", options=list(MODEL_PRESETS.keys()), index=0)
        preset = MODEL_PRESETS[preset_name]

        if preset is None:
            provider = st.selectbox(
                "Proveedor IA",
                options=["OpenAI", "DeepSeek", "OpenRouter"],
                index=0,
            )
            if provider == "OpenAI":
                default_model = "gpt-4o"
            elif provider == "DeepSeek":
                default_model = "deepseek-chat"
            else:
                default_model = "openai/gpt-5.2"
            default_tokens = 1200 if provider == "OpenRouter" else 2000
        else:
            provider = preset["provider"]
            default_model = preset["model"]
            default_tokens = preset["tokens"]
            st.caption(f"Proveedor: `{provider}`")

        default_key = os.getenv("OPENAI_API_KEY", "")
        if provider == "DeepSeek":
            default_key = os.getenv("DEEPSEEK_API_KEY", default_key)
        elif provider == "OpenRouter":
            default_key = os.getenv("OPENROUTER_API_KEY", default_key)

        model_name = st.text_input("Modelo", value=default_model)
        openai_api_key = st.text_input(
            "API_KEY",
            value=default_key,
            type="password",
        )
    else:
        provider = "OpenAI"
        model_name = "gpt-4o"
        openai_api_key = ""
        st.text_input("API_KEY", value="No requerido en modo manual", disabled=True)

with st.expander("Opciones avanzadas", expanded=False):
    if mode == "Automatico (IA)":
        a, b, c = st.columns(3)
        with a:
            max_pages = int(st.number_input("Max pages", min_value=1, value=1, step=1))
        with b:
            max_rows = int(st.number_input("Max rows", min_value=1, value=20, step=1))
        with c:
            max_output_tokens = int(
                st.number_input("Max output tokens", min_value=128, value=default_tokens, step=128)
            )
        low_budget_mode = st.toggle("Modo ahorro", value=(provider == "OpenRouter"))
        open_nested_pages = st.toggle("Abrir paginas detalle", value=False)
        if low_budget_mode:
            max_pages = 1
            max_output_tokens = min(max_output_tokens, 256)
            open_nested_pages = False
            st.caption(
                "Modo ahorro activo: max_pages=1, "
                "max_output_tokens<=256, sin paginas detalle."
            )
        if provider == "OpenRouter":
            site_url = st.text_input(
                "HTTP-Referer (opcional)",
                value=os.getenv("OPENROUTER_SITE_URL", ""),
                placeholder="https://tu-sitio.com",
            )
            site_name = st.text_input(
                "X-Title (opcional)",
                value=os.getenv("OPENROUTER_SITE_NAME", ""),
                placeholder="Mi Scraper",
            )
            st.info(
                "OpenRouter usa modo bajo consumo: HTML recortado y menos "
                "detecciones para evitar errores 402 por tokens."
            )
        else:
            site_url = ""
            site_name = ""
        st.markdown(
            '<p class="hint">Modo requests: no soporta paginacion xpath/scroll.</p>',
            unsafe_allow_html=True,
        )
        card_xpath = ""
        fields_text = ""
    else:
        max_pages = 1
        max_rows = int(st.number_input("Max rows", min_value=1, value=20, step=1))
        max_output_tokens = 512
        open_nested_pages = False
        low_budget_mode = False
        site_url = ""
        site_name = ""
        card_xpath = st.text_input(
            "XPath de items (opcional)",
            value=MANUAL_DEFAULT_CARD_XPATH,
            placeholder="//div[contains(@class,'product')]",
            help="Si lo dejas vacio, intenta extraer una sola fila sobre todo el HTML.",
        )
        fields_text = st.text_area(
            "Campos (uno por linea: nombre|xpath|modo)",
            height=150,
            value=MANUAL_DEFAULT_FIELDS,
        )
        st.markdown(
            "<p class=\"hint\">Ejemplo: nombre|.//h2/text()|text. "
            "Modo puede ser text, href o src.</p>",
            unsafe_allow_html=True,
        )

go = st.button("Empezar scraping", type="primary", use_container_width=True)

if go:
    if not start_url:
        st.error("Ingresa una URL.")
    elif mode == "Automatico (IA)" and not openai_api_key:
        st.error("Ingresa tu API_KEY.")
    else:
        if mode == "Automatico (IA)":
            config = None
            rows = None

            with st.spinner("1/2 Detectando configuracion..."):
                try:
                    config = _build_config_with_ai(
                        start_url=start_url,
                        api_key=openai_api_key,
                        provider=provider,
                        model_name=model_name,
                        site_url=site_url,
                        site_name=site_name,
                        max_output_tokens=max_output_tokens,
                        max_pages=max_pages,
                        max_rows=max_rows,
                        open_nested_pages=open_nested_pages,
                        low_budget_mode=low_budget_mode,
                    )
                except Exception as e:
                    st.error(f"No se pudo detectar configuracion: {e}")

            if config is not None:
                with st.spinner("2/2 Extrayendo datos..."):
                    try:
                        rows = _run_scraper(config)
                    except Exception as e:
                        st.error(f"Error durante scraping: {e}")

            if config is not None:
                st.success("Configuracion detectada.")
                cfg_json = _config_to_pretty_json(config)
                st.download_button(
                    "Descargar configuracion",
                    data=cfg_json.encode("utf-8"),
                    file_name="generated.scraperai.json",
                    mime="application/json",
                    use_container_width=True,
                )
                with st.expander("Ver configuracion JSON"):
                    st.code(cfg_json, language="json")
        else:
            rows = None
            with st.spinner("Extrayendo datos en modo manual..."):
                try:
                    rows = _run_manual_scrape(
                        start_url=start_url,
                        card_xpath=card_xpath,
                        fields_text=fields_text,
                        max_rows=max_rows,
                    )
                except Exception as e:
                    st.error(f"Error en modo manual: {e}")

        if rows is not None:
            df = _normalize_rows(rows)
            m1, m2 = st.columns(2)
            m1.metric("Filas extraidas", len(rows))
            m2.metric("Columnas", len(df.columns))
            if len(rows) == 0:
                st.warning(
                    "No se extrajeron filas. "
                    "Prueba Modo Manual (sin IA) con XPaths."
                )
            st.dataframe(df, use_container_width=True, height=480)
            st.download_button(
                "Descargar resultados JSON",
                data=json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="results.json",
                mime="application/json",
                use_container_width=True,
            )
            st.download_button(
                "Descargar resultados CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if not df.empty:
                try:
                    s3_key, s3_url = _upload_csv_to_s3(df)
                    if s3_key and s3_url:
                        st.success(f"CSV subido a S3: {s3_key}")
                        st.link_button(
                            "Descargar CSV desde S3 (link temporal 1h)",
                            s3_url,
                            use_container_width=True,
                        )
                except Exception as e:
                    st.warning(f"No se pudo subir CSV a S3: {e}")
