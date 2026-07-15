from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)

    def should_use_zotero_corpus(self) -> bool:
        return self.config.executor.reranker != "none"

    def _zotero_corpus_cache_path(self) -> Path | None:
        cache_path = self.config.zotero.get("corpus_cache_path", None)
        if cache_path in (None, ""):
            return None
        return Path(str(cache_path))

    @staticmethod
    def _parse_cached_datetime(value: str) -> datetime:
        if value.endswith("Z"):
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return datetime.fromisoformat(value)

    @staticmethod
    def _format_cached_datetime(value: datetime) -> str:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")

    def load_zotero_corpus_cache(self, cache_path: Path) -> list[CorpusPaper] | None:
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            papers = data["papers"] if isinstance(data, dict) else data
            corpus = [
                CorpusPaper(
                    title=str(paper["title"]),
                    abstract=str(paper["abstract"]),
                    added_date=self._parse_cached_datetime(str(paper["added_date"])),
                    paths=[str(path) for path in paper.get("paths", [])],
                )
                for paper in papers
                if paper.get("abstract")
            ]
        except Exception as exc:
            logger.warning(f"Failed to load Zotero corpus cache from {cache_path}: {exc}")
            return None
        logger.info(f"Loaded {len(corpus)} zotero papers from cache {cache_path}")
        return corpus

    def save_zotero_corpus_cache(self, cache_path: Path, corpus: list[CorpusPaper]) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "papers": [
                {
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "added_date": self._format_cached_datetime(paper.added_date),
                    "paths": paper.paths,
                }
                for paper in corpus
            ],
        }
        cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logger.info(f"Saved {len(corpus)} zotero papers to cache {cache_path}")

    def fetch_zotero_corpus_from_api(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        cache_path = self._zotero_corpus_cache_path()
        refresh_cache = self.config.zotero.get("refresh_corpus_cache", False)
        if cache_path is not None and not refresh_cache:
            cached_corpus = self.load_zotero_corpus_cache(cache_path)
            if cached_corpus is not None:
                return cached_corpus

        corpus = self.fetch_zotero_corpus_from_api()
        if cache_path is not None:
            self.save_zotero_corpus_cache(cache_path, corpus)
        return corpus
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    
    def run(self):
        corpus = []
        if self.should_use_zotero_corpus():
            corpus = self.fetch_zotero_corpus()
            corpus = self.filter_corpus(corpus)
            if len(corpus) == 0:
                logger.error(f"No zotero papers found. Please check your zotero settings:\n{self.config.zotero}")
                return
        else:
            logger.info("Reranker is 'none'; skipping Zotero corpus loading and similarity scoring.")
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Selecting papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
