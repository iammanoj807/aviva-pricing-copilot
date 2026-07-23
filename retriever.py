import glob
import os
from abc import ABC, abstractmethod

import pandas as pd

from schemas import Passage

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class Retriever(ABC):
    """Base class for retrieving unstructured evidence by segment.

    `query` param is included for future semantic implementations.
    """

    @abstractmethod
    def retrieve(self, segment: str, query: str) -> list[Passage]:
        ...


class TagFilterRetriever(Retriever):
    """Simple segment-tag filtering. Good enough at this data size."""

    def __init__(self, data_dir: str = DATA_DIR):
        self._market_docs = self._load_market_docs(data_dir)
        self._verbatims = self._load_verbatims(data_dir)

    def _load_market_docs(self, data_dir: str) -> list[Passage]:
        """Read .md files, extract segment from frontmatter."""
        docs = []
        for path in sorted(glob.glob(os.path.join(data_dir, "market_intel", "*.md"))):
            with open(path) as f:
                text = f.read()
            segment = self._parse_frontmatter_segment(text)
            docs.append(Passage(
                source=os.path.basename(path),
                segment=segment,
                kind="market_doc",
                text=text,
            ))
        return docs

    def _load_verbatims(self, data_dir: str) -> list[Passage]:
        """One Passage per customer verbatim, tagged with its segment column."""
        df = pd.read_csv(os.path.join(data_dir, "customer_feedback.csv"))
        return [
            Passage(
                source=row["feedback_id"],
                segment=row["segment"],
                kind="verbatim",
                text=f"[{row['channel']}] {row['verbatim']}",
            )
            for _, row in df.iterrows()
        ]

    @staticmethod
    def _parse_frontmatter_segment(text: str) -> str:
        """Grab the segment tag from YAML frontmatter. Defaults to 'all'."""
        for line in text.splitlines():
            if line.strip().startswith("segment:"):
                return line.split(":", 1)[1].strip()
        return "all"

    def retrieve(self, segment: str, query: str) -> list[Passage]:
        """Return every passage tagged for this segment (plus 'all' for cross-cutting docs).

        `query` is ignored here since there's nothing to rank at four docs.
        """
        return [
            p for p in (self._market_docs + self._verbatims)
            if p.segment == segment or p.segment == "all"
        ]
