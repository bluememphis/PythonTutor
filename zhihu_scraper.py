"""CLI utility for downloading Zhihu answers for a question.

The script fetches the most upvoted answers to a Zhihu question and stores
structured data in a CSV file.  Each row of the CSV contains the author name,
upvote count, favourite count, and the textual content of the answer.  Long
answers are automatically split across multiple CSV cells so that spreadsheet
software can ingest them safely.

Example:
    python zhihu_scraper.py --question 1904446392564430091 --limit 500 \
        --output answers.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Iterable, Iterator, List
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen


API_ROOT = "https://www.zhihu.com/api/v4/questions/"
DEFAULT_PAGE_SIZE = 20
DEFAULT_CHUNK_SIZE = 30_000


@dataclass
class AnswerRecord:
    """Container for the fields that we need to persist."""

    author_name: str
    voteup_count: int
    favlists_count: int
    content_text: str


class TextOnlyHTMLParser(HTMLParser):
    """Extract plain text from Zhihu's answer HTML.

    The parser removes scripts, styles, images, videos and other non-textual
    content.  Block-level elements are converted into newlines so that the
    resulting text remains readable.
    """

    #: tags whose entire contents should be ignored
    _IGNORED_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "video",
        "audio",
        "figure",
        "img",
        "iframe",
    }

    #: block-level tags that should yield a newline when they start or end
    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "br",
        "li",
        "ul",
        "ol",
        "blockquote",
        "pre",
        "code",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "table",
        "tr",
        "td",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: List[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in self._IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if self._ignored_depth and tag in self._IGNORED_TAGS:
            self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._ignored_depth:
            return
        if data.strip():
            self._parts.append(unescape(data))

    def handle_entityref(self, name: str) -> None:  # type: ignore[override]
        if self._ignored_depth:
            return
        self._parts.append(unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:  # type: ignore[override]
        if self._ignored_depth:
            return
        self._parts.append(unescape(f"&#{name};"))

    def get_text(self) -> str:
        text = "".join(self._parts)
        lines = [line.strip() for line in text.splitlines()]
        # Remove duplicate blank lines but keep intentional spacing
        filtered_lines: List[str] = []
        previous_blank = False
        for line in lines:
            is_blank = not line
            if is_blank and previous_blank:
                continue
            filtered_lines.append(line)
            previous_blank = is_blank
        return "\n".join(filtered_lines).strip()


def parse_answer_text(html_content: str) -> str:
    parser = TextOnlyHTMLParser()
    parser.feed(html_content)
    parser.close()
    return parser.get_text()


def chunk_text(text: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not text:
        return [""]
    chunks: List[str] = []
    start = 0
    length = len(text)
    while start < length:
        chunks.append(text[start : start + chunk_size])
        start += chunk_size
    return chunks


def build_request(url: str) -> Request:
    headers = {
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        ),
        "accept": "application/json",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "connection": "keep-alive",
    }
    return Request(url, headers=headers)


def fetch_answers(question_id: str, limit: int) -> Iterator[AnswerRecord]:
    collected = 0
    offset = 0
    per_page = DEFAULT_PAGE_SIZE

    include = (
        "data[*].is_normal,admin_closed_comment,comment_count,content," \
        "editable_content,voteup_count,reshipment_settings,comment_permission," \
        "created_time,updated_time,review_info,question,excerpt,relationship," \
        "is_labeled,is_recognized,favlists_count;data[*].author.follower_count," \
        "badge[*].topics"
    )

    while collected < limit:
        current_limit = min(per_page, limit - collected)
        query = urlencode(
            {
                "include": include,
                "limit": current_limit,
                "offset": offset,
                "platform": "desktop",
                "sort_by": "default",
            }
        )
        url = urljoin(API_ROOT, f"{question_id}/answers?{query}")
        req = build_request(url)
        try:
            with urlopen(req, timeout=15) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} when fetching {url}")
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network failure handling
            raise RuntimeError(f"Failed to fetch answers: {exc}") from exc

        answers = payload.get("data", [])
        if not answers:
            break

        for answer in answers:
            author = answer.get("author") or {}
            author_name = author.get("name") or "匿名用户"
            voteup_count = int(answer.get("voteup_count", 0))
            favlists_count = int(answer.get("favlists_count", 0))
            raw_content = answer.get("content") or ""
            parsed_content = parse_answer_text(raw_content)
            yield AnswerRecord(
                author_name=author_name,
                voteup_count=voteup_count,
                favlists_count=favlists_count,
                content_text=parsed_content,
            )
            collected += 1
            if collected >= limit:
                break

        paging = payload.get("paging", {})
        if paging.get("is_end"):
            break
        offset += current_limit
        # Friendly delay to avoid hitting rate limits when running locally
        time.sleep(0.3)


def write_answers_to_csv(
    records: Iterable[AnswerRecord], output_path: str, chunk_size: int
) -> None:
    with open(output_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        for record in records:
            content_chunks = chunk_text(record.content_text, chunk_size)
            row = [
                record.author_name,
                record.voteup_count,
                record.favlists_count,
                *content_chunks,
            ]
            writer.writerow(row)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        help="The numeric identifier of the Zhihu question to scrape.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of answers to fetch (default: 500).",
    )
    parser.add_argument(
        "--output",
        default="zhihu_answers.csv",
        help="Destination CSV file (default: zhihu_answers.csv).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=(
            "Maximum number of characters stored in a single CSV cell before "
            "the text is split into a new cell (default: 30000)."
        ),
    )
    args = parser.parse_args(argv)

    if not args.question:
        # When the script is executed from an interactive environment (such as
        # Spyder's "Run file" button) no command-line arguments are supplied.
        # In that situation we offer a friendly prompt instead of raising an
        # argparse error straight away.  When stdin is not interactive we keep
        # the original behaviour so that command line usage still fails fast.
        if sys.stdin.isatty():
            prompt = "Enter the Zhihu question ID to scrape: "
            args.question = input(prompt).strip()
        if not args.question:
            parser.error("the --question argument is required")

    return args


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        records = fetch_answers(args.question, args.limit)
        write_answers_to_csv(records, args.output, args.chunk_size)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
