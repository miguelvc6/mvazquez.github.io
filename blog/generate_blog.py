import json
from pathlib import Path
from typing import List, Tuple

import markdown
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader


def generate_toc(html_content: str) -> Tuple[List[Tuple[int, str]], str]:
    """Generate table of contents from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    toc = []

    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(header.name[1])
        title = header.get_text()
        # Create ID for the header
        header_id = title.lower().replace(" ", "-")
        header["id"] = header_id
        toc.append((level, title))

    return toc, str(soup)


def calculate_reading_time(text: str) -> int:
    """Calculate reading time of a blog post."""
    words = len(text.split())
    average_speed = 200  # words per minute
    return round(words / average_speed)


def fix_list_formatting(content: str) -> str:
    """Fix list formatting in the markdown content."""
    lines = content.split("\n")
    processed_lines = []
    for line in lines:
        # Count leading spaces to preserve indentation
        leading_spaces = len(line) - len(line.lstrip())
        if line.lstrip().startswith(("* ", "- ", "1. ")):
            if leading_spaces > 0:
                # Preserve indentation for nested items
                processed_lines.append(" " * leading_spaces + line.lstrip())
            else:
                # Add extra line break before top-level items
                processed_lines.append("\n" + line)
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)


def process_code_blocks(html_content: str) -> str:
    """Process code blocks in HTML content to add language classes."""
    soup = BeautifulSoup(html_content, "html.parser")
    for pre in soup.find_all("pre"):
        code_tag = pre.find("code")
        if code_tag and code_tag.has_attr("class"):
            classes = code_tag["class"]
            language_classes = [cls for cls in classes if cls.startswith("language-")]
            if language_classes:
                language = language_classes[0].replace("language-", "")
                pre["data-language"] = language
                # Add Prism.js classes
                pre["class"] = pre.get("class", []) + ["line-numbers", "toolbar-top"]
                code_tag["class"] = ["language-" + language]
    return str(soup)


def process_blog_post(post_dir: Path, md: markdown.Markdown, post_template) -> dict:
    """Process a single blog post and generate HTML content."""
    content_path = post_dir / "content.md"
    metadata_path = post_dir / "metadata.json"

    with content_path.open("r", encoding="utf-8") as f:
        content = f.read()

    estimated_reading_time = calculate_reading_time(content)
    content = fix_list_formatting(content)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata["estimated_reading_time"] = estimated_reading_time
    html_content = md.convert(content)
    html_content = process_code_blocks(html_content)
    toc, html_content = generate_toc(html_content)

    # Generate individual blog post HTML
    output_dir = Path("blog/output") / post_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    post_html = post_template.render(content=html_content, toc=toc, **metadata)
    output_path = output_dir / "content.html"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(post_html)

    metadata["url"] = f"{post_dir.name}/content.html"
    return metadata


def process_blog_posts() -> None:
    """Process blog posts, generate HTML files, and create blog list."""
    posts = []
    post_slugs = []
    env = Environment(loader=FileSystemLoader("blog/templates"))
    post_template = env.get_template("blog_post.html")

    posts_dir = Path("blog/posts")
    output_dir = Path("blog/output")
    output_dir.mkdir(exist_ok=True)

    # Get and sort directories by date (newest first)
    dirs = sorted(
        [d for d in posts_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )

    # Configure markdown with extra extensions for lists
    md = markdown.Markdown(extensions=["fenced_code", "nl2br", "sane_lists"])

    for post_dir in dirs:
        post_slugs.append(post_dir.name)
        metadata = process_blog_post(post_dir, md, post_template)
        posts.append(metadata)

    # Sort posts by date
    posts.sort(key=lambda x: x["date"], reverse=True)

    # Generate blog list HTML
    blog_list_template = env.get_template("blog_list.html")
    blog_list_html = blog_list_template.render(posts=posts)
    with (output_dir / "blog.html").open("w", encoding="utf-8") as f:
        f.write(blog_list_html)

    # Generate index.json
    with (posts_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(post_slugs, f, indent=2)


if __name__ == "__main__":
    process_blog_posts()
