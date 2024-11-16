import json
from pathlib import Path
from typing import List, Tuple

import markdown
from bs4 import BeautifulSoup
from jinja2 import Environment, FileSystemLoader
from PIL import Image


def generate_toc(html_content: str) -> Tuple[List[Tuple[int, str]], str]:
    """Generate table of contents from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    toc = []

    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(header.name[1])
        title = header.get_text()
        header_id = title.lower().replace(" ", "-")
        header["id"] = header_id
        toc.append((level, title))

    # Adjust relative levels starting from 1
    if toc:
        min_level = min(level for level, _ in toc)
        sorted_toc = [(level - min_level + 1, title) for level, title in toc]
    else:
        sorted_toc = []

    return sorted_toc, str(soup)


def calculate_reading_time(text: str) -> int:
    """Calculate reading time of a blog post."""
    words = len(text.split())
    average_speed = 200  # words per minute
    return max(1, round(words / average_speed))


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
            language_classes = [
                cls for cls in classes if cls.startswith("language-")
            ]
            if language_classes:
                language = language_classes[0].replace("language-", "")
                pre["data-language"] = language
                # Add Prism.js classes
                pre["class"] = pre.get("class", []) + [
                    "line-numbers",
                    "toolbar-top",
                ]
                code_tag["class"] = ["language-" + language]
    return str(soup)


def process_latex(content: str) -> Tuple[str, list, list]:
    """Pre-process markdown content to protect LaTeX expressions."""
    # Store LaTeX expressions temporarily
    latex_blocks = []
    latex_inline = []

    # Handle block LaTeX
    parts = content.split("\\[")
    for i in range(1, len(parts)):
        if "\\]" in parts[i]:
            math, rest = parts[i].split("\\]", 1)
            placeholder = f"LATEX_BLOCK_{len(latex_blocks)}"
            latex_blocks.append(math)
            parts[i] = placeholder + rest
    content = "\\[".join(parts)

    # Handle inline LaTeX
    parts = content.split("$$")
    for i in range(1, len(parts), 2):
        placeholder = f"LATEX_INLINE_{len(latex_inline)}"
        latex_inline.append(parts[i])
        parts[i] = placeholder
    content = "".join(parts)

    return content, latex_blocks, latex_inline


def restore_latex(
    html_content: str, latex_blocks: list, latex_inline: list
) -> str:
    """Restore LaTeX expressions in the HTML content."""
    # Restore block LaTeX
    for i, math in enumerate(latex_blocks):
        html_content = html_content.replace(
            f"LATEX_BLOCK_{i}", f"\\[{math}\\]"
        )

    # Restore inline LaTeX
    for i, math in enumerate(latex_inline):
        html_content = html_content.replace(f"LATEX_INLINE_{i}", f"$${math}$$")

    return html_content


def optimize_image(image_path: Path, max_width: int = 1920) -> None:
    """
    Optimize image by:
    - Resizing if wider than max_width
    - Converting to WebP format
    - Applying compression
    """
    try:
        with Image.open(image_path) as img:
            # Skip if image is already optimized (WebP and correct size)
            if image_path.suffix.lower() == ".webp" and img.width <= max_width:
                return

            # Calculate new dimensions if needed
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to WebP and save
            webp_path = image_path.with_suffix(".webp")
            img.save(webp_path, "WEBP", quality=85, method=6, lossless=False)

            # If conversion successful, delete original file
            if webp_path.exists() and webp_path != image_path:
                image_path.unlink()

    except Exception as e:
        print(f"Error optimizing {image_path}: {e}")


def process_blog_post(post_dir: Path, post_template, prev_post=None, next_post=None) -> dict:
    """Process a single blog post and generate HTML content."""
    content_path = post_dir / "content.md"
    metadata_path = post_dir / "metadata.json"

    # Create media directory if it doesn't exist
    media_dir = Path("blog/media") / post_dir.name
    media_dir.mkdir(parents=True, exist_ok=True)

    # Optimize images in media directory
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    for image_path in media_dir.glob("*"):
        if image_path.suffix.lower() in image_extensions:
            optimize_image(image_path)

    with content_path.open("r", encoding="utf-8") as f:
        content = f.read()

    # Calculate reading time before fixing list formatting
    estimated_reading_time = calculate_reading_time(content)
    content = fix_list_formatting(content)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Get title and estimated_reading_time into metadata
    first_line = content.split("\n", 1)[0]
    if first_line.startswith("#"):
        metadata["title"] = first_line.lstrip("#").strip()
        content = "\n".join(content.split("\n", 1)[1:])
    metadata["estimated_reading_time"] = estimated_reading_time

    # Extract date from the folder name
    folder_date = post_dir.name.split("-", 3)[:3]
    folder_date_str = "-".join(folder_date)

    # Ensure the date in metadata matches the folder date
    metadata["date"] = folder_date_str

    # Save updated metadata to JSON file
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    # Pre-process content to protect LaTeX
    content, latex_blocks, latex_inline = process_latex(content)

    # Create a new Markdown instance for each post
    md = markdown.Markdown(
        extensions=["fenced_code", "nl2br", "sane_lists", "footnotes"]
    )

    # Convert to HTML
    html_content = md.convert(content)

    # Restore LaTeX expressions
    html_content = restore_latex(html_content, latex_blocks, latex_inline)

    # Process code blocks
    html_content = process_code_blocks(html_content)

    # Generate Table of Contents
    toc, html_content = generate_toc(html_content)

    # Update image paths in HTML content
    soup = BeautifulSoup(html_content, "html.parser")
    for img in soup.find_all("img"):
        src = img.get("src")
        if src and not src.startswith(("http://", "https://", "/")):
            # Convert image path to WebP
            src_path = Path(src)
            webp_src = src_path.with_suffix(".webp")
            img["src"] = webp_src
            # Add loading="lazy" for better performance
            img["loading"] = "lazy"

    html_content = str(soup)

    # Add the base URL for social sharing
    base_url = "https://mvazquez.ai/blog/output"  # Replace with your actual domain
    post_url = f"{base_url}/{post_dir.name}/content.html"
    
    post_html = post_template.render(
        content=html_content,
        toc=toc,
        request={'url': post_url},
        prev_post=prev_post,
        next_post=next_post,
        **metadata
    )

    # Generate individual blog post HTML
    output_dir = Path("blog/output") / post_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "content.html"
    with output_path.open("w", encoding="utf-8") as f:
        f.write(post_html)

    metadata["url"] = f"{post_dir.name}/content.html"
    return metadata


def process_blog_posts() -> None:
    """Process blog posts, generate HTML files, and create blog list."""
    posts = []
    post_slugs = []
    search_index = []
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

    # First pass: collect all metadata
    for post_dir in dirs:
        metadata_path = post_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        metadata["url"] = f"{post_dir.name}/content.html"
        posts.append(metadata)

    # Sort posts by date
    posts.sort(key=lambda x: x["date"], reverse=True)
    
    # Second pass: process posts with prev/next navigation
    for i, post_dir in enumerate(dirs):
        prev_post = posts[i-1] if i > 0 else None
        next_post = posts[i+1] if i < len(posts)-1 else None
        metadata = process_blog_post(post_dir, post_template, prev_post, next_post)
        
        # Skip draft posts in the main blog list and search index
        if not metadata.get("draft", False):
            post_slugs.append(post_dir.name)
            
            # Add to search index
            with (post_dir / "content.md").open("r", encoding="utf-8") as f:
                content = f.read()
            
            search_index.append({
                "title": metadata["title"],
                "date": metadata["date"],
                "url": metadata["url"],
                "content": ' '.join(content.split())[:200],
                "author": metadata.get("author", ""),
            })

    # Generate blog list HTML (excluding drafts)
    blog_list_template = env.get_template("blog_list.html")
    blog_list_html = blog_list_template.render(posts=[p for p in posts if not p.get("draft", False)])
    with (output_dir / "blog.html").open("w", encoding="utf-8") as f:
        f.write(blog_list_html)

    # Generate index.json
    with (posts_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(post_slugs, f, indent=2)

    # Generate search index JSON
    with (Path("blog/output") / "search-index.json").open("w", encoding="utf-8") as f:
        json.dump(search_index, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    process_blog_posts()
