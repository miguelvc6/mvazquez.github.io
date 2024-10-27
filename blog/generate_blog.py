import os
import json
import markdown
from bs4 import BeautifulSoup

from jinja2 import Environment, FileSystemLoader
from typing import List, Tuple


def generate_toc(html_content: str) -> List[Tuple[int, str]]:
    """Generate table of contents from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    toc = []
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(header.name[1])
        title = header.get_text()
        toc.append((level, title))
    return toc


def process_blog_posts() -> None:
    """Process blog posts, generate HTML files, and create blog list."""
    posts = []
    env = Environment(loader=FileSystemLoader("blog/templates"))
    post_template = env.get_template("blog_post.html")

    for post_dir in os.listdir("blog/posts"):
        with open(f"blog/posts/{post_dir}/content.md", "r") as f:
            content = f.read()
        with open(f"blog/posts/{post_dir}/metadata.json", "r") as f:
            metadata = json.load(f)

        html_content = markdown.markdown(content)
        # Export raw HTML content
        # Create output directory for post
        # os.makedirs(f"blog/output/{post_dir}", exist_ok=True)

        # with open(f"blog/output/{post_dir}/content.html", "w") as f:
        #     f.write(html_content)

        toc = generate_toc(html_content)

        # Generate individual blog post HTML
        post_html = post_template.render(content=html_content, toc=toc, **metadata)
        with open(f"blog/output/{post_dir}/content.html", "w", encoding="utf-8") as f:
            f.write(post_html)

        posts.append({**metadata, "url": f"{post_dir}/content.html"})

    # Generate blog list HTML
    blog_list_template = env.get_template("blog_list.html")
    blog_list_html = blog_list_template.render(posts=posts)
    with open("blog/output/blog.html", "w", encoding="utf-8") as f:
        f.write(blog_list_html)


if __name__ == "__main__":
    process_blog_posts()
