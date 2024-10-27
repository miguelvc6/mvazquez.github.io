import os
import json
import markdown
from jinja2 import Environment, FileSystemLoader
from typing import List, Tuple


def generate_toc(html_content: str) -> List[Tuple[int, str]]:
    """Generate table of contents from HTML content."""
    toc = []
    for line in html_content.split("\n"):
        if line.startswith("<h"):
            level = int(line[2])
            title = line.split(">", 1)[1].split("<")[0]
            toc.append((level, title))
    return toc


def process_blog_posts() -> None:
    """Process blog posts, generate HTML files, and create blog list."""
    posts = []
    env = Environment(loader=FileSystemLoader("templates"))
    post_template = env.get_template("blog_post.html")

    for post_dir in os.listdir("posts"):
        with open(f"posts/{post_dir}/content.md", "r") as f:
            content = f.read()
        with open(f"posts/{post_dir}/metadata.json", "r") as f:
            metadata = json.load(f)

        html_content = markdown.markdown(content)
        toc = generate_toc(html_content)

        # Generate individual blog post HTML
        post_html = post_template.render(content=html_content, toc=toc, **metadata)
        with open(f"output/posts/{post_dir}.html", "w") as f:
            f.write(post_html)

        posts.append({**metadata, "url": f"posts/{post_dir}.html"})

    # Generate blog list HTML
    blog_list_template = env.get_template("blog_list.html")
    blog_list_html = blog_list_template.render(posts=posts)
    with open("output/blog.html", "w") as f:
        f.write(blog_list_html)


if __name__ == "__main__":
    process_blog_posts()
