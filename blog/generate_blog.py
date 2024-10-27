import os
import json
import markdown
from bs4 import BeautifulSoup

from jinja2 import Environment, FileSystemLoader
from typing import List, Tuple


def generate_toc(html_content: str) -> Tuple[List[Tuple[int, str]], str]:
    """Generate table of contents from HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    toc = []
    
    for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        level = int(header.name[1])
        title = header.get_text()
        # Create ID for the header
        header_id = title.lower().replace(' ', '-')
        header['id'] = header_id
        toc.append((level, title))
    
    # Convert back to string after modifications
    return toc, str(soup)


def process_blog_posts() -> None:
    """Process blog posts, generate HTML files, and create blog list."""
    posts = []
    post_slugs = []
    env = Environment(loader=FileSystemLoader("blog/templates"))
    post_template = env.get_template("blog_post.html")

    # Get and sort directories by date (newest first)
    dirs = [d for d in os.listdir("blog/posts") if os.path.isdir(os.path.join("blog/posts", d))]
    dirs.sort(reverse=True)

    # Configure markdown with extra extensions for lists
    md = markdown.Markdown(extensions=['fenced_code', 'nl2br', 'sane_lists'])
    
    for post_dir in dirs:
        post_slugs.append(post_dir)
        with open(f"blog/posts/{post_dir}/content.md", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Fix list formatting by ensuring proper line breaks and indentation
        lines = content.split('\n')
        processed_lines = []
        for line in lines:
            # Count leading spaces to preserve indentation
            leading_spaces = len(line) - len(line.lstrip())
            if line.lstrip().startswith(('* ', '- ', '1. ')):
                if leading_spaces > 0:
                    # Preserve indentation for nested items
                    processed_lines.append(' ' * leading_spaces + line.lstrip())
                else:
                    # Add extra line break before top-level items
                    processed_lines.append('\n' + line)
            else:
                processed_lines.append(line)
        
        content = '\n'.join(processed_lines)
        
        with open(f"blog/posts/{post_dir}/metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        html_content = md.convert(content)
        
        # Process code blocks to add language class
        soup = BeautifulSoup(html_content, "html.parser")
        for pre in soup.find_all('pre'):
            if pre.code and 'language-' in pre.code.get('class', []):
                language = pre.code['class'][0].replace('language-', '')
                pre['data-language'] = language
                # Add Prism.js classes
                pre['class'] = pre.get('class', []) + ['line-numbers', 'toolbar-top']
                pre.code['class'] = ['language-' + language]
        
        html_content = str(soup)
        toc, html_content = generate_toc(html_content)

        # Generate individual blog post HTML
        post_html = post_template.render(content=html_content, toc=toc, **metadata)
        with open(f"blog/output/{post_dir}/content.html", "w", encoding="utf-8") as f:
            f.write(post_html)

        posts.append({**metadata, "url": f"{post_dir}/content.html"})

    # Sort posts by date from metadata (newest first)
    posts.sort(key=lambda x: x['date'], reverse=True)
    
    # Generate blog list HTML
    blog_list_template = env.get_template("blog_list.html")
    blog_list_html = blog_list_template.render(posts=posts)
    with open("blog/output/blog.html", "w", encoding="utf-8") as f:
        f.write(blog_list_html)

    # Generate index.json
    with open("blog/posts/index.json", "w", encoding="utf-8") as f:
        json.dump(post_slugs, f, indent=2)


if __name__ == "__main__":
    process_blog_posts()
