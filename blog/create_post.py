import argparse
import json
from datetime import datetime
from pathlib import Path


def create_post(title: str) -> None:
    """Create a new blog post scaffold with the given title."""
    # Create slug from title
    slug = title.lower().replace(" ", "-")

    # Create date-based directory name
    today = datetime.now()
    dir_name = f"{today.strftime('%Y-%m-%d')}-{slug}"

    # Setup paths
    posts_dir = Path("blog/posts")
    post_dir = posts_dir / dir_name
    media_dir = Path("blog/media") / dir_name

    # Create directories
    post_dir.mkdir(parents=True, exist_ok=True)
    media_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata.json
    metadata = {
        "title": title,
        "author": "Miguel Vázquez",
        "date": today.strftime("%Y-%m-%d"),
        "tags": [],
        "description": "",
    }

    with open(post_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Create content.md with template
    markdown_template = f"""# {title}

## Introduction

Write your introduction here.

## Section 1

Write your content here.

<!-- To add images, place them in the media folder and reference them like this: -->
<!-- ![Alt text](../../../media/{dir_name}/image.png) -->

## Section 2

Write more content here.

## Conclusion

Write your conclusion here.
"""

    with open(post_dir / "content.md", "w", encoding="utf-8") as f:
        f.write(markdown_template)

    print("\nBlog post scaffold created successfully!")
    print(f"Post directory: {post_dir}")
    print(f"Media directory: {media_dir}")
    print("\nNext steps:")
    print("1. Add your content to content.md")
    print("2. Update metadata in metadata.json")
    print("3. Add images to the media directory")
    print("4. Run the blog generator to build your post")


def main():
    parser = argparse.ArgumentParser(
        description="Create a new blog post scaffold"
    )
    parser.add_argument("title", help="Title of the blog post")
    args = parser.parse_args()

    create_post(args.title)


if __name__ == "__main__":
    main()

# Example: python blog/create_post.py "My New Blog Post"

# generates the following:

# blog/
# ├── posts/
# │   └── 2024-03-14-my-new-blog-post/
# │       ├── content.md
# │       └── metadata.json
# └── media/
#     └── 2024-03-14-my-new-blog-post/