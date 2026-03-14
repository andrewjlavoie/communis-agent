import yaml


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter. Returns (metadata, content)."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---\n", 3)
    if end == -1:
        return {}, text
    frontmatter = text[4:end]
    content = text[end + 5:]
    try:
        meta = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, content
