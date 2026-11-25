import katex from "katex";

function visit(node, callback) {
  if (node.type === 'element') callback(node);
  for (const child of node.children ?? []) visit(child, callback);
}

function textOf(node) {
  if (node.type === 'text') return node.value;
  return (node.children ?? []).map(textOf).join('');
}

// Keep the heading fragments produced by the Python generator on old posts.
export function legacyHeadings() {
  return (tree, file) => {
    if (!/2024-\d\d-\d\d-/.test(file.path ?? '')) return;
    const ids = new Set();
    visit(tree, (node) => {
      if (!/^h[1-6]$/.test(node.tagName) || node.properties.id) return;
      if (node.tagName === 'h1') node.tagName = 'h2';
      const base = textOf(node).toLowerCase().replaceAll(' ', '-');
      let id = base || 'section';
      let number = 1;
      while (ids.has(id)) id = `${base}-${++number}`;
      ids.add(id);
      node.properties.id = id;
    });
  };
}

export function writingResources() {
  return (tree) => visit(tree, (node) => {
    for (const key of ['src', 'href']) {
      const url = node.properties[key];
      if (typeof url === 'string' && url.startsWith('../../media/')) {
        node.properties[key] = url.replace('../../media/', '/blog/media/');
      }
    }
    if (node.tagName === 'img') {
      node.properties.loading = 'lazy';
      node.properties.decoding = 'async';
    }
    if (node.tagName === 'table') {
      const table = { ...node };
      node.tagName = 'div';
      node.properties = { className: ['table-scroll'], tabIndex: 0, role: 'region', ariaLabel: 'Article table' };
      node.children = [table];
    }
  });
}

// Replace math nodes at build time so readers can use native MathML.
export function renderMath() {
  return (tree) => {
    function transform(parent) {
      parent.children = parent.children?.map((node) => {
        const code = node.tagName === 'pre' ? node.children?.[0] : node;
        const classes = code?.properties?.className ?? [];
        if (classes.includes('language-math')) {
          const value = katex.renderToString(textOf(code).trim(), {
            displayMode: classes.includes('math-display'),
            output: 'mathml',
            throwOnError: true,
          });
          return { type: 'raw', value };
        }
        if (node.children) transform(node);
        return node;
      });
    }
    transform(tree);
  };
}
