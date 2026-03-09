#!/usr/bin/env node
/**
 * MathJax LaTeX → SVG 服务端渲染
 *
 * 用法: node render_mathjax.js '<LaTeX公式>'
 * 输出: SVG 字符串到 stdout
 *
 * 依赖: 本地 MathJax 3.x (es5/)
 */

const path = require('path');

const MATHJAX_PATH = path.resolve(__dirname, '..', 'assets', 'mathjax', 'es5');

// ★ 必须在 require MathJax 之前设置全局配置
global.MathJax = {
  loader: {
    paths: { mathjax: MATHJAX_PATH },
    require: require,
    load: ['input/tex-full', 'output/svg', 'adaptors/liteDOM'],
  },
  tex: {
    packages: { '[+]': ['ams', 'boldsymbol', 'cancel', 'color'] },
  },
  svg: {
    fontCache: 'none',
  },
  startup: {
    typeset: false,
  },
};

// 加载 MathJax（读取上面的全局配置）
require(path.join(MATHJAX_PATH, 'startup.js'));

const formula = process.argv[2];
if (!formula) {
  console.error('用法: node render_mathjax.js "<LaTeX公式>"');
  process.exit(1);
}

// 等待 MathJax 初始化完成后渲染
MathJax.startup.promise.then(() => {
  const adaptor = MathJax.startup.adaptor;
  const html = MathJax.startup.document;

  const node = html.convert(formula, { display: true });
  const svgStr = adaptor.outerHTML(node);

  process.stdout.write(svgStr);
}).catch(err => {
  console.error('MathJax render error:', err.message || err);
  process.exit(2);
});
