# 网页内容维护与发布

这个分支包含一个纯静态项目网页，不需要 Node.js、Python 或额外构建工具。网页结构参考 `hybManuAccEro` 仓库的 `web` 分支，但使用了更精简、便于手动维护的 HTML、CSS 和 JavaScript。

## 文件说明

- `index.html`：页面内容、章节顺序、链接和图片位置。
- `styles.css`：页面布局、颜色、字体和响应式样式。
- `script.js`：移动端菜单、复制 BibTeX、图片自动加载和页脚年份。
- `web-assets/`：建议统一保存网页图片、视频封面、PDF 等静态资源。

## 本地预览

切换到 `web` 分支后，在仓库根目录运行：

```bash
python -m http.server 8000
```

浏览器打开 `http://localhost:8000`。完成预览后，在终端按 `Ctrl+C` 停止服务。

也可以直接打开 `index.html`，但本地 HTTP 服务更接近 GitHub Pages 的实际行为。

## 修改文字和链接

当前论文摘要、方法说明、实验结论、作者、单位、期刊和代码链接已经填写。后续主要需要更新：

1. Paper 的正式 URL；
2. Video 的嵌入链接；
3. DOI、卷号、期号和文章号；
4. 最终 BibTeX。

页面导航通过章节的 `id` 定位，例如 `href="#method"` 对应 `id="method"`。修改章节名时，注意保持二者一致。

## 添加和替换图片

把图片放入 `web-assets/`。页面会自动检测并加载以下文件；文件不存在时保留带命名提示的占位块：

```text
web-assets/
├── fig-01-overview.webp
├── fig-03-pipeline.webp
├── fig-04-femur.webp
├── fig-10-distortion-ablation.webp
├── fig-15-scalability.webp
└── fig-17-physical-models.webp
```

建议从论文原始图文件导出，而不是对 PDF 截屏。优先使用 WebP，使用 sRGB 色彩空间，宽度建议为 1800–2400 px，单张尽量控制在 1 MB 内。Fig. 10 是实验区主图，建议宽度 2200–2600 px；其余图片宽度 1600–2200 px 即可。

如果需要加入单位 logo，使用透明背景的 SVG 或 PNG，并命名为：

```text
web-assets/
├── logo-digital-manufacturing-lab.svg
└── logo-university-of-manchester.svg
```

当前设计使用文字单位署名，不依赖 logo；收到正式 logo 文件后再加入页面即可。

## 添加视频

推荐嵌入 YouTube 视频，避免把大型视频文件提交到 Git 仓库。将视频占位块替换为：

```html
<iframe
  class="video-frame"
  src="https://www.youtube.com/embed/VIDEO_ID"
  title="Field optimization for hybrid manufacturing project video"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen
></iframe>
```

将 `VIDEO_ID` 替换成 YouTube 链接中的视频 ID。

## 调整样式

在 `styles.css` 顶部的 `:root` 中可以集中修改主色、背景色、文字颜色和页面最大宽度。通常不需要逐项查找所有 CSS：

```css
:root {
  --accent: #ed653e;
  --teal: #0c6862;
  --max-width: 1180px;
}
```

## 通过 GitHub Pages 发布

内容确认后，在 `web` 分支提交并推送：

```bash
git add index.html styles.css script.js WEB_GUIDE.md web-assets
git commit -m "Update project website"
git push -u origin web
```

第一次发布时，在 GitHub 仓库页面进行以下设置：

1. 打开 **Settings → Pages**；
2. 在 **Build and deployment** 中把 **Source** 设为 **Deploy from a branch**；
3. 分支选择 **web**，目录选择 **/(root)**；
4. 点击 **Save**。

首次部署通常需要几分钟。GitHub 会在 Pages 设置页显示最终网址，项目仓库的默认地址通常为：

```text
https://yongxue-chen.github.io/hybManuFieldOpt/
```

以后只需继续向 `web` 分支提交并推送，GitHub Pages 会自动更新网站。部署状态可以在仓库的 **Actions** 页面查看。

如果之后绑定自定义域名，请在 **Settings → Pages → Custom domain** 中配置；不要手动添加 `CNAME`，除非已经确定最终域名。
