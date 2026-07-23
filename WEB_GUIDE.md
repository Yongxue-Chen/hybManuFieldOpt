# 网页内容维护与发布

这个分支包含一个纯静态项目网页，不需要 Node.js、Python 或额外构建工具。网页结构参考 `hybManuAccEro` 仓库的 `web` 分支，但使用了更精简、便于手动维护的 HTML、CSS 和 JavaScript。

## 文件说明

- `index.html`：页面内容、章节顺序、链接和图片位置。
- `styles.css`：页面布局、颜色、字体和响应式样式。
- `script.js`：移动端菜单、复制 BibTeX 和页脚年份。
- `web-assets/`：建议统一保存网页图片、视频封面、PDF 等静态资源。

## 本地预览

切换到 `web` 分支后，在仓库根目录运行：

```bash
python -m http.server 8000
```

浏览器打开 `http://localhost:8000`。完成预览后，在终端按 `Ctrl+C` 停止服务。

也可以直接打开 `index.html`，但本地 HTTP 服务更接近 GitHub Pages 的实际行为。

## 修改文字和链接

打开 `index.html`，搜索 `TODO:` 即可找到需要替换的位置，主要包括：

1. 论文会议和年份；
2. 项目标题、作者、单位和作者主页；
3. Paper、Code、Video 链接；
4. 项目简介、方法说明和贡献；
5. 实验结果说明；
6. BibTeX。

页面导航通过章节的 `id` 定位，例如 `href="#method"` 对应 `id="method"`。修改章节名时，注意保持二者一致。

## 添加和替换图片

把图片放入 `web-assets/`，推荐使用只含小写字母、数字和连字符的文件名，例如：

```text
web-assets/
├── teaser.webp
├── pipeline.webp
├── result-comparison.webp
└── physical-validation.webp
```

优先使用 WebP 或经过压缩的 JPEG/PNG。GitHub Pages 仓库不适合存放体积很大的原始视频或无压缩图片。

将对应的占位块：

```html
<div class="media-placeholder media-placeholder-wide">
  <span>Teaser image / overview video</span>
</div>
```

替换为：

```html
<img
  class="project-media"
  src="web-assets/teaser.webp"
  alt="A concise description of the teaser figure"
/>
```

请填写有意义的 `alt` 文本。结果区的图片也可以用同样方式替换每个 `.media-placeholder`。

## 添加视频

推荐嵌入 YouTube 视频，避免把大型视频文件提交到 Git 仓库。将视频占位块替换为：

```html
<iframe
  class="video-frame"
  src="https://www.youtube.com/embed/VIDEO_ID"
  title="FieldOpt-HM project video"
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
https://yongxue-chen.github.io/fieldOpt_HM/
```

以后只需继续向 `web` 分支提交并推送，GitHub Pages 会自动更新网站。部署状态可以在仓库的 **Actions** 页面查看。

如果之后绑定自定义域名，请在 **Settings → Pages → Custom domain** 中配置；不要手动添加 `CNAME`，除非已经确定最终域名。
