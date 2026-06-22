# GitHub 上传说明

当前项目应在工作区根目录作为 Git 仓库管理：

```bash
git status
git add .
git commit -m "Initial cooperative puzzle MARL project"
git remote add origin https://github.com/<your-name>/<repo-name>.git
git branch -M main
git push -u origin main
```

注意：

- 不要提交 `outputs/`、模型权重、虚拟环境和 Python 缓存。
- 如果还没有 GitHub 空仓库，需要先在 GitHub 网页端创建 repository。
- 如果本机没有配置 Git 身份，先执行：

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

