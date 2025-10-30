#!/bin/bash

# GitHub Repository Push and Setup Commands
# GitHub 仓库推送和设置命令
# This script helps push the project to GitHub and configure repository settings
# 此脚本帮助将项目推送到 GitHub 并配置仓库设置

set -e

REPO_NAME="vllm_benchmark_block_fp8"
DESCRIPTION=$(cat GITHUB_DESCRIPTION.txt 2>/dev/null || echo "Automated Triton w8a8 block FP8 kernel tuning tool for vLLM")

echo "==================== GitHub Setup ===================="
echo "Repository: $REPO_NAME"
echo "Description: $DESCRIPTION"
echo "======================================================"
echo

# Check if git is initialized / 检查 git 是否已初始化
if [[ ! -d .git ]]; then
    echo "Initializing git repository..."
    echo "初始化 git 仓库..."
    git init
    git add .
    git commit -m "Initial commit: vLLM Block FP8 Kernel Tuning Tool"
    echo "Git repository initialized."
    echo "Git 仓库已初始化"
    echo
fi

# Check if remote exists / 检查远程仓库是否存在
if git remote | grep -q "^origin$"; then
    echo "Remote 'origin' already exists."
    echo "远程 'origin' 已存在"
else
    echo "Please create the repository on GitHub first, then run:"
    echo "请先在 GitHub 上创建仓库，然后运行："
    echo "  git remote add origin https://github.com/YOUR_USERNAME/$REPO_NAME.git"
    echo "  git push -u origin main"
    echo
    exit 1
fi

# Push to GitHub / 推送到 GitHub
echo "Pushing to GitHub..."
echo "推送到 GitHub..."
git add .
git commit -m "Update: Improve model auto-detection and add preset scripts" || echo "No changes to commit"
git push -u origin main || git push -u origin master

echo
echo "Setting repository description..."
echo "设置仓库描述..."
gh repo edit --description "$DESCRIPTION" || echo "Warning: Failed to set description. Make sure GitHub CLI is installed and authenticated."

echo
echo "Adding repository topics..."
echo "添加仓库主题..."
gh repo edit --add-topic vllm --add-topic fp8 --add-topic kernel-tuning --add-topic triton --add-topic quantization --add-topic qwen3 --add-topic deepseek-v3 || echo "Warning: Failed to add topics."

echo
echo "==================== Done ===================="
echo "Repository setup completed!"
echo "仓库设置完成！"
echo "Visit: https://github.com/YOUR_USERNAME/$REPO_NAME"
echo "访问: https://github.com/YOUR_USERNAME/$REPO_NAME"

