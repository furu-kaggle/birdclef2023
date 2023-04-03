#!/bin/bash

# コミットメッセージの生成
commit_message="Update: $(date '+%Y-%m-%d %H:%M:%S')"

# 更新された*.pyファイルを自動追加
git add -u 1.train.py update.sh
git add -u src/*.py

# コミット
git commit -m "$commit_message"

# コミットUUIDの取得
commit_uuid=$(git rev-parse HEAD)

# リモートリポジトリにプッシュ
git push origin HEAD

# 最後のコミットメッセージとUUIDを表示
echo "Last commit message: $commit_message"
echo "Last commit UUID: $commit_uuid"

# Kaggle datasetの更新
cp src/*.py src/weight/exp/
cd src/weight
cp -r exp "$commit_uuid"
rm exp/*
kaggle datasets version -m "$commit_message" -r tar