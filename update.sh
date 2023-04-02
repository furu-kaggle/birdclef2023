#!/bin/bash

# コミットメッセージの生成
commit_message="Update: $(date '+%Y-%m-%d %H:%M:%S')"

# 更新された*.pyファイルを自動追加
git add -u '*.py *.sh'
git add -u 'src/*.py'

# コミット
git commit -m "$commit_message"

# コミットUUIDの取得
commit_uuid=$(git rev-parse HEAD)

# リモートリポジトリにプッシュ
git push origin feature4-cloud

# 最後のコミットメッセージとUUIDを表示
echo "Last commit message: $commit_message"
echo "Last commit UUID: $commit_uuid"

# Kaggle datasetの更新
cp src/*.py src/weight/exp/
#cd src/weight
#cp -r exp "$commit_message"
#rm exp/*
kaggle datasets version -m "$commit_message" -p src/weight -r tar