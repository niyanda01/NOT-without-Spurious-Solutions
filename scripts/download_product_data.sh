#!/usr/bin/env bash
# Downloads the pix2pix edges2shoes / edges2handbags archives used by
# notebook 8 (Handbags -> Shoes) into ../data/.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${repo_root}/data"
mkdir -p "${data_root}"

for dataset in edges2shoes edges2handbags; do
    archive="${data_root}/${dataset}.tar.gz"
    target="${data_root}/${dataset}"
    if [[ -d "${target}" ]]; then
        echo "[skip] ${target} already exists"
        continue
    fi
    url="https://efrosgans.eecs.berkeley.edu/pix2pix/datasets/${dataset}.tar.gz"
    echo "[download] ${url}"
    wget -c --progress=dot:giga "${url}" -O "${archive}"
    echo "[extract] ${archive}"
    tar -xzf "${archive}" -C "${data_root}"
    rm "${archive}"
done

echo "Prepared:"
find "${data_root}/edges2shoes" "${data_root}/edges2handbags" -type f | wc -l
