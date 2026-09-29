#!/usr/bin/env bash
# Clones the upstream code that four of the eight open models need (Kimi-Audio,
# MiMo-Audio, Fun-Audio-Chat, Step-Audio 2) into third_party/ at the commits we used,
# and links MiMo-Audio's packages into musiclistenbench/backends/.
# Model weights are downloaded separately (the checkpoint names are in the paper).
#
#   bash scripts/setup_third_party.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p third_party

clone() {  # clone <url> <dir> <commit|"">
    local url="$1" dir="third_party/$2" commit="$3"
    [[ -d "$dir/.git" ]] || git clone --recurse-submodules "$url" "$dir"
    [[ -n "$commit" ]] && git -C "$dir" checkout "$commit"
}

clone https://github.com/MoonshotAI/Kimi-Audio.git      Kimi-Audio      349251e1d8f4f98d58fda59246381faecd7392e0
clone https://github.com/XiaomiMiMo/MiMo-Audio.git      MiMo-Audio      691ce54144a6844cc641fd96046a6ba20776c8b0
clone https://github.com/FunAudioLLM/Fun-Audio-Chat.git Fun-Audio-Chat  89f51924d48039848350366941dd6171332a0c96
clone https://github.com/stepfun-ai/Step-Audio2.git     Step-Audio2     ""   # the backend only needs the repository's utils.py

# MiMo-Audio's own top-level package is called `src`; the backend imports its two
# sub-packages as musiclistenbench.backends.mimo_audio / .mimo_audio_tokenizer.
ln -sfn ../../third_party/MiMo-Audio/src/mimo_audio           musiclistenbench/backends/mimo_audio
ln -sfn ../../third_party/MiMo-Audio/src/mimo_audio_tokenizer musiclistenbench/backends/mimo_audio_tokenizer
echo "third_party/ ready"
