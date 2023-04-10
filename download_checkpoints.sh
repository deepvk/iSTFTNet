#!/bin/bash
cur="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)" # to make running the script from anywhere available
cd $cur
# download data.zip (awesome_checkpoints - model checkpoints, deep_voices_wav, deep_voices_mel)
gdown https://drive.google.com/uc?id=https://drive.google.com/uc?id=15--TZc6kv4wn814p4AdfYI5sW9VKjv-s;
unzip -qq data.zip;