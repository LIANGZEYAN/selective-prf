from huggingface_hub import snapshot_download

if __name__ == '__main__':
    snapshot_download(
        repo_id="Qwen/Qwen2.5-14B-Instruct",
        cache_dir="/mnt/primary/QE audit/hf_cache",
        resume_download=True,
        endpoint="https://hf-mirror.com"  # 可选：使用镜像
    )