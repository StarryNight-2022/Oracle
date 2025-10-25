export CUDA_VISIBLE_DEVICES=1
vllm serve /home/ouyk/project/Runtime/Model/Qwen3-Embedding-0.6B             \
            --api-key token-abc123 --dtype auto                              \
            --trust-remote-code                                              \
            --served-model-name embed                                        \
            --max-model-len   32768                                          \
            # --task embed

# vllm serve /home/ouyk/project/Runtime/Model/Qwen3-Embedding-0.6B             \
#             --host 0.0.0.0 --port 8080  --block-size 16                      \
#             --api-key 123456 --dtype auto                                    \
#             --trust-remote-code                                              \
#             --served-model-name embed                                        \
#             --enable-prefix-caching                                          \
#             --max-model-len   32768                                          \
#             --gpu-memory-utilization 0.9                                     \
#             --task embed --disable-log-requests