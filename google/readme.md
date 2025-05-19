由于mexma-siglip2模型运行时需要google/siglip2-so400m-patch16-512模型文件，为了避免临时下载模型文件，需要在在此目录下建立siglip2-so400m-patch16-512目录，然后将google/siglip2-so400m-patch16-512模型文件复制到该目录下：

```
$ ls -l google/siglip2-so400m-patch16-512/
total 4477552
-rw-rw-r-- 1 richard richard        537 Mar 12 07:42 config.json
-rw-rw-r-- 1 richard richard 4546331880 Mar 12 07:58 model.safetensors
-rw-rw-r-- 1 richard richard        394 Mar 12 07:42 preprocessor_config.json
-rw-rw-r-- 1 richard richard       3381 Mar 12 07:42 README.md
-rw-rw-r-- 1 richard richard        636 Mar 12 07:42 special_tokens_map.json
-rw-rw-r-- 1 richard richard      47164 Mar 12 07:42 tokenizer_config.json
-rw-rw-r-- 1 richard richard   34363039 Mar 12 07:43 tokenizer.json
-rw-rw-r-- 1 richard richard    4241003 Mar 12 07:42 tokenizer.model
```

或者建立软链接：

```
$ ls google/ -l
total 0
-rw-rw-r-- 1 richard richard  0 Mar 13 01:07 readme.txt
lrwxrwxrwx 1 richard richard 40 Mar 12 07:46 siglip2-so400m-patch16-512 -> /data/models/siglip2-so400m-patch16-512/

```