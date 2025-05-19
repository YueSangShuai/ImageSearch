由于mexma-siglip2模型运行时需要facebook/MEXMA/模型文件，为了避免临时下载模型文件，需要在在此目录下建立MEXMA目录，然后将facebook/MEXMA模型文件复制到该目录下：

```
$ ls facebook/MEXMA/ -l
total 2183032
-rw-rw-r-- 1 richard richard        703 Mar 12 07:28 config.json
-rw-rw-r-- 1 richard richard 2235408584 Mar 12 07:35 model.safetensors
-rw-rw-r-- 1 richard richard       2397 Mar 12 07:28 README.md
```

或者建立软链接：

```
$ ls facebook/ -l
total 0
lrwxrwxrwx 1 richard richard 19 Mar 12 07:45 MEXMA -> /data/models/MEXMA/
-rw-rw-r-- 1 richard richard  0 Mar 13 01:12 readme.txt

```