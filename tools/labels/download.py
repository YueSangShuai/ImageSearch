import os
import asyncio
import aiohttp
import aiofiles
import pyarrow.parquet as pq
from tqdm import tqdm

# 参数配置
PARQUET_PATH = "/data/yuesang/person_attribute/HumanCaption-10M/HumanCaption-10M.parquet"
SAVE_ROOT = "/data/yuesang/person_attribute/HumanCaption-10M_dataset"
FAILED_LOG = "/data/yuesang/person_attribute/failed.txt"
MAX_RETRIES = 3
MAX_CONCURRENT = 50
BATCH_SIZE = 500  # 每批次读取500条

# 删除旧的失败日志
if os.path.exists(FAILED_LOG):
    os.remove(FAILED_LOG)

async def download_image_with_retry(session, url, image_path, semaphore):
    img_full_path = os.path.join(SAVE_ROOT, image_path)

    # 跳过已存在图像
    if os.path.exists(img_full_path):
        return

    os.makedirs(os.path.dirname(img_full_path), exist_ok=True)

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        async with aiofiles.open(img_full_path, mode='wb') as f:
                            await f.write(data)
                        return
        except Exception:
            await asyncio.sleep(0.5 * (attempt + 1))

    # 写入失败日志
    async with aiofiles.open(FAILED_LOG, mode='a', encoding='utf-8') as f:
        await f.write(f"{url}\t{image_path}\n")

async def process_batch(batch, session, semaphore):
    df = batch.to_pandas()
    df = df[['url', 'image', 'human_caption']].dropna()
    df = df.drop_duplicates(subset='image')

    tasks = []
    for row in df.itertuples(index=False):
        # 保存 human_caption 到对应 txt 文件
        img_path = os.path.join(SAVE_ROOT, row.image)
        caption_path = os.path.splitext(img_path)[0] + ".txt"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        if not os.path.exists(caption_path):
            caption_str = str(row.human_caption)  # 👈 修复点
            async with aiofiles.open(caption_path, mode='w', encoding='utf-8') as f:
                await f.write(caption_str)

        # 下载图片任务
        tasks.append(download_image_with_retry(session, row.url, row.image, semaphore))

    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        await coro

async def main():
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    timeout = aiohttp.ClientTimeout(total=15)

    parquet_file = pq.ParquetFile(PARQUET_PATH)
    print("文件总行数:", parquet_file.metadata.num_rows)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for batch in tqdm(parquet_file.iter_batches(batch_size=BATCH_SIZE)):
            await process_batch(batch, session, semaphore)

if __name__ == "__main__":
    asyncio.run(main())
