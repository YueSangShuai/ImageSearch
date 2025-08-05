from mteb import MTEB
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",trust_remote_code=True)
evaluation = MTEB(tasks=["AmazonCounterfactualClassification", "STSBenchmark"])
evaluation.run(model, output_folder="results/")
