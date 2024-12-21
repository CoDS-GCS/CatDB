from be_great import GReaT
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True).frame

print(data)

model = GReaT(llm='distilgpt2', batch_size=32,  epochs=2, fp16=True)
model.fit(data)
synthetic_data = model.sample(n_samples=100)