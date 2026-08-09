import subprocess
import json
import tarfile
import sys
import zstandard
import io
import random
import csv
import os
import torch
import math
import threading
import queue
vLen = 0
batch = 0
with open("vocab.csv", encoding="utf-8", errors="backslashreplace") as vocab:
	for r in csv.reader(vocab):
		vLen += len(r)
def pThread(c, p, pQueue, iQ):
	while True:
		rw = pQueue.get()
		if rw is None:
			break
		p.stdin.write(b"\0".join(rw) + bytes([0]))
		p.stdin.flush()
		iQ.put(c)
process = [None] * int(sys.argv[2])
resivor = [None] * int(sys.argv[3])
rsize = 0
batchSize = 96 * int(sys.argv[4])
batchFill = 0
batchCap = 9375 * int(sys.argv[4])
resFill = 0
step = 0
toBeAdded = None
max_rate = 6e-4
min_rate = 6e-5
warmup = 6000
max_iters = 1800000
class MaskedAttention(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.heads = heads
		self.qMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.kMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.vMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.oMat = torch.nn.Linear(heads * dims, emb_dim, bias=False)
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		# return: (batch_size, seq_len, emb_dim)
	   
		q = self.qMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		k = self.kMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		v = self.vMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		# attention: (batch_size, heads, seq_len, emb_dim)
		attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.1).transpose(-3, -2).contiguous()
		attention = self.oMat(attention.view(attention.size(0), attention.size(1), -1))
		return attention
class Layer(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.dropout = torch.nn.Dropout(0.1)
		self.heads = heads
		self.attention = MaskedAttention(emb_dim, heads, dims)
		self.normalize = torch.nn.LayerNorm([emb_dim])
		self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
		self.normalize2 = torch.nn.LayerNorm([emb_dim])
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		attention = self.attention(x)
		norm = self.normalize(x + self.dropout(attention))
		lin = self.linear(norm)
		norm2 = self.normalize2(norm + self.dropout(lin))
		return norm2
class Network(torch.nn.Module):
	def  __init__(self):
		super().__init__()
		self.dropout = torch.nn.Dropout(0.1)
		self.emb = torch.nn.Embedding(50256, 768)
		self.register_buffer("pos", (torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0) * torch.sin(torch.arange(int(sys.argv[4])).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim)) + (1 - torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0)) * torch.cos(torch.arange(int(sys.argv[4])).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim))))
		self.temp = [Layer(self.emb.embedding_dim, 12, 64) for _ in range(12)]
		self.layers = torch.nn.Sequential(*self.temp)
		self.linear3 = torch.nn.Linear(self.emb.embedding_dim, 50256, bias=False)
		self.emb.weight = self.linear3.weight
	def forward(self, x):
		embed = self.dropout(self.emb(x) + self.pos[:x.size(-1), :].unsqueeze(0))
		layer = self.layers(embed)
		lin3 = self.linear3(layer)
		return torch.log_softmax(lin3, dim=-1)
rate = 0
torch.set_float32_matmul_precision('high')
model = torch.compile(Network().to("cuda"))
lFunc = torch.nn.NLLLoss(ignore_index=-1)
tQueue = [queue.Queue() for _ in range(int(sys.argv[2]))]
iQueue = queue.Queue()
pprocess = [None] * int(sys.argv[2])
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, fused=True)
for i in range(int(sys.argv[2])):
	pprocess[i] = subprocess.Popen(["./tokenizer_parallel", "output_" + str(i) + ".csv", "vocab.csv", "vocab_" + str(i) + ".csv", sys.argv[4], "0.003"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	process[i] = threading.Thread(target=pThread, args=(i, pprocess[i], tQueue[i], iQueue))
	process[i].start()
	iQueue.put(i)
dctx = zstandard.ZstdDecompressor()
toBeAdded = None
model.zero_grad()
source = torch.empty(0, int(sys.argv[4])).to("cuda").long()
target = torch.empty(0, int(sys.argv[4])).to("cuda").long()
with tarfile.open("openwebtext2.jsonl.zst.tar") as t:
	for m in t.getmembers():
		if m.isfile():
			f = t.extractfile(m)
			if f is not None:
				with dctx.stream_reader(f) as d:
					s = io.TextIOWrapper(d, encoding="utf-8")
					for l in s:
						c = l.strip()
						if c:
							j = json.loads(c)
							if "text" in j and len(j["text"]) != 0:
								rawWritten = list(filter(None, bytearray(j["text"], "utf-8").split(b"\0")))
								if len(rawWritten) > 0:
									count = iQueue.get()
									tQueue[count].put(rawWritten)
									batch += 1
								if batch == int(sys.argv[1]):
									for tt in tQueue:
										tt.put(None)
									for tt in process:
										tt.join()
									while not iQueue.empty():
										iQueue.get_nowait()
									for tt in range(int(sys.argv[2])):
										size = int(pprocess[tt].communicate()[0])
										if pprocess[tt].returncode == 1:
											sys.exit(1)
										with open("output_" + str(tt) + ".csv", "r+b") as f:
											f.truncate(size)
									batch = 0
									additionalVocab = set()
									vocabMap = []
									for v in range(int(sys.argv[2])):
										with open("vocab_" + str(v) + ".csv", encoding="utf-8", errors="backslashreplace") as vocab:
											additionalVocab |= set(list(map(lambda x: bytes([int(x[2:], 16)]) if x.startswith("\\x") else x.encode(), next(csv.reader(vocab), []))))
									if len(additionalVocab) > 0:
										oldW = model.linear3.weight
										model.linear3.out_features = model.linear3.weight.size(0) + len(additionalVocab)
										model.emb.num_embeddings = model.emb.weight.size(0) + len(additionalVocab)
										nextEmb = torch.cat((model.linear3.weight.detach().to("cpu"), torch.empty(len(additionalVocab), model.linear3.weight.size(-1))), 0)
										torch.nn.init.kaiming_uniform_(nextEmb[-len(additionalVocab):, :], a=5 ** 0.5, mode="fan_in")
										oldI = 0
										oldG = None
										for gIndex, group in enumerate(optimizer.param_groups):
											for index, param in enumerate(group["params"]):
												if param is oldW:
													oldI = index
													oldG = gIndex
										oldD = (oldW.dtype, oldW.device, None if oldW.grad is None else oldW.grad.detach().to("cpu"), oldW.size(-1))
										old = {}
										if oldW in optimizer.state:
											for ok, ov in optimizer.state[oldW].items():
												old[ok] = ov.to("cpu")
											del ov
											optimizer.state.pop(oldW)
										del oldW
										model.emb.weight = model.linear3.weight = torch.nn.Parameter(nextEmb.to(oldD[1]))
										if oldD[2] is not None:
											model.emb.weight.grad = model.linear3.weight.grad = torch.cat((oldD[2], torch.zeros(len(additionalVocab), oldD[3], dtype=oldD[0])), 0).to(oldD[1])
										if old:
											optimizer.state[model.linear3.weight] = {"step": old["step"].to(oldD[1]), "exp_avg": torch.cat((old["exp_avg"], torch.zeros(len(additionalVocab), oldD[3], dtype=oldD[0])), 0).to(oldD[1]), "exp_avg_sq": torch.cat((old["exp_avg_sq"], torch.zeros(len(additionalVocab), oldD[3], dtype=oldD[0])), 0).to(oldD[1])}
										if oldG is not None:
											optimizer.param_groups[oldG]["params"][oldI] = model.linear3.weight
									for v in range(int(sys.argv[2])):
										with open("vocab_" + str(v) + ".csv", encoding="utf-8", errors="backslashreplace") as vocab:
											vocabMap += [list(map(lambda x: list(additionalVocab).index(x), list(map(lambda x: bytes([int(x[2:], 16)]) if x.startswith("\\x") else x.encode(), next(csv.reader(vocab), [])))))]
									for v in range(int(sys.argv[2])):
										while os.path.isfile("output_" + str(v) + ".csv"):
											if os.path.getsize("output_" + str(v) + ".csv") > 0 or toBeAdded != None:
												if toBeAdded == None:
													with open("output_" + str(v) + ".csv", "r+b") as tokFile:
														tokFile.seek(0, 2)
														while tokFile.tell() != 0 and (tokFile.peek(1) == b"" or tokFile.peek(1)[0] != ord("\n")):
															tokFile.seek(-1, 1)
															if tokFile.peek(1)[0] == ord("\n") or tokFile.tell() == 0:
																tempPos = tokFile.tell()
																toBeAdded = tokFile.read()
																tokFile.seek(tempPos)
																if toBeAdded[0] == ord("\n"):
																	toBeAdded = toBeAdded[1:]
																toBeAdded = toBeAdded.decode("utf-8").replace("\"", "").split(",")
																toBeAdded = [list(map(lambda x: x if x < vLen else (vocabMap[v][x - vLen] + vLen), list(map(int, toBeAdded[0::2])))), list(map(lambda x: x if x < vLen else (vocabMap[v][x - vLen] + vLen), list(map(int, toBeAdded[1::2]))))]
														tokFile.truncate()
												if resFill < int(sys.argv[3]):
													resivor[resFill] = toBeAdded
													resFill += 1
													toBeAdded = None
												else:
													if batchFill == batchSize or source.size(0) == 16:
														with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
															loss = lFunc(model(torch.maximum(source, torch.zeros(source.size()).to("cuda").long())).reshape(-1, model.linear3.weight.size(0)), target.reshape(-1)) * ((target != -1).sum() / batchSize)
															loss.backward()
														source = torch.empty(0, int(sys.argv[4])).to("cuda").long()
														target = torch.empty(0, int(sys.argv[4])).to("cuda").long()
														if batchFill == batchSize:
															torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
															if rate < warmup:
																learning = max_rate * (rate + 1) / (warmup + 1)
																for param in optimizer.param_groups:
																	param["lr"] = learning
															elif rate > max_iters:
																for param in optimizer.param_groups:
																	param["lr"] = min_rate
															else:
																learning = min_rate + 0.5 * (1.0 + math.cos(math.pi * (rate - warmup) / (max_iters - warmup))) * (max_rate - min_rate)
																for param in optimizer.param_groups:
																	param["lr"] = learning
															optimizer.step()
															rate += 1
															model.zero_grad()
															batchFill = 0
															batchSize = (batchSize + int(sys.argv[4]) * 16) if batchSize < batchCap else batchSize
													source = torch.cat((source, -torch.ones(1, int(sys.argv[4])).to("cuda").long()), 0)
													target = torch.cat((target, -torch.ones(1, int(sys.argv[4])).to("cuda").long()), 0)
													if random.randint(0, int(sys.argv[3])) == 0:
														if batchSize - batchFill < len(toBeAdded[0]):
															source[-1, :batchSize - batchFill] = torch.tensor(toBeAdded[0][:batchSize - batchFill]).to("cuda").long()
															toBeAdded[0] = toBeAdded[0][batchSize - batchFill:]
															target[-1, :batchSize - batchFill] = torch.tensor(toBeAdded[1][:batchSize - batchFill]).to("cuda").long()
															toBeAdded[1] = toBeAdded[1][batchSize - batchFill:]
															batchFill = batchSize
														else:
															source[-1, :len(toBeAdded[0])] = torch.tensor(toBeAdded[0]).to("cuda").long()
															target[-1, :len(toBeAdded[1])] = torch.tensor(toBeAdded[1]).to("cuda").long()
															batchFill += len(toBeAdded[0])
															toBeAdded = None
													else:
														replaceIndex = random.randint(0, int(sys.argv[3]) - 1)
														if batchSize - batchFill < len(resivor[replaceIndex][0]):
															source[-1, :batchSize - batchFill] = torch.tensor(resivor[replaceIndex][0][:batchSize - batchFill]).to("cuda").long()
															resivor[replaceIndex][0] = resivor[replaceIndex][0][batchSize - batchFill:]
															target[-1, :batchSize - batchFill] = torch.tensor(resivor[replaceIndex][1][:batchSize - batchFill]).to("cuda").long()
															resivor[replaceIndex][1] = resivor[replaceIndex][1][batchSize - batchFill:]
															batchFill = batchSize
														else:
															source[-1, :len(resivor[replaceIndex][0])] = torch.tensor(resivor[replaceIndex][0]).to("cuda").long()
															target[-1, :len(resivor[replaceIndex][1])] = torch.tensor(resivor[replaceIndex][1]).to("cuda").long()
															batchFill += len(resivor[replaceIndex][0])
															resivor[replaceIndex] = toBeAdded
															toBeAdded = None
											if toBeAdded == None and os.path.getsize("output_" + str(v) + ".csv") == 0:
												os.remove("output_" + str(v) + ".csv")
									if os.path.getsize("vocab.csv") != 0 and len(additionalVocab) != 0:
										with open("vocab.csv", "ab") as vF:
											vF.write(b",")
									with open("vocab.csv", "ab") as vF:
										vF.write(b",".join(list(map(lambda x: b"\"\"\"\"" if x == b"\"" else (b"\n" if x == b"\n" else (b"\",\"" if x == b"," else x)), list(additionalVocab)))))
									vLen += len(list(additionalVocab))
									process = [None] * int(sys.argv[2])
									for i in range(int(sys.argv[2])):
										pprocess[i] = subprocess.Popen(["./tokenizer_parallel", "output_" + str(i) + ".csv", "vocab.csv", "vocab_" + str(i) + ".csv", sys.argv[4], "0.003"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
										process[i] = threading.Thread(target=pThread, args=(i, pprocess[i], tQueue[i], iQueue))
										process[i].start()
										iQueue.put(i)
for t in tQueue:
	t.put(None)
for t in process:
	t.join()
while not iQueue.empty():
	iQueue.get_nowait()
for t in range(int(sys.argv[2])):
	size = int(pprocess[t].communicate()[0])
	if pprocess[t].returncode == 1:
		sys.exit(1)
	with open("output_" + str(t) + ".csv", "r+b") as f:
		f.truncate(size)
batch = 0
additionalVocab = set()
vocabMap = []
for v in range(int(sys.argv[2])):
	with open("vocab_" + str(v) + ".csv", encoding="utf-8", errors="backslashreplace") as vocab:
		additionalVocab |= set(list(map(lambda x: bytes([int(x[2:], 16)]) if x.startswith("\\x") else x.encode(), next(csv.reader(vocab), []))))
if len(additionalVocab) > 0:
	oldW = model.linear3.weight
	model.linear3.out_features = model.linear3.weight.size(0) + len(additionalVocab)
	model.emb.num_embeddings = model.emb.weight.size(0) + len(additionalVocab)
	nextEmb = torch.cat((model.linear3.weight.detach().to("cpu"), torch.empty(len(additionalVocab), model.linear3.weight.size(-1))), 0)
	torch.nn.init.kaiming_uniform_(nextEmb[-len(additionalVocab):, :], a=5 ** 0.5, mode="fan_in")
	oldI = 0
	oldG = None
	for gIndex, group in enumerate(optimizer.param_groups):
		for index, param in enumerate(group["params"]):
			if param is oldW:
				oldI = index
				oldG = gIndex
	oldD = (oldW.dtype, oldW.device, None if oldW.grad is None else oldW.grad.detach().to("cpu"), oldW.size(-1))
	old = {}
	if oldW in optimizer.state:
		for ok, ov in optimizer.state[oldW].items():
			old[ok] = ov.to("cpu")
		del ov
		optimizer.state.pop(oldW)
	del oldW
	model.emb.weight = model.linear3.weight = torch.nn.Parameter(nextEmb.to(oldD[1]))
	if oldD[2] is not None:
		model.emb.weight.grad = model.linear3.weight.grad = torch.cat((oldD[2], torch.zeros(len(additionalVocab), oldD[3], dtype=oldD[0])), 0).to(oldD[1])
	if old:
		optimizer.state[model.linear3.weight] = {"step": old["step"].to(oldD[1]), "exp_avg": torch.cat((old["exp_avg"], torch.zeros(len(additionalVocab), oldD[3], dtype=oldD[0])), 0).to(oldD[1]), "exp_avg_sq": torch.cat((old["exp_avg_sq"], torch.zeros(len(additionalVocab), oldD[3], dtype=oldD[0])), 0).to(oldD[1])}
	if oldG is not None:
		optimizer.param_groups[oldG]["params"][oldI] = model.linear3.weight
for v in range(int(sys.argv[2])):
	with open("vocab_" + str(v) + ".csv", encoding="utf-8", errors="backslashreplace") as vocab:
		vocabMap += [list(map(lambda x: list(additionalVocab).index(x), list(map(lambda x: bytes([int(x[2:], 16)]) if x.startswith("\\x") else x.encode(), next(csv.reader(vocab), [])))))]
for v in range(int(sys.argv[2])):
	while os.path.isfile("output_" + str(v) + ".csv"):
		if os.path.getsize("output_" + str(v) + ".csv") > 0 or toBeAdded != None:
			if toBeAdded == None:
				with open("output_" + str(v) + ".csv", "r+b") as tokFile:
					tokFile.seek(0, 2)
					while tokFile.tell() != 0 and (tokFile.peek(1) == b"" or tokFile.peek(1)[0] != ord("\n")):
						tokFile.seek(-1, 1)
						if tokFile.peek(1)[0] == ord("\n") or tokFile.tell() == 0:
							tempPos = tokFile.tell()
							toBeAdded = tokFile.read()
							tokFile.seek(tempPos)
							if toBeAdded[0] == ord("\n"):
								toBeAdded = toBeAdded[1:]
							toBeAdded = toBeAdded.decode("utf-8").replace("\"", "").split(",")
							toBeAdded = [list(map(lambda x: x if x < vLen else (vocabMap[v][x - vLen] + vLen), list(map(int, toBeAdded[0::2])))), list(map(lambda x: x if x < vLen else (vocabMap[v][x - vLen] + vLen), list(map(int, toBeAdded[1::2]))))]
					tokFile.truncate()
			if resFill < int(sys.argv[3]):
				resivor[resFill] = toBeAdded
				resFill += 1
				toBeAdded = None
			else:
				if batchFill == batchSize or source.size(0) == 16:
					with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
						loss = lFunc(model(torch.maximum(source, torch.zeros(source.size()).to("cuda").long())).reshape(-1, model.linear3.weight.size(0)), target.reshape(-1)) * ((target != -1).sum() / batchSize)
						loss.backward()
					source = torch.empty(0, int(sys.argv[4])).to("cuda").long()
					target = torch.empty(0, int(sys.argv[4])).to("cuda").long()
					if batchFill == batchSize:
						torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
						if rate < warmup:
							learning = max_rate * (rate + 1) / (warmup + 1)
							for param in optimizer.param_groups:
								param["lr"] = learning
						elif rate > max_iters:
							for param in optimizer.param_groups:
								param["lr"] = min_rate
						else:
							learning = min_rate + 0.5 * (1.0 + math.cos(math.pi * (rate - warmup) / (max_iters - warmup))) * (max_rate - min_rate)
							for param in optimizer.param_groups:
								param["lr"] = learning
						optimizer.step()
						rate += 1
						model.zero_grad()
						batchFill = 0
						batchSize = (batchSize + int(sys.argv[4]) * 16) if batchSize < batchCap else batchSize
				source = torch.cat((source, -torch.ones(1, int(sys.argv[4])).to("cuda").long()), 0)
				target = torch.cat((target, -torch.ones(1, int(sys.argv[4])).to("cuda").long()), 0)
				if random.randint(0, int(sys.argv[3])) == 0:
					if batchSize - batchFill < len(toBeAdded[0]):
						source[-1, :batchSize - batchFill] = torch.tensor(toBeAdded[0][:batchSize - batchFill]).to("cuda").long()
						toBeAdded[0] = toBeAdded[0][batchSize - batchFill:]
						target[-1, :batchSize - batchFill] = torch.tensor(toBeAdded[1][:batchSize - batchFill]).to("cuda").long()
						toBeAdded[1] = toBeAdded[1][batchSize - batchFill:]
						batchFill = batchSize
					else:
						source[-1, :len(toBeAdded[0])] = torch.tensor(toBeAdded[0]).to("cuda").long()
						target[-1, :len(toBeAdded[1])] = torch.tensor(toBeAdded[1]).to("cuda").long()
						batchFill += len(toBeAdded[0])
						toBeAdded = None
				else:
					replaceIndex = random.randint(0, int(sys.argv[3]) - 1)
					if batchSize - batchFill < len(resivor[replaceIndex][0]):
						source[-1, :batchSize - batchFill] = torch.tensor(resivor[replaceIndex][0][:batchSize - batchFill]).to("cuda").long()
						resivor[replaceIndex][0] = resivor[replaceIndex][0][batchSize - batchFill:]
						target[-1, :batchSize - batchFill] = torch.tensor(resivor[replaceIndex][1][:batchSize - batchFill]).to("cuda").long()
						resivor[replaceIndex][1] = resivor[replaceIndex][1][batchSize - batchFill:]
						batchFill = batchSize
					else:
						source[-1, :len(resivor[replaceIndex][0])] = torch.tensor(resivor[replaceIndex][0]).to("cuda").long()
						target[-1, :len(resivor[replaceIndex][1])] = torch.tensor(resivor[replaceIndex][1]).to("cuda").long()
						batchFill += len(resivor[replaceIndex][0])
						resivor[replaceIndex] = toBeAdded
						toBeAdded = None
		if toBeAdded == None and os.path.getsize("output_" + str(v) + ".csv") == 0:
			os.remove("output_" + str(v) + ".csv")
if os.path.getsize("vocab.csv") != 0 and len(additionalVocab) != 0:
	with open("vocab.csv", "ab") as vF:
		vF.write(b",")
with open("vocab.csv", "ab") as vF:
	vF.write(b",".join(list(map(lambda x: b"\"\"\"\"" if x == b"\"" else (b"\n" if x == b"\n" else (b"\",\"" if x == b"," else x)), list(additionalVocab)))))
resivor = resivor[:resFill]
random.shuffle(resivor)
replaceIndex = 0
while replaceIndex < len(resivor):
	if batchFill == batchSize or source.size(0) == 16:
		with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
			loss = lFunc(model(torch.maximum(source, torch.zeros(source.size()).to("cuda").long())).reshape(-1, model.linear3.weight.size(0)), target.reshape(-1)) * ((target != -1).sum() / batchSize)
			loss.backward()
		source = torch.empty(0, int(sys.argv[4])).to("cuda").long()
		target = torch.empty(0, int(sys.argv[4])).to("cuda").long()
		if batchFill == batchSize:
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			if rate < warmup:
				learning = max_rate * (rate + 1) / (warmup + 1)
				for param in optimizer.param_groups:
					param["lr"] = learning
			elif rate > max_iters:
				for param in optimizer.param_groups:
					param["lr"] = min_rate
			else:
				learning = min_rate + 0.5 * (1.0 + math.cos(math.pi * (rate - warmup) / (max_iters - warmup))) * (max_rate - min_rate)
				for param in optimizer.param_groups:
					param["lr"] = learning
			optimizer.step()
			rate += 1
			model.zero_grad()
			batchFill = 0
			batchSize = (batchSize + int(sys.argv[4]) * 16) if batchSize < batchCap else batchSize
	source = torch.cat((source, -torch.ones(1, int(sys.argv[4])).to("cuda").long()), 0)
	target = torch.cat((target, -torch.ones(1, int(sys.argv[4])).to("cuda").long()), 0)
	if batchSize - batchFill < len(resivor[replaceIndex][0]):
		source[-1, :batchSize - batchFill] = torch.tensor(resivor[replaceIndex][0][:batchSize - batchFill]).to("cuda").long()
		resivor[replaceIndex][0] = resivor[replaceIndex][0][batchSize - batchFill:]
		target[-1, :batchSize - batchFill] = torch.tensor(resivor[replaceIndex][1][:batchSize - batchFill]).to("cuda").long()
		resivor[replaceIndex][1] = resivor[replaceIndex][1][batchSize - batchFill:]
		batchFill = batchSize
	else:
		source[-1, :len(resivor[replaceIndex][0])] = torch.tensor(resivor[replaceIndex][0]).to("cuda").long()
		target[-1, :len(resivor[replaceIndex][1])] = torch.tensor(resivor[replaceIndex][1]).to("cuda").long()
		batchFill += len(resivor[replaceIndex][0])
		replaceIndex += 1
if source.size(0) > 0:
	with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
		loss = lFunc(model(torch.maximum(source, torch.zeros(source.size()).to("cuda").long())).reshape(-1, model.linear3.weight.size(0)), target.reshape(-1)) * ((target != -1).sum() / batchSize)
		loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
if rate < warmup:
	learning = max_rate * (rate + 1) / (warmup + 1)
	for param in optimizer.param_groups:
		param["lr"] = learning
elif rate > max_iters:
	for param in optimizer.param_groups:
		param["lr"] = min_rate
else:
	learning = min_rate + 0.5 * (1.0 + math.cos(math.pi * (rate - warmup) / (max_iters - warmup))) * (max_rate - min_rate)
	for param in optimizer.param_groups:
		param["lr"] = learning
optimizer.step()
class EvalMaskedAttention(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.heads = heads
		self.qMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.kMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.vMat = torch.nn.Linear(emb_dim, heads * dims, bias=False)
		self.oMat = torch.nn.Linear(heads * dims, emb_dim, bias=False)
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		# return: (batch_size, seq_len, emb_dim)
		q = self.qMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		k = self.kMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		v = self.vMat(x).view(x.size(0), x.size(1), self.heads, -1).transpose(-3, -2)
		# attention: (batch_size, heads, seq_len, emb_dim)
		attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(-3, -2).contiguous()
		attention = self.oMat(attention.view(attention.size(0), attention.size(1), -1))
		return attention
class EvalLayer(torch.nn.Module):
	def __init__(self, emb_dim, heads, dims):
		super().__init__()
		self.heads = heads
		self.attention = EvalMaskedAttention(emb_dim, heads, dims)
		self.normalize = torch.nn.LayerNorm([emb_dim])
		self.linear = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))
		self.normalize2 = torch.nn.LayerNorm([emb_dim])
	def forward(self, x):
		# x: (batch_size, seq_len, emb_dim)
		attention = self.attention(x)
		norm = self.normalize(x + attention)
		lin = self.linear(norm)
		norm2 = self.normalize2(norm + lin)
		return norm2
class EvalNetwork(torch.nn.Module):
	def  __init__(self):
		super().__init__()
		self.emb = torch.nn.Embedding(model.emb.weight.size(0), 768)
		self.register_buffer("pos", (torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0) * torch.sin(torch.arange(int(sys.argv[4])).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim)) + (1 - torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0)) * torch.cos(torch.arange(int(sys.argv[4])).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim))))
		self.temp = [EvalLayer(self.emb.embedding_dim, 12, 64) for _ in range(12)]
		self.layers = torch.nn.Sequential(*self.temp)
		self.linear3 = torch.nn.Linear(self.emb.embedding_dim, model.emb.weight.size(0), bias=False)
		self.emb.weight = self.linear3.weight
	def forward(self, x):
		embed = self.emb(x) + self.pos[:x.size(-1), :].unsqueeze(0)
		layer = self.layers(embed)
		lin3 = self.linear3(layer)
		return torch.log_softmax(lin3, dim=-1)
class BufferStart(torch.nn.Module):
	def  __init__(self):
		super().__init__()
		self.emb = torch.nn.Embedding(model.emb.weight.size(0), 768)
		self.register_buffer("pos", (torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0) * torch.sin(torch.arange(int(sys.argv[4])).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim)) + (1 - torch.fmod(torch.arange(self.emb.embedding_dim), 2).unsqueeze(0)) * torch.cos(torch.arange(int(sys.argv[4])).unsqueeze(1) / torch.pow(torch.tensor(10000), torch.floor(torch.arange(self.emb.embedding_dim) / 2).unsqueeze(0) * 2 / self.emb.embedding_dim))))
		self.temp = [EvalLayer(self.emb.embedding_dim, 12, 64) for _ in range(12)]
		self.layers = torch.nn.Sequential(*self.temp)
		self.linear3 = torch.nn.Linear(self.emb.embedding_dim, model.emb.weight.size(0), bias=False)
	def forward(self, x):
		embed = x + self.pos[:x.size(-2), :].unsqueeze(0)
		layer = self.layers(embed)
		lin3 = self.linear3(layer)
		return torch.log_softmax(lin3, dim=-1)
evalModel = EvalNetwork()
evalModel.load_state_dict(model._orig_mod.state_dict())
startGen = BufferStart()
del model
for param in evalModel.parameters():
	param.requires_grad = False
evalModel.to("cuda")
torch.jit.trace(evalModel, torch.randint(evalModel.emb.weight.size(0), (1, int(sys.argv[4]))).to("cuda")).save("model.pt")
evalModel = torch.compile(evalModel)
startGen.load_state_dict(evalModel._orig_mod.state_dict())
for param in startGen.parameters():
	param.requires_grad = False
cBuffer = torch.multinomial(torch.exp(startGen(evalModel.emb.weight.mean(0, keepdim=True).to("cpu")).view(-1)), 1).to("cuda").unsqueeze(-1)
vocab = []
with open("vocab.csv", encoding="utf-8", errors="replace") as f:
	vocab = next(csv.reader(f), [])
while True:
	print(vocab[cBuffer[0, -1]], end="")
	if cBuffer.size(-1) < int(sys.argv[4]):
		cBuffer = torch.cat((cBuffer, torch.multinomial(torch.exp(evalModel(cBuffer)[0, -1, :]), 1).unsqueeze(-1)), -1)
	else:
		cBuffer = torch.cat((cBuffer[:, 1:], torch.multinomial(torch.exp(evalModel(cBuffer)[0, -1, :]), 1).unsqueeze(-1)), -1)